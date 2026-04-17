"""
Contrastive pre‑training and k‑NN inference for ETF selection.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import config
from data_manager import load_master_data, prepare_data, get_universe_returns
from augmentations import apply_augmentations
from encoder import ContrastiveModel
from change_point_detector import universe_adaptive_start_date
from push_results import push_daily_result


def info_nce_loss(z_i, z_j, temperature):
    """InfoNCE loss for a batch of positive pairs."""
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # [2N, dim]
    sim = torch.mm(z, z.t()) / temperature  # cosine similarity (since z is normalized)
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)
    mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device)
    mask = mask.fill_diagonal_(False)
    for i in range(batch_size):
        mask[i, i + batch_size] = False
        mask[i + batch_size, i] = False
    negative_samples = sim[mask].reshape(2 * batch_size, -1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


def create_samples(returns: pd.DataFrame, window_size: int, scaler: StandardScaler = None):
    """Create sliding windows from returns DataFrame."""
    data = returns.values
    if scaler is None:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)

    samples = []
    for i in range(len(data) - window_size + 1):
        samples.append(data[i:i+window_size])
    return np.array(samples), scaler


def train_contrastive(returns: pd.DataFrame, model: ContrastiveModel, device: str):
    """Pre‑train the encoder using SimCLR on the full history."""
    window_size = config.WINDOW_SIZE
    samples, scaler = create_samples(returns, window_size)
    samples = torch.tensor(samples, dtype=torch.float32)

    dataset = TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"  Pre‑training for {config.CONTRASTIVE_EPOCHS} epochs...")
    for epoch in range(config.CONTRASTIVE_EPOCHS):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Two augmented views
            view1 = apply_augmentations(batch, strength="medium")
            view2 = apply_augmentations(batch, strength="medium")

            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = info_nce_loss(z1, z2, config.TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | Avg Loss: {total_loss / len(dataset):.6f}")

    return model, scaler


def compute_embeddings(returns: pd.DataFrame, model: ContrastiveModel, scaler: StandardScaler, device: str):
    """Compute embeddings for all windows in the dataset."""
    window_size = config.WINDOW_SIZE
    samples, _ = create_samples(returns, window_size, scaler)
    samples = torch.tensor(samples, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        embeddings, _ = model(samples)
    return embeddings.cpu().numpy()


def predict_top_etf(embeddings: np.ndarray, returns: pd.DataFrame, tickers: list,
                    current_embedding: np.ndarray, k: int = 20):
    """
    Given current market embedding, find k‑nearest historical windows
    and select the ETF with the best average forward return.
    """
    # Compute cosine similarities
    sims = np.dot(embeddings, current_embedding)  # embeddings are normalized
    top_k_idx = np.argsort(sims)[-k:]

    # Forward returns: shift(-1) to get next‑day return
    forward_returns = returns.shift(-1).iloc[config.WINDOW_SIZE-1:-1].values

    # Average forward return per ETF over the k‑nearest windows
    avg_returns = forward_returns[top_k_idx].mean(axis=0)
    best_idx = np.argmax(avg_returns)
    best_ticker = tickers[best_idx]
    pred_return = avg_returns[best_idx]
    similarity_score = sims[top_k_idx].mean()

    return best_ticker, pred_return, similarity_score


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}
    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret_series.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    hit_rate = (ret_series > 0).mean()
    cum_return = (1 + ret_series).prod() - 1
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_dd": max_dd, "hit_rate": hit_rate, "cum_return": cum_return,
        "n_days": len(ret_series)
    }


def train_global(universe: str, returns: pd.DataFrame) -> dict:
    print(f"\n--- Global Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    total_days = len(returns)
    train_end = int(total_days * config.TRAIN_RATIO)
    val_end = train_end + int(total_days * config.VAL_RATIO)

    train_ret = returns.iloc[:train_end]
    test_ret = returns.iloc[val_end:]

    input_dim = len(tickers)
    model = ContrastiveModel(input_dim, config.WINDOW_SIZE, config.EMBEDDING_DIM, config.PROJECTION_DIM)
    model.to(config.DEVICE)

    model, scaler = train_contrastive(train_ret, model, config.DEVICE)

    # Compute embeddings for all training windows
    train_embeddings = compute_embeddings(train_ret, model, scaler, config.DEVICE)

    # Current market window (last available)
    current_window = returns.iloc[-config.WINDOW_SIZE:].values
    current_window = scaler.transform(current_window)
    current_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        current_emb, _ = model(current_tensor)
    current_emb = current_emb.cpu().numpy().squeeze()

    top_etf, pred_return, similarity = predict_top_etf(
        train_embeddings, train_ret, tickers, current_emb, config.K_NEIGHBORS
    )

    metrics = evaluate_etf(top_etf, test_ret)
    print(f"  Selected ETF: {top_etf}, Predicted Return: {pred_return*100:.2f}%, Similarity: {similarity:.3f}")
    return {
        "ticker": top_etf,
        "pred_return": pred_return,
        "similarity_score": similarity,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def train_adaptive(universe: str, returns: pd.DataFrame) -> dict:
    print(f"\n--- Adaptive Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    cp_date = universe_adaptive_start_date(returns)
    print(f"  Adaptive window starts: {cp_date.date()}")

    end_date = returns.index[-1] - pd.Timedelta(days=config.MIN_TEST_DAYS)
    if end_date <= cp_date:
        end_date = returns.index[-1] - pd.Timedelta(days=10)
    train_mask = (returns.index >= cp_date) & (returns.index <= end_date)
    train_ret = returns.loc[train_mask]
    test_ret = returns.loc[returns.index > end_date]

    if len(train_ret) < config.MIN_TRAIN_DAYS:
        print(f"  Insufficient training days. Falling back to global.")
        return train_global(universe, returns)

    input_dim = len(tickers)
    model = ContrastiveModel(input_dim, config.WINDOW_SIZE, config.EMBEDDING_DIM, config.PROJECTION_DIM)
    model.to(config.DEVICE)

    model, scaler = train_contrastive(train_ret, model, config.DEVICE)

    train_embeddings = compute_embeddings(train_ret, model, scaler, config.DEVICE)

    current_window = returns.iloc[-config.WINDOW_SIZE:].values
    current_window = scaler.transform(current_window)
    current_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        current_emb, _ = model(current_tensor)
    current_emb = current_emb.cpu().numpy().squeeze()

    top_etf, pred_return, similarity = predict_top_etf(
        train_embeddings, train_ret, tickers, current_emb, config.K_NEIGHBORS
    )

    metrics = evaluate_etf(top_etf, test_ret) if len(test_ret) > 0 else {}
    lookback = (returns.index[-1] - cp_date).days
    print(f"  Selected ETF: {top_etf}, Predicted Return: {pred_return*100:.2f}%, Similarity: {similarity:.3f}")
    return {
        "ticker": top_etf,
        "pred_return": pred_return,
        "similarity_score": similarity,
        "metrics": metrics,
        "change_point_date": cp_date.strftime("%Y-%m-%d"),
        "lookback_days": lookback,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


def run_training():
    print("Loading data...")
    df_raw = load_master_data()
    df = prepare_data(df_raw)

    all_results = {}
    for universe in ["fi", "equity", "combined"]:
        print(f"\n{'='*50}\nProcessing {universe.upper()}\n{'='*50}")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            continue
        global_res = train_global(universe, returns)
        adaptive_res = train_adaptive(universe, returns)
        all_results[universe] = {"global": global_res, "adaptive": adaptive_res}
    return all_results


if __name__ == "__main__":
    output = run_training()
    if config.HF_TOKEN:
        push_daily_result(output)
    else:
        print("HF_TOKEN not set.")
