# P2 ETF Contrastive Learning Engine (ETF‑CLR)

**Regime‑invariant ETF embeddings via SimCLR‑style contrastive learning with k‑NN prediction.**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-CONTRASTIVE-LEARNING/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-CONTRASTIVE-LEARNING/actions/workflows/daily_run.yml)

## Overview

This engine learns robust, regime‑invariant representations of ETF return windows using self‑supervised contrastive learning (SimCLR). A lightweight 1D‑CNN encoder is pre‑trained to maximize agreement between augmented views of the same market period. At inference time, the current market embedding is compared to historical embeddings using k‑nearest neighbors, and the ETF with the best average forward return among the most similar past windows is selected.

**Key Features:**
- **Self‑Supervised Pre‑training**: No labels required; learns purely from return dynamics.
- **Regime‑Invariant Embeddings**: Augmentations (noise, scaling, time shift, dropout) force the encoder to capture stable relationships.
- **k‑NN Prediction**: Transparent, interpretable selection based on historical similarity.
- **Three Universes**: FI/Commodities, Equity Sectors, and Combined.
- **Global & Adaptive Training**: Fixed 80/10/10 split and post‑change‑point adaptive windows.

## Data

- **Input**: `P2SAMAPA/fi-etf-macro-signal-master-data` (master_data.parquet)
- **Output**: `P2SAMAPA/p2-etf-contrastive-learning-results`

## Usage

```bash
pip install -r requirements.txt
python contrastive_trainer.py   # Runs pre‑training and inference
streamlit run streamlit_app.py
Configuration
All parameters are in config.py:

WINDOW_SIZE: lookback days for each sample (default 60)

EMBEDDING_DIM: encoder output dimension (default 32)

TEMPERATURE: InfoNCE temperature (default 0.07)

CONTRASTIVE_EPOCHS: pre‑training epochs (default 50)

K_NEIGHBORS: number of neighbors for prediction (default 20)
