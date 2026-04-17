"""
Configuration for P2-ETF-CONTRASTIVE-LEARNING.
"""
import os

# Hugging Face configuration
HF_INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_INPUT_FILE = "master_data.parquet"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-contrastive-learning-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Universes
FI_COMMODITY_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "XME"]
COMBINED_TICKERS = FI_COMMODITY_TICKERS + EQUITY_TICKERS

BENCHMARK_FI = "AGG"
BENCHMARK_EQ = "SPY"

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_TRAIN_DAYS = 252 * 2
MIN_TEST_DAYS = 63
TRADING_DAYS_PER_YEAR = 252

# Change Point Detection (for adaptive window)
CP_PENALTY = 3.0
CP_MODEL = "l2"
CP_MIN_DAYS_BETWEEN = 20
CP_CONSENSUS_FRACTION = 0.5

# Contrastive learning parameters
WINDOW_SIZE = 60                # lookback days for each sample
EMBEDDING_DIM = 32              # output dimension of encoder
PROJECTION_DIM = 64             # projection head dimension
TEMPERATURE = 0.07              # InfoNCE temperature
BATCH_SIZE = 128                # training batch size
CONTRASTIVE_EPOCHS = 50         # pre‑training epochs
LEARNING_RATE = 0.001
K_NEIGHBORS = 20                # number of neighbors for k‑NN prediction
DEVICE = "cpu"
