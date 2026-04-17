import streamlit as st
import numpy as np
from datetime import datetime
import config
from push_results import load_latest_result
from us_calendar import next_trading_day

st.set_page_config(page_title="Contrastive ETF Engine", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header { font-size: 2.5rem; font-weight: 600; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
.sub-header { font-size: 1rem; color: #6B7280; margin-bottom: 2rem; font-weight: 400; }
.card { background-color: #FFFFFF; border-radius: 16px; padding: 1.8rem 2rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #F0F2F5; }
.ticker-large { font-size: 4.5rem; font-weight: 700; margin: 0; line-height: 1.1; color: #111827; }
.pred-return { font-size: 1.4rem; color: #059669; font-weight: 500; margin: 0.3rem 0 0.5rem 0; }
.meta-text { color: #6B7280; font-size: 0.9rem; }
.source-badge { background-color: #F3F4F6; display: inline-block; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500; color: #374151; margin-top: 0.5rem; }
.section-divider { margin: 1.5rem 0; border-top: 1px solid #E5E7EB; }
.metric-label { font-size: 0.85rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.03em; font-weight: 500; }
.metric-value { font-size: 1.3rem; font-weight: 600; color: #111827; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">ETF‑CLR — Contrastive Learning ETF Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SimCLR · Regime‑Invariant Embeddings · k‑NN Prediction</div>', unsafe_allow_html=True)

tab_fi, tab_eq, tab_comb = st.tabs(["FI/Commodities", "Equity Sectors", "Combined Universe"])
results = load_latest_result()

def format_pct(v): return f"{v*100:.1f}%" if v is not None and not np.isnan(v) else "—"
def format_num(v, d=2): return f"{v:.{d}f}" if v is not None and not np.isnan(v) else "—"

def display_metrics(metrics):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-label">ANN RETURN</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(metrics.get("ann_return"))} ({format_pct(metrics.get("cum_return"))})</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-label">ANN VOL</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(metrics.get("ann_vol"))}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-label">SHARPE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_num(metrics.get("sharpe"))}</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-label">MAX DD</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(metrics.get("max_dd"))}</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-label">HIT RATE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(metrics.get("hit_rate"))}</div>', unsafe_allow_html=True)

def display_card(data, mode="global"):
    if not data or not data.get("ticker"):
        st.info("⏳ Waiting for training output...")
        return
    ticker = data["ticker"]
    pred_return = data.get("pred_return")
    similarity = data.get("similarity_score")
    metrics = data.get("metrics", {})
    next_day = next_trading_day(datetime.utcnow())
    gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f'<div class="ticker-large">{ticker}</div>', unsafe_allow_html=True)
        if pred_return is not None:
            st.markdown(f'<div class="pred-return">Predicted Return: {pred_return*100:.2f}%</div>', unsafe_allow_html=True)
        if similarity is not None:
            st.markdown(f'<div class="meta-text">Similarity Score: {similarity:.3f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta-text">Signal for {next_day.strftime("%Y-%m-%d")} · Generated {gen_time}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="source-badge">Source: {mode} Training</div>', unsafe_allow_html=True)
    with col2:
        if mode == "Adaptive":
            st.markdown(f'<div class="meta-text">Change Point: {data.get("change_point_date", "—")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="meta-text">Lookback: {data.get("lookback_days", 0)} days</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meta-text">Test: {data.get("test_start", "")} → {data.get("test_end", "")} ({metrics.get("n_days", "—")} days)</div>', unsafe_allow_html=True)
    display_metrics(metrics)
    st.markdown('</div>', unsafe_allow_html=True)

for tab, key in [(tab_fi, "fi"), (tab_eq, "equity"), (tab_comb, "combined")]:
    with tab:
        st.subheader(key.capitalize())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Global Training")
            display_card(results.get(key, {}).get("global", {}), "Global")
        with col2:
            st.markdown("### Adaptive Window")
            display_card(results.get(key, {}).get("adaptive", {}), "Adaptive")
