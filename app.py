import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta, timezone
from scipy import stats
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CRYPTO ARBITRAGE BOT — Consolidated Streamlit App
# Modules: 1) Multi-Exchange Detection  2) Execution Simulation
#          3) Naive vs. Realistic Backtest  4) Robustness & Risk
# ============================================================

st.set_page_config(page_title="Crypto Arbitrage Bot", layout="wide")

PAIRS = {
    "BTC": {"binance": "BTCUSDT", "coinbase": "BTC-USD", "kraken": "XBTUSDT"},
    "ETH": {"binance": "ETHUSDT", "coinbase": "ETH-USD", "kraken": "ETHUSDT"},
    "SOL": {"binance": "SOLUSDT", "coinbase": "SOL-USD", "kraken": "SOLUSDT"},
}
QUOTE_CURRENCY = {"binance": "USDT", "coinbase": "USD", "kraken": "USDT"}
EXCHANGE_PAIRS = [("binance", "coinbase"), ("binance", "kraken"), ("coinbase", "kraken")]

TAKER_FEES = {"binance": 0.0010, "coinbase": 0.0060, "kraken": 0.0026}
FEE_SCENARIOS = {
    "Base taker (retail)": {"binance": 0.0010, "coinbase": 0.0060, "kraken": 0.0026},
    "High-volume VIP taker": {"binance": 0.00017, "coinbase": 0.0005, "kraken": 0.0010},
    "Maker (limit orders, fill not guaranteed)": {"binance": 0.0000, "coinbase": 0.0040, "kraken": 0.0000},
}

LATENCY_SECONDS = 3
TRADE_SIZE_USD = 10000
P_LEG_FAILURE = 0.05
UNWIND_DELAY_SECONDS = 10

# ------------------------------------------------------------
# Data fetch functions (identical logic to notebook, bug fixes included)
# ------------------------------------------------------------
def fetch_binance(symbol, start, end):
    url = "https://api.binance.us/api/v3/klines"
    all_rows = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": "1m", "startTime": cursor, "limit": 1000}
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
        except Exception:
            break
        if not isinstance(data, list) or len(data) == 0:
            break
        all_rows.extend(data)
        cursor = data[-1][6] + 1
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "close"])
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]]


def fetch_coinbase(product_id, start, end):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    all_rows = []
    chunk_start = start
    chunk_size = timedelta(minutes=299)
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_size, end)
        params = {"granularity": 60, "start": chunk_start.isoformat(), "end": chunk_end.isoformat()}
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
        except Exception:
            data = None
        if isinstance(data, list):
            all_rows.extend(data)
        chunk_start = chunk_end
        time.sleep(0.3)
    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "close"])
    df = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]].sort_values("timestamp")


def fetch_kraken(pair, start, end):
    url = "https://api.kraken.com/0/public/OHLC"
    all_rows = []
    since = int(start.timestamp())
    end_ts = int(end.timestamp())
    attempts = 0
    while since < end_ts and attempts < 10:
        params = {"pair": pair, "interval": 1, "since": since}
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
        except Exception:
            attempts += 1
            time.sleep(5)
            continue
        if data.get("error"):
            attempts += 1
            time.sleep(5)
            continue
        result_key = [k for k in data["result"].keys() if k != "last"][0]
        rows = data["result"][result_key]
        if not rows:
            break
        all_rows.extend(rows)
        since = data["result"]["last"]
        time.sleep(1.5)
        if rows[-1][0] >= end_ts:
            break
    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "close"])
    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]]


@st.cache_data(ttl=900, show_spinner=False)
def build_aligned_panel(lookback_days):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    price_series = {}
    fetch_log = {}
    for asset, symbols in PAIRS.items():
        price_series[(asset, "binance")] = fetch_binance(symbols["binance"], start_time, end_time)
        price_series[(asset, "coinbase")] = fetch_coinbase(symbols["coinbase"], start_time, end_time)
        price_series[(asset, "kraken")] = fetch_kraken(symbols["kraken"], start_time, end_time)

    for key, df in price_series.items():
        fetch_log[key] = len(df)

    common_grid = pd.date_range(
        pd.Timestamp(start_time).floor("min"), pd.Timestamp(end_time).floor("min"),
        freq="1min", tz="UTC"
    )
    aligned = pd.DataFrame(index=common_grid)
    for (asset, exch), df in price_series.items():
        if len(df) == 0:
            continue
        df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
        s = df.set_index("timestamp")["close"].reindex(common_grid, method="nearest", tolerance=pd.Timedelta("2min"))
        aligned[f"{asset}_{exch}"] = s
    aligned = aligned.dropna(how="all")
    return aligned, fetch_log


@st.cache_data(ttl=900, show_spinner=False)
def detect_opportunities(_aligned, threshold_bps):
    detections = []
    for asset in PAIRS:
        for exch_a, exch_b in EXCHANGE_PAIRS:
            col_a, col_b = f"{asset}_{exch_a}", f"{asset}_{exch_b}"
            if col_a not in _aligned.columns or col_b not in _aligned.columns:
                continue
            gap_bps = (_aligned[col_a] - _aligned[col_b]) / _aligned[col_b] * 10000
            gap_bps = gap_bps.dropna()
            flagged = gap_bps[gap_bps.abs() > threshold_bps]
            for ts, val in flagged.items():
                detections.append({
                    "timestamp": ts, "asset": asset,
                    "exchange_a": exch_a, "exchange_b": exch_b, "gap_bps": val,
                    "cross_quote_currency": QUOTE_CURRENCY[exch_a] != QUOTE_CURRENCY[exch_b],
                })
    detections_df = pd.DataFrame(detections)
    if len(detections_df) == 0:
        return detections_df, detections_df
    clean = detections_df[~detections_df["cross_quote_currency"]].copy()
    cross_quote = detections_df[detections_df["cross_quote_currency"]].copy()
    return clean, cross_quote


def simulate_execution(detections_df, aligned):
    if len(detections_df) == 0:
        return pd.DataFrame()
    results = []
    for _, row in detections_df.iterrows():
        asset, exch_a, exch_b = row["asset"], row["exchange_a"], row["exchange_b"]
        col_a, col_b = f"{asset}_{exch_a}", f"{asset}_{exch_b}"
        detect_ts = row["timestamp"]
        if detect_ts not in aligned.index:
            continue
        price_a_detect = aligned.loc[detect_ts, col_a]
        price_b_detect = aligned.loc[detect_ts, col_b]

        exec_ts = detect_ts + pd.Timedelta(seconds=LATENCY_SECONDS)
        nearest_exec_ts = aligned.index[aligned.index.get_indexer([exec_ts], method="nearest")[0]]
        price_a_exec = aligned.loc[nearest_exec_ts, col_a]
        price_b_exec = aligned.loc[nearest_exec_ts, col_b]
        if pd.isna(price_a_exec) or pd.isna(price_b_exec):
            continue

        if price_a_detect < price_b_detect:
            buy_exch, sell_exch = exch_a, exch_b
            buy_price, sell_price = price_a_exec, price_b_exec
        else:
            buy_exch, sell_exch = exch_b, exch_a
            buy_price, sell_price = price_b_exec, price_a_exec

        units = TRADE_SIZE_USD / buy_price
        slippage_bps = 2 + (TRADE_SIZE_USD / 50000) * 3
        buy_price_slipped = buy_price * (1 + slippage_bps / 10000)
        sell_price_slipped = sell_price * (1 - slippage_bps / 10000)

        buy_fee = TAKER_FEES[buy_exch] * units * buy_price_slipped
        sell_fee = TAKER_FEES[sell_exch] * units * sell_price_slipped

        gross_pnl = (sell_price - buy_price) * units
        net_pnl = (sell_price_slipped - buy_price_slipped) * units - buy_fee - sell_fee

        results.append({
            "timestamp": detect_ts, "asset": asset,
            "buy_exchange": buy_exch, "sell_exchange": sell_exch,
            "gross_gap_bps": row["gap_bps"], "gross_pnl": gross_pnl, "net_pnl": net_pnl,
            "fees_paid": buy_fee + sell_fee, "slippage_bps": slippage_bps,
        })
    return pd.DataFrame(results)


def rerun_with_fees(sim_df, fee_schedule):
    recomputed = sim_df.copy()
    pre_fee_pnl = recomputed["net_pnl"] + recomputed["fees_paid"]
    new_fees = recomputed.apply(
        lambda r: (fee_schedule[r["buy_exchange"]] + fee_schedule[r["sell_exchange"]]) * TRADE_SIZE_USD, axis=1
    )
    recomputed["net_pnl_new"] = pre_fee_pnl - new_fees
    return recomputed


def simulate_execution_risk(sim_df, aligned, seed=42):
    if len(sim_df) == 0:
        return np.array([])
    rng = np.random.RandomState(seed)
    results = []
    for _, row in sim_df.iterrows():
        fails = rng.random_sample() < P_LEG_FAILURE
        if not fails:
            results.append(row["net_pnl"])
            continue
        asset = row["asset"]
        which_failed = rng.choice(["buy", "sell"])
        stuck_exchange = row["sell_exchange"] if which_failed == "buy" else row["buy_exchange"]
        col = f"{asset}_{stuck_exchange}"
        unwind_ts = row["timestamp"] + pd.Timedelta(seconds=UNWIND_DELAY_SECONDS)
        if col not in aligned.columns:
            results.append(row["net_pnl"])
            continue
        try:
            nearest_ts = aligned.index[aligned.index.get_indexer([unwind_ts], method="nearest")[0]]
            price_at_unwind = aligned.loc[nearest_ts, col]
            price_at_detect = aligned.loc[row["timestamp"], col] if row["timestamp"] in aligned.index else np.nan
            if pd.isna(price_at_unwind) or pd.isna(price_at_detect):
                results.append(row["net_pnl"])
                continue
            units = TRADE_SIZE_USD / price_at_detect
            direction = 1 if which_failed == "sell" else -1
            stuck_pnl = direction * (price_at_unwind - price_at_detect) * units
            fee = TAKER_FEES[stuck_exchange] * TRADE_SIZE_USD * 2
            results.append(stuck_pnl - fee)
        except Exception:
            results.append(row["net_pnl"])
    return np.array(results)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Crypto Arbitrage Bot")
module = st.sidebar.radio(
    "Module",
    ["1. Detection", "2. Execution Simulation", "3. Naive vs. Realistic", "4. Robustness & Risk"]
)
lookback_days = st.sidebar.slider("Lookback window (days)", 1, 5, 3)
threshold_bps = st.sidebar.slider("Detection threshold (bps)", 5, 30, 10)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Real exchange data (Binance.US, Coinbase, Kraken public APIs). "
    "Kraken's 1-minute candle history is limited to ~12 hours by the API itself. "
    "Coinbase quotes in USD; Binance/Kraken in USDT — comparisons are split into "
    "'clean' (same quote currency) and 'cross-quote' (USD vs USDT) tracks throughout."
)

with st.spinner("Fetching live exchange data and aligning..."):
    aligned, fetch_log = build_aligned_panel(lookback_days)

with st.spinner("Detecting cross-exchange price gaps..."):
    clean_detections, cross_quote_detections = detect_opportunities(aligned, threshold_bps)

with st.spinner("Simulating execution..."):
    clean_sim = simulate_execution(clean_detections, aligned)
    cross_quote_sim = simulate_execution(cross_quote_detections, aligned)

# ============================================================
# MODULE 1
# ============================================================
if module == "1. Detection":
    st.header("Module 1 — Multi-Exchange Price Feed & Arbitrage Detection")

    st.subheader("Data Fetch Summary")
    fetch_df = pd.DataFrame([
        {"asset": k[0], "exchange": k[1], "rows": v} for k, v in fetch_log.items()
    ])
    st.dataframe(fetch_df, use_container_width=True)
    st.caption("Kraken rows will be much lower than Binance/Coinbase — a real API limitation "
               "(1-minute candles only retained for ~12 hours), not a bug.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total detections", len(clean_detections) + len(cross_quote_detections))
    c2.metric("Clean (same-quote)", len(clean_detections))
    c3.metric("Cross-quote (USD vs USDT)", len(cross_quote_detections))

    st.info(
        f"{len(clean_detections)} of {len(clean_detections) + len(cross_quote_detections)} detections "
        "are genuine same-currency crypto price comparisons. The remainder involve a USD/USDT quote "
        "mismatch and may partly or wholly reflect stablecoin basis rather than crypto mispricing."
    )

    for label, df in [("Clean", clean_detections), ("Cross-quote", cross_quote_detections)]:
        if len(df) > 0:
            st.subheader(f"{label} — Gap Size Distribution (bps)")
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.hist(df["gap_bps"].abs(), bins=30, color="#4C72B0", edgecolor="white")
            ax.set_xlabel("Gap (bps)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ============================================================
# MODULE 2
# ============================================================
elif module == "2. Execution Simulation":
    st.header("Module 2 — Execution Simulation")
    st.caption(
        f"Latency: {LATENCY_SECONDS}s · Trade size: ${TRADE_SIZE_USD:,} · "
        "Fees: real published taker schedules · Slippage: proxy (no free historical order-book depth available)"
    )

    for label, sim_df in [("Clean (same-quote-currency)", clean_sim), ("Cross-quote (USD vs USDT)", cross_quote_sim)]:
        st.subheader(label)
        if len(sim_df) == 0:
            st.write("No opportunities simulated.")
            continue
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Opportunities", len(sim_df))
        c2.metric("Mean gross P&L", f"${sim_df['gross_pnl'].mean():.2f}")
        c3.metric("Mean net P&L", f"${sim_df['net_pnl'].mean():.2f}")
        c4.metric("% profitable net", f"{(sim_df['net_pnl'] > 0).mean():.1%}")

# ============================================================
# MODULE 3
# ============================================================
elif module == "3. Naive vs. Realistic":
    st.header("Module 3 — Net Profitability Backtest (Naive vs. Realistic)")

    for label, sim_df in [("Clean (same-quote-currency)", clean_sim), ("Cross-quote (USD vs USDT)", cross_quote_sim)]:
        st.subheader(label)
        if len(sim_df) == 0:
            st.write("No data.")
            continue
        naive_total = sim_df["gross_pnl"].sum()
        realistic_total = sim_df["net_pnl"].sum()
        t_stat, p_val = stats.ttest_1samp(sim_df["net_pnl"], 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Naive total P&L", f"${naive_total:,.0f}")
        c2.metric("Realistic total P&L", f"${realistic_total:,.0f}")
        c3.metric("Naive % profitable", "100%")
        c4.metric("Realistic % profitable", f"{(sim_df['net_pnl'] > 0).mean():.0%}")
        st.caption(f"One-sample t-test (net P&L vs 0): t={t_stat:.2f}, p={p_val:.2e}")

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.bar(["Naive (gross)", "Realistic (net)"], [naive_total, realistic_total],
               color=["#55A868", "#C44E52" if realistic_total < 0 else "#55A868"])
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_ylabel("Total P&L ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# MODULE 4
# ============================================================
elif module == "4. Robustness & Risk":
    st.header("Module 4 — Robustness & Risk Factors")

    st.subheader("Fee-Tier Sensitivity")
    st.caption("VIP tiers require high trading volume to qualify; maker orders have no fill guarantee (see execution risk below).")
    fee_rows = []
    for track_name, sim_df in [("Clean", clean_sim), ("Cross-quote", cross_quote_sim)]:
        if len(sim_df) == 0:
            continue
        for scenario_name, fees in FEE_SCENARIOS.items():
            recomputed = rerun_with_fees(sim_df, fees)
            fee_rows.append({
                "Track": track_name, "Fee Scenario": scenario_name,
                "Mean Net P&L": recomputed["net_pnl_new"].mean(),
                "% Profitable": (recomputed["net_pnl_new"] > 0).mean(),
            })
    if fee_rows:
        fee_df = pd.DataFrame(fee_rows)
        st.dataframe(
            fee_df.style.format({"Mean Net P&L": "${:.2f}", "% Profitable": "{:.1%}"}),
            use_container_width=True
        )

    st.subheader("Execution Risk (leg failure)")
    st.caption(f"{P_LEG_FAILURE:.0%} chance either leg fails to fill; stuck position unwound after {UNWIND_DELAY_SECONDS}s.")
    for track_name, sim_df in [("Clean", clean_sim), ("Cross-quote", cross_quote_sim)]:
        if len(sim_df) == 0:
            continue
        with_risk = simulate_execution_risk(sim_df, aligned)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{track_name}: mean net P&L (no exec. risk)", f"${sim_df['net_pnl'].mean():.2f}")
        c2.metric(f"{track_name}: mean net P&L (with exec. risk)", f"${with_risk.mean():.2f}")
        c3.metric(f"{track_name}: worst simulated outcome", f"${with_risk.min():.2f}")

    st.subheader("Daily Stability")
    for track_name, sim_df in [("Clean", clean_sim), ("Cross-quote", cross_quote_sim)]:
        if len(sim_df) == 0:
            continue
        daily = sim_df.copy()
        daily["date"] = daily["timestamp"].dt.date
        daily_summary = daily.groupby("date")["net_pnl"].agg(["mean", "count"])
        st.write(f"**{track_name}**")
        st.dataframe(daily_summary.style.format({"mean": "${:.2f}"}), use_container_width=True)
        st.caption(f"Days with positive mean net P&L: {(daily_summary['mean'] > 0).sum()} / {len(daily_summary)}")