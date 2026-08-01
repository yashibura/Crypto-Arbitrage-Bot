# Crypto Arbitrage Bot

A cross-exchange cryptocurrency arbitrage detection and execution-simulation engine built in Python (requests/numpy/scipy for all core computation — real exchange data via public REST APIs, no pre-built arbitrage or exchange-connectivity libraries), completing an eighth project alongside the Options Pricing Engine, VaR Calculator, Statistical Arbitrage BackTester, HF Market Simulator, Portfolio Optimizer, Factor Model Portfolio Builder, and Machine Learning Alpha Model by asking a question that combines two earlier themes in a new setting: **does a genuine price discrepancy for the same asset across different exchanges actually translate into profit once real execution costs — latency, fees, and slippage — are applied?**

**Live app:** [cryptoarbitragebot.streamlit.app](https://cryptoarbitragebot.streamlit.app)

---

## Overview

The HF Market Simulator studied execution mechanics inside a single, synthetic order book. The Statistical Arbitrage BackTester found mispricings *within* one exchange's asset universe. This project combines both ideas in a new setting: real price data for the same three assets (BTC, ETH, SOL) pulled live from three different real exchanges (Binance, Coinbase, Kraken), testing whether a detected cross-exchange price gap survives the same "naive vs. realistic" scrutiny that the HF Market Simulator and Portfolio Optimizer applied elsewhere in this series. Unlike those single-exchange projects, this one also introduces a genuinely new risk: **execution risk**, the possibility that one leg of a two-exchange trade fails or fills late while the other doesn't — a risk with no equivalent when trading on a single venue.

---

## 1. Multi-Exchange Price Feed & Arbitrage Detection

### What it does
Pulls historical 1-minute OHLCV candles for BTC, ETH, and SOL from Binance, Coinbase, and Kraken's public REST APIs (no API key required), aligns all three exchanges' data onto a common timestamp grid, and flags moments where the price gap between any two exchanges for the same asset exceeds a threshold (10 bps) as a raw, "gross" arbitrage opportunity — before any execution costs are applied.

### Why it's important
Every later module depends on this detection step being genuine and correctly aligned; a misaligned timestamp grid or a data quality issue here would silently corrupt every downstream profitability conclusion.

### A known, documented data limitation
Coinbase's public API quotes these pairs in USD; Binance and Kraken quote in USDT. Any detected "gap" therefore blends genuine cross-exchange crypto price discrepancy with the USDT/USD stablecoin basis for any comparison involving Coinbase. Rather than assume this away, every detection is tagged with its quote currencies and split into two separate, honestly-labeled tracks carried through every subsequent module: **"clean"** (same quote currency on both exchanges — a genuine crypto price comparison) and **"cross-quote"** (USD vs. USDT — may partly or wholly reflect stablecoin basis rather than crypto mispricing).

### A second known data limitation
Kraken's public OHLC endpoint at 1-minute granularity only retains roughly the most recent 12 hours of candle history, regardless of how far back a request asks for — a genuine, documented API constraint, not a pagination bug. This means every Kraken-involving comparison has meaningfully less overlapping history (and therefore less statistical power) than the Binance-Coinbase comparison, which spans the full multi-day lookback window.

### Real bugs found and corrected during development
- **Binance geo-block**: `api.binance.com` returned empty data for every request, traced to Binance blocking API access from US-based IPs (where Google Colab's servers run) — fixed by switching to `api.binance.us`, which serves equivalent public market data without the geo-restriction.
- **A `datetime`/`Timestamp` method mismatch**: code called `.floor("min")` on a plain Python `datetime` object, which doesn't have that method (it belongs to pandas' `Timestamp`), causing an `AttributeError` — fixed by explicitly wrapping the values in `pd.Timestamp(...)` before calling `.floor()`.
- **Duplicate timestamps at pagination seams**: overlapping boundaries between successive paginated API calls (most likely Coinbase's chunked candle requests) produced a duplicate timestamp in at least one price series, which caused `reindex()` to fail outright, since pandas requires a unique index for that operation — fixed by explicitly deduplicating each series on timestamp (keeping the last occurrence) before setting it as the index.

### Real finding from validation
Of ~9,490 total detections, only **~0.9% (85 of 9,486 in the validated run) were "clean" same-currency comparisons** — the overwhelming majority of raw detected "opportunities" involve the USD/USDT quote mismatch and likely reflect stablecoin basis rather than genuine crypto mispricing. This is exactly the kind of striking, single-look result that needed a check before being treated as a finding — the same discipline as the false alarms caught in the HF Market Simulator and Portfolio Optimizer projects.

---

## 2. Execution Simulation

### What it does
For every detected opportunity, simulates what actually happens when trying to trade on it: a configurable latency delay (3 seconds) between detection and execution, real published taker fee schedules for both legs of the trade (buy on the cheaper exchange, sell on the pricier one), and a slippage proxy scaled by trade size — standing in for real historical order-book depth, which free public APIs do not provide at scale (a limitation directly analogous to the fundamentals data constraint in the Factor Model project, here applied to market depth instead).

### Why it's important
This is the module that turns Module 1's raw price gaps into an actual, cost-aware trading outcome — the same "what does it cost to actually act on this" question the HF Market Simulator's latency module asked in a single-exchange context.

### Real finding from validation
**Both tracks showed 0% of trades profitable net of costs** — mean net P&L of roughly -$28 (clean) and -$63 (cross-quote) per $10,000 attempted trade, against mean gross P&L of roughly +$13-15. The mechanism is straightforward: combined two-leg taker fees at standard retail tiers (roughly 26-70 bps depending on exchange pair) frequently exceed the entire detected price gap (typically only 13-15 bps), before slippage is even added.

---

## 3. Net Profitability Backtest (Naive vs. Realistic)

### What it does
Directly compares the "naive" view (treating every Module 1 detection as captured profit with zero costs — the implicit assumption of just looking at raw price gaps) against the "realistic" view (Module 2's actual net P&L), aggregated across the full detected set in each track, with a one-sample t-test confirming the realistic result is statistically distinguishable from zero.

### Why it's important
This is the same naive-vs-realistic comparison used in the HF Market Simulator and Portfolio Optimizer projects, applied here to cross-exchange arbitrage specifically — the module that directly quantifies the cost of treating a raw price gap as if it were free money.

### Real finding from validation
An exceptionally clean, statistically overwhelming result: naive backtesting suggested **$1,145 (clean) and $137,562 (cross-quote)** in total gains with 100% of trades "winning" (true by construction, since only positive gaps were ever flagged). Realistic execution flipped both decisively negative: **-$2,360 (clean) and -$588,142 (cross-quote)**, with **0% of trades profitable** and t-statistics so extreme (t = -78 and -858) that p-values round to zero. Unlike the marginal, ambiguous null results in the Factor Model and ML Alpha Model projects, this is an unambiguous, uniform result across every asset and every exchange-pair breakdown — no outlier is driving the conclusion.

---

## 4. Robustness & Risk Factors

### What it does
Tests three dimensions of the Module 3 result: (1) sensitivity to the assumed fee tier (base retail taker, high-volume VIP taker, and maker/limit orders), (2) execution risk specific to two-legged arbitrage — the possibility that one leg fails to fill while the other doesn't, requiring an unhedged position to be unwound later at an unknown price — and (3) whether the net-negative result holds consistently day by day, or is concentrated in one anomalous stretch.

### Why it's important
Module 3 established that arbitrage is unprofitable at standard retail fees — but that's an incomplete answer without checking whether a different, realistic fee structure (available to some traders, not others) changes the picture, and without accounting for the genuinely new risk this project introduces that none of the single-exchange projects needed to model.

### Real finding: fee-tier access is the real bottleneck, not a uniformly hopeless opportunity
At base retail taker fees, both tracks are unprofitable (0% win rate). At **high-volume VIP tier fees**, the picture flips meaningfully for the cross-quote track (66.8% profitable, mean +$1.71) — because Binance-Coinbase's combined VIP fee (~6.7 bps) drops below the typical ~14 bps detected gap — but stays largely unprofitable for the clean track (10.6% profitable), since Kraken's VIP fee still roughly matches the typical clean-track gap. At **maker (limit order) fees**, the clean track flips to 100% profitable (both Binance and Kraken charge near-zero maker fees), while cross-quote remains unprofitable (Coinbase's maker fee alone exceeds most gaps) — critically, maker orders require posting a limit order with no fill guarantee, directly implicating the next finding.

### Real finding: execution risk adds meaningful tail risk beyond the already-negative average
A modest 5% leg-failure probability roughly **doubled the standard deviation of outcomes** in both tracks and produced worst-case single-trade losses of -$52 (clean) and -$120 (cross-quote) — a real, distinct risk layer on top of an already negative expected outcome, and the exact trade-off implied by the maker-fee finding above (lower fees require limit orders, which carry fill risk).

### Real finding: the negative result is uniform across every day tested, not concentrated in an anomaly
0 of 2 days (clean) and 0 of 4 days (cross-quote) showed a positive mean net P&L — the conclusion holds consistently throughout the entire tested window, not driven by one unusual stretch.

### The honest, final conclusion
Genuine cross-exchange crypto arbitrage (the clean track) is unprofitable at retail fees and remains largely unprofitable even at institutional VIP tiers in this sample — detected price gaps are simply too small and infrequent relative to any realistic combined fee structure. The much larger cross-quote pattern (likely USD/USDT stablecoin basis) is a structurally different opportunity that *could* become viable, but only for a trader with access to high-volume VIP fee tiers. The practical takeaway is closer to **"this specific strategy is a fee-tier-access and infrastructure problem for a retail trader"** than a simple "arbitrage doesn't exist."

---

## What We Achieved

- **A complete, four-module cross-exchange arbitrage pipeline** built on real, live exchange data: multi-exchange detection with an honest clean/cross-quote split, cost-aware execution simulation, a naive-vs-realistic net profitability backtest, and a robustness pass covering fee-tier sensitivity, execution risk, and daily stability
- **Formal correctness discipline throughout**: every detected "opportunity" tagged and split by quote-currency validity rather than pooled indiscriminately; a documented, non-hidden data limitation (Kraken's 12-hour candle retention) accounted for explicitly in interpreting results; statistical significance testing (t-tests) rather than trusting raw totals
- **Real bugs found and corrected during development — documented, not hidden:**
  - A Binance geo-block silently returning empty data from US-based infrastructure, fixed by switching to the Binance.US endpoint
  - A `datetime`-vs-`Timestamp` method mismatch (`.floor()` called on a plain `datetime` object) causing an `AttributeError`, fixed with explicit `pd.Timestamp()` wrapping
  - Duplicate timestamps at API pagination seams breaking `reindex()`'s uniqueness requirement, fixed with explicit deduplication before indexing
  - A misidentified Kraken "rate limit" issue on first inspection that turned out to be a genuine, documented API data-retention limit (only ~12 hours of 1-minute candles available), correctly re-diagnosed rather than assumed to be a bug in the retry logic
- **An honest, statistically grounded set of conclusions**: only ~0.9% of raw detected opportunities are genuine same-currency crypto price comparisons, the rest likely reflecting USD/USDT stablecoin basis; both tracks show 0% profitability at retail fee tiers with p-values indistinguishable from zero; the cross-quote track becomes majority-profitable at institutional VIP fee tiers while the clean track largely does not; execution risk roughly doubles outcome volatility; and the unprofitable result holds uniformly across every day tested, with no exception
- **Deployed as a live, interactive web application** (Streamlit Community Cloud) making live outbound requests to three real exchange APIs, with adjustable lookback window and detection threshold, and all four modules navigable from a single sidebar

---

## Tech Stack

- **Core computation**: Python, numpy, scipy (stats)
- **Live exchange data**: `requests` against Binance.US, Coinbase Exchange, and Kraken public REST APIs (no API keys required)
- **Visualization**: matplotlib (non-interactive `Agg` backend for server-side rendering stability)
- **Web interface**: Streamlit
- **Deployment**: Streamlit Community Cloud
- **Development environment**: Google Colab

## Repository Structure
```
├── app.py                            # Consolidated Streamlit application (all 4 modules)
├── requirements.txt                  # Python dependencies
└── Crypto_Arbitrage_Bot.ipynb        # Development notebook: all 4 modules, diagnostics, and validation
```

The notebook documents the full development and validation process across all four modules, including every bug found and corrected along the way — most notably the Binance geo-block and the Kraken data-retention limitation initially misdiagnosed as a rate-limit bug. `app.py` consolidates the final, validated logic into a single deployable application that makes live requests to real exchange APIs on each run, cross-checked module-by-module against the notebook's own results after deployment — all four modules reproduced the central findings closely, with small numeric differences fully explained by each run fetching a fresh, current 3-day window rather than replaying an identical historical snapshot.
