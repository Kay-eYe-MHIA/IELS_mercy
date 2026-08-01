# Stock Breakout Predictor — XGBoost Baseline

A baseline, walk-forward-validated model that scores daily breakout
probability from OHLCV data and pairs each signal with an ATR-based
stop-loss, as a first-pass alternative to jumping straight into deep
learning (CNN/LSTM) architectures.

## Why this design

- **XGBoost on engineered indicators**, not a CNN/LSTM on raw price images.
  On financial time series, boosted trees on well-chosen features are
  frequently competitive with or better than deep nets, train in seconds,
  and are far easier to debug and reason about.
- **Breakout is a classification target**, not a price-regression target.
  Predicting an exact future price is close to impossible with any real
  edge; predicting "will this break out past its N-day high by X% within
  K days" is a much better-posed problem.
- **Stop-loss is a formula, not a prediction.** It's computed as a
  chandelier-style `entry_price - atr_multiplier * ATR`, the standard
  volatility-adjusted risk-management level — this is a numeric technique,
  not something to spend model capacity learning.
- **Walk-forward validation only.** Random/shuffled cross-validation leaks
  future data into training for time series and produces misleadingly high
  accuracy. Every fold here trains only on the past and tests only on data
  strictly after it.

## Pipeline

```
data.py       -> download or load OHLCV (yfinance, CSV, or synthetic)
features.py   -> technical indicators (SMA/EMA/MACD/RSI/ATR/Bollinger/
                 Donchian/volume/volatility) + forward-looking breakout
                 label + ATR stop-loss level
model.py      -> walk-forward (expanding window) XGBoost training/eval
backtest.py   -> turn out-of-sample breakout signals into simulated trades
                 with ATR stop-loss / R-multiple take-profit / time exit
cli.py        -> orchestrates the above and prints a report
```

## Usage

```bash
pip install -r ../requirements.txt

# Offline smoke test, no network required
python -m stock_predictor.cli --synthetic

# Real data via yfinance (free, no key) — Bursa Malaysia uses the .KL suffix
python -m stock_predictor.cli --symbol AAPL --start 2015-01-01
python -m stock_predictor.cli --symbol 4456.KL --start 2015-01-01   # DNEX

# Real data via EODHD (needs an API key, Bursa uses the .KLSE suffix)
python -m stock_predictor.cli --symbol 4456.KLSE --source eodhd \
    --api-key YOUR_KEY --start 2015-01-01
# or: export EODHD_API_KEY=YOUR_KEY

# From a local CSV (columns: date, open, high, low, close, volume)
python -m stock_predictor.cli --csv my_data.csv --out trades.csv
```

### Data sources: yfinance vs. EODHD

- **yfinance** (default): free, no signup, good enough for large-cap
  counters. Ticker suffix for Bursa Malaysia is `.KL` (e.g. `4456.KL` for
  DNEX).
- **EODHD**: official, ToS-compliant API with real Bursa Malaysia coverage
  (a free tier is available at eodhd.com); more reliable for smaller/less
  liquid counters where Yahoo's data can have gaps. Ticker suffix is
  `.KLSE` (e.g. `4456.KLSE`).
- **Not TradingView**: it has no public historical-data API. The packages
  that pull from it scrape TradingView's private websocket feed, which is
  against their Terms of Service and can break without notice — not used
  here.

Key tunables (see `config.py` for the full list): `--horizon` (days ahead
the breakout must occur), `--breakout-margin` (how far past the N-day high
counts as a breakout), `--proba-threshold` (signal confidence required to
trade), `--n-splits` (walk-forward folds).

## Output

The CLI prints per-fold out-of-sample classifier metrics (accuracy,
precision, recall, ROC-AUC) and backtest metrics (win rate, profit factor,
max drawdown, trade Sharpe-like ratio, and the breakdown of exits by
stop-loss / take-profit / time).

## Honest limitations

- This is a decision-support signal, not a black box you should trade real
  money on unmodified. Treat reported accuracy skeptically and re-validate
  on the specific symbols/timeframes you care about.
- No transaction costs, slippage, or market-impact modeling yet — real
  performance will be worse than the backtest.
- Single-symbol, long-only. Extending to a universe of symbols needs
  cross-sectional feature normalization to avoid regime/scale leakage.

## Next step (stage 2)

If this baseline shows real, stable out-of-sample edge, the natural
extension is a CNN over candlestick images (or a CNN-LSTM/Transformer
hybrid) trained to recognize the same breakout label directly from chart
shape, then compared against this baseline — not built blind, on the
strength of the metrics this baseline gives you.
