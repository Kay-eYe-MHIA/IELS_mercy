"""CLI entry point: fetch data, engineer features, walk-forward train/eval
an XGBoost breakout classifier, backtest the resulting signals, and print a
report.

Examples:
    python -m stock_predictor.cli --symbol AAPL --start 2015-01-01
    python -m stock_predictor.cli --csv my_data.csv
    python -m stock_predictor.cli --synthetic   # no network required
"""

import argparse

import pandas as pd

from . import backtest, data, model
from .config import Config
from .features import build_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--csv", default=None, help="Load OHLCV from a local CSV instead of downloading")
    p.add_argument("--synthetic", action="store_true", help="Use generated synthetic data (offline smoke test)")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--breakout-margin", type=float, default=None)
    p.add_argument("--proba-threshold", type=float, default=None)
    p.add_argument("--n-splits", type=int, default=None)
    p.add_argument("--out", default=None, help="Optional path to write the trades CSV")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(symbol=args.symbol, start=args.start, end=args.end)
    if args.horizon is not None:
        cfg.horizon = args.horizon
    if args.breakout_margin is not None:
        cfg.breakout_margin = args.breakout_margin
    if args.proba_threshold is not None:
        cfg.proba_threshold = args.proba_threshold
    if args.n_splits is not None:
        cfg.n_splits = args.n_splits

    if args.synthetic:
        df = data.make_synthetic()
    elif args.csv:
        df = data.load_csv(args.csv)
    else:
        df = data.fetch_ohlcv(cfg.symbol, cfg.start, cfg.end)

    print(f"Loaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")

    X, y, meta = build_dataset(df, cfg)
    print(f"Dataset after feature/label engineering: {len(X)} rows, {X.shape[1]} features, "
          f"positive rate {y.mean():.2%}")

    results = model.train_walk_forward(X, y, cfg)
    fold_summary = model.summarize_folds(results)
    pd.set_option("display.width", 120)
    print("\nWalk-forward classifier performance (out-of-sample, per fold):")
    print(fold_summary.round(3))

    oos = model.out_of_sample_predictions(results)
    trades = backtest.simulate_trades(df, oos, meta, cfg)
    metrics = backtest.compute_metrics(trades)

    print("\nBacktest (breakout signals -> ATR stop-loss / R-multiple take-profit trades):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.out and not trades.empty:
        trades.to_csv(args.out, index=False)
        print(f"\nSaved {len(trades)} trades to {args.out}")


if __name__ == "__main__":
    main()
