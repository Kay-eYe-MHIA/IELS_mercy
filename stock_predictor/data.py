"""OHLCV data loading. Prefers a live yfinance download; falls back to a
local CSV so the pipeline runs in offline/sandboxed environments too."""

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna()


def fetch_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data for `symbol` via yfinance."""
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'")
    return _standardize(df)


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a local CSV. Expects a date column plus
    open/high/low/close/volume (case-insensitive)."""
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), df.columns[0])
    df = df.set_index(date_col)
    return _standardize(df)


def make_synthetic(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic OHLCV series with injected breakout runs, for
    offline testing of the pipeline without network access."""
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n)

    returns = rng.normal(0.0003, 0.012, n)
    # Inject occasional multi-day breakout runs so the label has positives.
    i = 0
    while i < n:
        if rng.random() < 0.03:
            run_len = rng.integers(3, 8)
            returns[i:i + run_len] += rng.uniform(0.01, 0.03)
            i += run_len
        else:
            i += 1

    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.0, 0.012, n))
    low = close * (1 - rng.uniform(0.0, 0.012, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df
