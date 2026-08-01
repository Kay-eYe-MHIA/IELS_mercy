"""Technical-indicator feature engineering and breakout / stop-loss labeling.

All indicators are computed causally (only past/current data at each row) so
that walk-forward training never leaks future information. The label is the
only forward-looking column and is dropped before the final `horizon` rows.
"""

import numpy as np
import pandas as pd

from .config import Config

FEATURE_COLUMNS: list[str] = []  # populated by build_features()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    for w in cfg.sma_windows:
        sma = close.rolling(w).mean()
        out[f"sma_{w}_ratio"] = close / sma - 1

    out["ema_12_26_macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["macd_signal"] = out["ema_12_26_macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["ema_12_26_macd"] - out["macd_signal"]

    out["rsi"] = _rsi(close, cfg.rsi_window)

    atr = _atr(df, cfg.atr_window)
    out["atr_pct"] = atr / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_width"] = (4 * bb_std) / bb_mid
    out["bb_pctb"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    donchian_high = high.rolling(cfg.donchian_window).max()
    donchian_low = low.rolling(cfg.donchian_window).min()
    out["dist_to_donchian_high"] = close / donchian_high - 1
    out["dist_to_donchian_low"] = close / donchian_low - 1
    out["donchian_width"] = donchian_high / donchian_low - 1

    vol_avg = volume.rolling(cfg.volume_window).mean()
    out["volume_ratio"] = volume / vol_avg

    out["volatility"] = close.pct_change().rolling(cfg.volatility_window).std()

    for lag in (1, 3, 5, 10):
        out[f"return_{lag}d"] = close.pct_change(lag)

    out["atr"] = atr  # kept for stop-loss calc; excluded from model features below

    global FEATURE_COLUMNS
    FEATURE_COLUMNS = [c for c in out.columns if c != "atr"]
    return out


def build_labels(df: pd.DataFrame, features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Forward-looking breakout label + ATR-based stop-loss level.

    breakout_label = 1 if, within the next `horizon` trading days, the close
    exceeds today's Donchian high by at least `breakout_margin`.
    stop_loss = today's close - atr_multiplier * ATR (chandelier-style),
    the risk-management level a trade entered on this signal would use.
    """
    close = df["close"]
    donchian_high = df["high"].rolling(cfg.donchian_window).max()
    target = donchian_high * (1 + cfg.breakout_margin)

    future_max_close = close.shift(-1).rolling(cfg.horizon, min_periods=1).max().shift(-(cfg.horizon - 1))
    breakout_label = (future_max_close >= target).astype(int)

    stop_loss = close - cfg.atr_multiplier * features["atr"]

    labels = pd.DataFrame(
        {
            "breakout_label": breakout_label,
            "stop_loss": stop_loss,
            "entry_price": close,
        },
        index=df.index,
    )
    return labels


def build_dataset(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (X, y, meta) aligned and cleaned: X are model features, y is
    the breakout label, meta carries entry_price/stop_loss for backtesting."""
    features = build_features(df, cfg)
    labels = build_labels(df, features, cfg)

    combined = pd.concat([features, labels], axis=1)
    combined = combined.iloc[:-cfg.horizon]  # drop tail rows without a full future window
    combined = combined.dropna()

    X = combined[FEATURE_COLUMNS]
    y = combined["breakout_label"]
    meta = combined[["entry_price", "stop_loss"]]
    return X, y, meta
