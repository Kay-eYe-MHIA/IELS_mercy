from dataclasses import dataclass


@dataclass
class Config:
    # Data
    symbol: str = "AAPL"
    start: str = "2015-01-01"
    end: str | None = None

    # Indicator windows
    sma_windows: tuple[int, ...] = (5, 10, 20, 50)
    rsi_window: int = 14
    atr_window: int = 14
    donchian_window: int = 20
    volume_window: int = 20
    volatility_window: int = 20

    # Breakout label
    # A day is labeled a positive "breakout" if, within `horizon` future
    # trading days, price closes above the current N-day Donchian high by
    # at least `breakout_margin` (e.g. 0.02 = 2%).
    horizon: int = 5
    breakout_margin: float = 0.02

    # ATR-based stop-loss (chandelier-style): stop = entry - atr_multiplier * ATR
    atr_multiplier: float = 2.0

    # Walk-forward evaluation
    n_splits: int = 5
    min_train_size: int = 500

    # Model
    xgb_params: dict = None  # filled in __post_init__

    # Backtest
    proba_threshold: float = 0.6
    max_holding_days: int = 10
    take_profit_r_multiple: float = 2.0  # exit at profit = R * (entry - stop)
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01  # fraction of capital risked per trade

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = dict(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
                n_jobs=-1,
            )
