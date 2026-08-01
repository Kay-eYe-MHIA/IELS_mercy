"""Walk-forward backtest: turn breakout-probability signals into simulated
trades using an ATR-based stop-loss and R-multiple take-profit, sized by a
fixed fraction of capital risked per trade."""

import numpy as np
import pandas as pd

from .config import Config


def simulate_trades(df: pd.DataFrame, oos: pd.DataFrame, meta: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dates = df.index
    signal_dates = oos.index[oos["y_proba"] >= cfg.proba_threshold]

    trades = []
    for signal_date in signal_dates:
        entry_price = meta.loc[signal_date, "entry_price"]
        stop_loss = meta.loc[signal_date, "stop_loss"]
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            continue
        take_profit = entry_price + cfg.take_profit_r_multiple * risk_per_share

        pos = dates.get_loc(signal_date)
        entry_idx = pos + 1
        if entry_idx >= len(dates):
            continue
        entry_exec_price = df["open"].iloc[entry_idx]

        end_idx = min(entry_idx + cfg.max_holding_days, len(dates) - 1)
        exit_price, exit_date, exit_reason = None, None, None
        for idx in range(entry_idx, end_idx + 1):
            if df["low"].iloc[idx] <= stop_loss:
                exit_price, exit_date, exit_reason = stop_loss, dates[idx], "stop_loss"
                break
            if df["high"].iloc[idx] >= take_profit:
                exit_price, exit_date, exit_reason = take_profit, dates[idx], "take_profit"
                break
        if exit_price is None:
            exit_price, exit_date, exit_reason = df["close"].iloc[end_idx], dates[end_idx], "time_exit"

        shares = (cfg.initial_capital * cfg.risk_per_trade) / risk_per_share
        pnl = (exit_price - entry_exec_price) * shares
        ret_pct = exit_price / entry_exec_price - 1

        trades.append(
            dict(
                signal_date=signal_date,
                entry_date=dates[entry_idx],
                exit_date=exit_date,
                entry_price=entry_exec_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exit_price=exit_price,
                exit_reason=exit_reason,
                shares=shares,
                pnl=pnl,
                return_pct=ret_pct,
                proba=oos.loc[signal_date, "y_proba"],
            )
        )
    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()

    equity = trades["pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = drawdown.min()

    returns = trades["return_pct"]
    sharpe_like = returns.mean() / returns.std() if returns.std() > 0 else float("nan")

    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_return_pct": returns.mean(),
        "total_pnl": trades["pnl"].sum(),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_drawdown": max_drawdown,
        "trade_sharpe_like": sharpe_like,
        "stop_loss_exits": (trades["exit_reason"] == "stop_loss").mean(),
        "take_profit_exits": (trades["exit_reason"] == "take_profit").mean(),
        "time_exits": (trades["exit_reason"] == "time_exit").mean(),
    }
