"""FastAPI wrapper around the stock_predictor pipeline, so the mobile
frontend (hosted on Vercel) can trigger a real walk-forward XGBoost
analysis over HTTP instead of running Python locally.

Runs on a normal always-on host (e.g. Render), not Vercel serverless --
xgboost + pandas + scikit-learn are far too large for Vercel's ~250MB
serverless function limit.
"""

import math
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from stock_predictor import backtest, data, model
from stock_predictor.config import Config
from stock_predictor.features import build_dataset

app = FastAPI(title="Stock Breakout Predictor API")

# Personal-tool CORS: no auth/secrets pass through this API (EODHD keys are
# supplied per-request and forwarded, never stored), so allowing any origin
# is an acceptable tradeoff. Tighten to your Vercel domain if you prefer.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    symbol: str = "AAPL"
    source: Literal["yfinance", "eodhd", "synthetic"] = "yfinance"
    api_key: Optional[str] = None
    start: str = "2015-01-01"
    end: Optional[str] = None
    horizon: int = Field(default=5, ge=1, le=60)
    breakout_margin: float = Field(default=0.02, ge=0.0, le=1.0)
    proba_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    n_splits: int = Field(default=5, ge=2, le=20)


def _json_safe(value):
    """Replace non-finite floats (NaN/Infinity) with None -- standard JSON
    (and JS's JSON.parse) can't represent them."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating,)):
        return _json_safe(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    return _json_safe(df.to_dict(orient="records"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    cfg = Config(symbol=req.symbol, start=req.start, end=req.end)
    cfg.horizon = req.horizon
    cfg.breakout_margin = req.breakout_margin
    cfg.proba_threshold = req.proba_threshold
    cfg.n_splits = req.n_splits

    try:
        if req.source == "synthetic":
            df = data.make_synthetic()
        else:
            df = data.fetch(req.symbol, req.start, req.end, source=req.source, api_key=req.api_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data fetch failed: {exc}") from exc

    try:
        X, y, meta = build_dataset(df, cfg)
        if len(X) < cfg.min_train_size + cfg.n_splits:
            raise ValueError(
                f"Not enough usable rows ({len(X)}) for {cfg.n_splits} walk-forward "
                f"splits with min_train_size={cfg.min_train_size}. Use a longer "
                f"date range or fewer splits."
            )
        results = model.train_walk_forward(X, y, cfg)
        fold_summary = model.summarize_folds(results).reset_index(names="fold")
        oos = model.out_of_sample_predictions(results)
        trades = backtest.simulate_trades(df, oos, meta, cfg)
        backtest_metrics = backtest.compute_metrics(trades)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}") from exc

    return {
        "symbol": req.symbol,
        "source": req.source,
        "rows": len(df),
        "dataset_rows": len(X),
        "date_range": [df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d")],
        "positive_rate": _json_safe(float(y.mean())),
        "fold_metrics": _records(fold_summary),
        "backtest_metrics": _json_safe(backtest_metrics),
        "trades": _records(trades.tail(50)) if not trades.empty else [],
    }
