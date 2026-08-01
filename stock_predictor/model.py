"""Walk-forward XGBoost training and evaluation for the breakout classifier.

Walk-forward (expanding window, chronological) is used instead of random
k-fold: shuffled CV leaks future information into training for time series
and produces inflated, unrealistic accuracy.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from .config import Config


@dataclass
class FoldResult:
    test_index: pd.Index
    y_true: np.ndarray
    y_proba: np.ndarray
    metrics: dict = field(default_factory=dict)


def walk_forward_splits(n: int, cfg: Config):
    """Yield (train_idx, test_idx) index arrays for an expanding-window
    walk-forward split. Each test fold is strictly after its train fold."""
    fold_size = (n - cfg.min_train_size) // cfg.n_splits
    if fold_size <= 0:
        raise ValueError("Not enough rows for the configured n_splits/min_train_size")
    for i in range(cfg.n_splits):
        train_end = cfg.min_train_size + i * fold_size
        test_end = train_end + fold_size if i < cfg.n_splits - 1 else n
        yield np.arange(0, train_end), np.arange(train_end, test_end)


def train_walk_forward(X: pd.DataFrame, y: pd.Series, cfg: Config) -> list[FoldResult]:
    results = []
    for train_idx, test_idx in walk_forward_splits(len(X), cfg):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = XGBClassifier(**cfg.xgb_params)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan"),
            "positive_rate": y_test.mean(),
            "n_test": len(y_test),
        }
        results.append(FoldResult(X_test.index, y_test.values, proba, metrics))
    return results


def summarize_folds(results: list[FoldResult]) -> pd.DataFrame:
    return pd.DataFrame([r.metrics for r in results], index=[f"fold_{i}" for i in range(len(results))])


def out_of_sample_predictions(results: list[FoldResult]) -> pd.DataFrame:
    """Stitch every fold's held-out predictions into one chronological
    out-of-sample series, for backtesting."""
    frames = [
        pd.DataFrame({"y_true": r.y_true, "y_proba": r.y_proba}, index=r.test_index)
        for r in results
    ]
    return pd.concat(frames).sort_index()
