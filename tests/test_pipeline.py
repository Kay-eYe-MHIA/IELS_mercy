"""Offline smoke tests for the data -> features -> model -> backtest pipeline.
Uses synthetic OHLCV data so it runs without network access."""

from unittest.mock import Mock, patch

import pytest

from stock_predictor import backtest, data, model
from stock_predictor.config import Config
from stock_predictor.features import build_dataset


def _small_dataset():
    cfg = Config(n_splits=3, min_train_size=200)
    df = data.make_synthetic(n=800, seed=1)
    X, y, meta = build_dataset(df, cfg)
    return cfg, df, X, y, meta


def test_build_dataset_shapes():
    cfg, df, X, y, meta = _small_dataset()
    assert len(X) == len(y) == len(meta)
    assert len(X) > 0
    assert not X.isna().any().any()
    assert set(y.unique()) <= {0, 1}
    assert (meta["entry_price"] > 0).all()


def test_stop_loss_below_entry():
    cfg, df, X, y, meta = _small_dataset()
    # Chandelier stop-loss must sit below the entry price for a long setup.
    assert (meta["stop_loss"] < meta["entry_price"]).all()


def test_walk_forward_no_lookahead():
    cfg, df, X, y, meta = _small_dataset()
    for train_idx, test_idx in model.walk_forward_splits(len(X), cfg):
        assert train_idx.max() < test_idx.min()


def test_eodhd_requires_api_key():
    with pytest.raises(ValueError, match="API key"):
        data.fetch_ohlcv_eodhd("4456.KLSE", "2015-01-01", api_key=None)


def test_eodhd_parses_response():
    fake_records = [
        {"date": "2024-01-02", "open": 1.0, "high": 1.05, "low": 0.98, "close": 1.02,
         "adjusted_close": 1.02, "volume": 1_000_000},
        {"date": "2024-01-03", "open": 1.02, "high": 1.10, "low": 1.01, "close": 1.08,
         "adjusted_close": 1.08, "volume": 1_200_000},
    ]
    mock_resp = Mock()
    mock_resp.json.return_value = fake_records
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp) as mock_get:
        df = data.fetch_ohlcv_eodhd("4456.KLSE", "2024-01-01", api_key="fake-key")

    assert mock_get.call_args.kwargs["params"]["api_token"] == "fake-key"
    assert list(df.columns) == data.REQUIRED_COLUMNS
    assert len(df) == 2
    assert df.iloc[-1]["close"] == 1.08


def test_train_and_backtest_smoke():
    cfg, df, X, y, meta = _small_dataset()
    results = model.train_walk_forward(X, y, cfg)
    assert len(results) == cfg.n_splits
    for r in results:
        assert 0.0 <= r.metrics["accuracy"] <= 1.0

    oos = model.out_of_sample_predictions(results)
    assert len(oos) == len(X) - cfg.min_train_size

    trades = backtest.simulate_trades(df, oos, meta, cfg)
    metrics = backtest.compute_metrics(trades)
    assert "n_trades" in metrics
    if metrics["n_trades"] > 0:
        assert 0.0 <= metrics["win_rate"] <= 1.0
