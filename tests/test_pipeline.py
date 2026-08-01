"""Offline smoke tests for the data -> features -> model -> backtest pipeline.
Uses synthetic OHLCV data so it runs without network access."""

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
