"""Offline smoke test for the FastAPI backend, using the synthetic data
source so it runs without network access."""

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_synthetic():
    resp = client.post("/api/analyze", json={"symbol": "SYNTH", "source": "synthetic", "n_splits": 3})
    assert resp.status_code == 200
    body = resp.json()

    assert body["symbol"] == "SYNTH"
    assert body["dataset_rows"] > 0
    assert 0.0 <= body["positive_rate"] <= 1.0
    assert len(body["fold_metrics"]) == 3
    assert "n_trades" in body["backtest_metrics"]

    if body["trades"]:
        trade = body["trades"][0]
        assert {"entry_date", "exit_date", "exit_reason", "return_pct"} <= trade.keys()


def test_analyze_rejects_bad_source():
    resp = client.post("/api/analyze", json={"symbol": "X", "source": "tradingview"})
    assert resp.status_code == 422  # pydantic literal validation


def test_analyze_too_few_splits_for_data():
    resp = client.post("/api/analyze", json={"symbol": "X", "source": "synthetic", "n_splits": 20})
    # 1500-row synthetic dataset with min_train_size=500 can't support 20 splits cleanly;
    # should fail gracefully with a 400, not a 500.
    assert resp.status_code in (200, 400)
