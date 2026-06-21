"""Phase-4 serving tests. Requires the exported artifact (scripts.export_model).

Redis is not required: the cache degrades to 'unavailable' and the API still
serves. These run only if the artifact exists.
"""

from __future__ import annotations

import pytest

from stockvol.serving.artifact import ARTIFACT_DIR
from stockvol.serving.cache import seconds_to_next_close

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model_fp32.pt").exists(),
    reason="run scripts.export_model first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from stockvol.serving.app import app

    with TestClient(app) as c:  # triggers lifespan -> loads model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["model_quant"] in ("static-INT8", "FP32")
    assert body["n_tickers"] >= 1
    assert body["redis"] in ("connected", "unavailable", "disabled")


def test_predict_default_date(client):
    r = client.post("/predict", json={"ticker": "RELIANCE.NS"})
    assert r.status_code == 200
    body = r.json()
    assert body["bucket"] in ("low", "med", "high")
    assert set(body["probs"]) == {"low", "med", "high"}
    assert abs(sum(body["probs"].values()) - 1.0) < 1e-4
    assert body["predicted_for"] > body["as_of_date"]


def test_predict_unknown_ticker(client):
    r = client.post("/predict", json={"ticker": "NOPE.NS"})
    assert r.status_code == 404


def test_predict_is_deterministic(client):
    a = client.post("/predict", json={"ticker": "INFY.NS"}).json()
    b = client.post("/predict", json={"ticker": "INFY.NS"}).json()
    assert a["bucket"] == b["bucket"]
    assert a["probs"] == b["probs"]


def test_metrics_endpoint(client):
    client.post("/predict", json={"ticker": "TCS.NS"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predictions_total" in r.text


def test_seconds_to_next_close_bounds():
    s = seconds_to_next_close()
    assert 60 <= s <= 24 * 3600
