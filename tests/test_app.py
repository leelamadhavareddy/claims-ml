from fastapi.testclient import TestClient
from claims_ml.serving.app import app

client = TestClient(app)


def test_health_ok():
    assert client.get("/health").json() == {"status": "ok"}


def test_version_reports_champion():
    assert client.get("/version").json()["alias"] == "champion"


def test_predict_rejects_bad_input():
    # age below minimum -> 422 validation error, no model needed
    r = client.post("/predict", json={"age": 5})
    assert r.status_code == 422