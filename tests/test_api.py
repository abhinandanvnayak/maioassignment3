import os
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_FEATURES = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def test_health_returns_status_and_model_version() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": os.getenv("MODEL_VERSION", "v0.1")}


def test_predict_returns_numeric_prediction() -> None:
    response = client.post("/predict", json=VALID_FEATURES)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_reports_missing_feature() -> None:
    features = VALID_FEATURES.copy()
    del features["s6"]

    response = client.post("/predict", json=features)

    assert response.status_code == 422
    assert "s6" in response.json()["detail"][0]["loc"]
