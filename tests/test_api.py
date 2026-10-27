from __future__ import annotations

from app import create_app
from src.model import train_model


def _valid_features() -> dict[str, object]:
    return {
        "year": 2022,
        "mileage": 28000,
        "make": "Toyota",
        "model": "Sedan",
        "fuel_type": "hybrid",
        "transmission": "automatic",
        "engine_size_l": 2.0,
        "owner_count": 1,
    }


def test_health_endpoint_reports_model_metadata(tmp_path):
    model_path = tmp_path / "rf.joblib"
    train_model(model_path, random_state=3)
    app = create_app(str(model_path))
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model"] == "random_forest"
    assert payload["feature_count"] >= 8300


def test_predict_endpoint_returns_price_for_valid_payload(tmp_path):
    model_path = tmp_path / "rf.joblib"
    train_model(model_path, random_state=11)
    app = create_app(str(model_path))
    client = app.test_client()

    response = client.post("/predict", json={"features": _valid_features()})

    assert response.status_code == 200
    body = response.get_json()
    assert isinstance(body["predicted_price"], float)
    assert body["predicted_price"] > 0


def test_predict_endpoint_rejects_missing_required_fields(tmp_path):
    model_path = tmp_path / "rf.joblib"
    train_model(model_path, random_state=7)
    app = create_app(str(model_path))
    client = app.test_client()

    response = client.post("/predict", json={"features": {"year": 2020, "make": "Ford"}})

    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]
