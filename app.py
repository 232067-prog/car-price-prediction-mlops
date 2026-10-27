from __future__ import annotations

import os
from flask import Flask, jsonify, request

from src.model import N_FEATURES, load_or_train_model, predict_price, validate_features


def create_app(model_path: str | None = None) -> Flask:
    app = Flask(__name__)
    resolved_model_path = model_path or os.getenv("CAR_MODEL_PATH", "models/random_forest.joblib")
    model = load_or_train_model(resolved_model_path)

    @app.get("/health")
    def health() -> tuple[dict[str, str | int], int]:
        return {"status": "ok", "model": "random_forest", "feature_count": N_FEATURES}, 200

    @app.post("/predict")
    def predict() -> tuple[dict[str, float | str], int]:
        payload = request.get_json(silent=True)
        if payload is None:
            return {"error": "Request body must be valid JSON"}, 400

        features = payload.get("features", payload)
        try:
            validate_features(features)
        except ValueError as exc:
            return {"error": str(exc)}, 400

        predicted_price = predict_price(model, features)
        return {"predicted_price": predicted_price, "currency": "USD"}, 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
