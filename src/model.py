from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

N_FEATURES = 8400
REQUIRED_FIELDS = {
    "year",
    "mileage",
    "make",
    "model",
    "fuel_type",
    "transmission",
}


def _to_dense_matrix(values: Any) -> np.ndarray:
    return values.toarray()


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_hasher", FeatureHasher(n_features=N_FEATURES, input_type="dict", alternate_sign=False)),
            ("to_dense", FunctionTransformer(_to_dense_matrix, accept_sparse=True)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=20,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _synthetic_training_data(sample_size: int = 450, random_state: int = 42) -> tuple[list[dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(random_state)
    makes = np.array(["Toyota", "Honda", "Ford", "BMW", "Audi", "Hyundai"])
    models = np.array(["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Wagon"])
    fuels = np.array(["petrol", "diesel", "hybrid", "electric"])
    transmissions = np.array(["automatic", "manual"])

    rows: list[dict[str, Any]] = []
    prices = np.empty(sample_size, dtype=float)

    for i in range(sample_size):
        year = int(rng.integers(2005, 2026))
        mileage = int(rng.integers(5_000, 240_000))
        engine_size = float(rng.uniform(1.0, 4.5))
        doors = int(rng.integers(2, 6))

        row = {
            "year": year,
            "mileage": mileage,
            "make": str(rng.choice(makes)),
            "model": str(rng.choice(models)),
            "fuel_type": str(rng.choice(fuels)),
            "transmission": str(rng.choice(transmissions)),
            "engine_size_l": round(engine_size, 1),
            "doors": doors,
            "has_navigation": bool(rng.integers(0, 2)),
            "owner_count": int(rng.integers(1, 5)),
        }
        rows.append(row)

        depreciation = max(0, 2026 - year) * 950 + (mileage * 0.035)
        premium = engine_size * 2500 + (3500 if row["fuel_type"] == "electric" else 0)
        prices[i] = max(2_000.0, 45_000 + premium - depreciation + rng.normal(0, 1_200))

    return rows, prices


def train_model(output_path: str | os.PathLike[str], random_state: int = 42) -> Pipeline:
    model = build_model(random_state=random_state)
    X, y = _synthetic_training_data(random_state=random_state)
    model.fit(X, y)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return model


def load_or_train_model(model_path: str | os.PathLike[str]) -> Pipeline:
    model_file = Path(model_path)
    if model_file.exists():
        return joblib.load(model_file)
    return train_model(model_file)


def validate_features(features: Any) -> None:
    if not isinstance(features, dict):
        raise ValueError("'features' must be a JSON object")

    missing = sorted(field for field in REQUIRED_FIELDS if field not in features)
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def predict_price(model: Pipeline, features: dict[str, Any]) -> float:
    prediction = float(model.predict([features])[0])
    return round(max(prediction, 0.0), 2)
