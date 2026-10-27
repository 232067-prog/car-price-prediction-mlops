from pathlib import Path
from datetime import date

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

CURRENT_YEAR = date.today().year

MODEL = None
LABEL_ENCODERS = {}
MODEL_COLUMNS = []

CATEGORICAL_FIELDS = [
    "manufacturer",
    "model",
    "condition",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
    "state",
]

REQUIRED_FIELDS = [
    "year",
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
    "state",
]


def load_model_artifacts():
    global MODEL, LABEL_ENCODERS, MODEL_COLUMNS

    model_path = MODEL_DIR / "car_price_model.pkl"
    encoders_path = MODEL_DIR / "label_encoders.pkl"
    columns_path = MODEL_DIR / "model_columns.pkl"

    missing_files = [
        str(path)
        for path in [model_path, encoders_path, columns_path]
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing model artifacts: " + ", ".join(missing_files)
        )

    MODEL = joblib.load(model_path)
    LABEL_ENCODERS = joblib.load(encoders_path)
    MODEL_COLUMNS = joblib.load(columns_path)

    if not isinstance(MODEL_COLUMNS, list):
        raise ValueError("model_columns.pkl must contain an ordered list of column names.")


try:
    load_model_artifacts()
    app.config["ARTIFACTS_LOADED"] = True
    app.config["ARTIFACTS_ERROR"] = None
except Exception as exc:  # pragma: no cover - defensive for startup without model files
    app.config["ARTIFACTS_LOADED"] = False
    app.config["ARTIFACTS_ERROR"] = str(exc)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def validate_range(field_name, value, min_value=None, max_value=None):
    if min_value is not None and value < min_value:
        return f"{field_name} must be at least {min_value}."
    if max_value is not None and value > max_value:
        return f"{field_name} must be at most {max_value}."
    return None


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def encode_category(value, encoder):
    if encoder is None:
        return 0

    raw_value = normalize_text(value)
    if raw_value == "":
        return 0

    if not hasattr(encoder, "classes_") or not len(encoder.classes_):
        return 0

    try:
        return int(encoder.transform([raw_value])[0])
    except ValueError:
        fallback = str(encoder.classes_[0])
        try:
            return int(encoder.transform([fallback])[0])
        except ValueError:
            return 0


def build_feature_row(payload):
    year = int(payload.get("year", CURRENT_YEAR))
    computed_age = payload.get("car_age")
    if computed_age is None or computed_age == "":
        computed_age = max(0, CURRENT_YEAR - year)

    row = {"year": year, "car_age": int(safe_float(computed_age, default=max(0, CURRENT_YEAR - year)))}
    for field in REQUIRED_FIELDS:
        if field not in row:
            row[field] = payload.get(field, "")

    for field in CATEGORICAL_FIELDS:
        if field in row:
            row[field] = normalize_text(row[field])

    for field in ["year", "cylinders", "odometer", "car_age"]:
        if field in row:
            try:
                row[field] = float(row[field])
            except (TypeError, ValueError):
                row[field] = 0.0

    for field in CATEGORICAL_FIELDS:
        if field not in MODEL_COLUMNS:
            continue

        encoder = None
        if isinstance(LABEL_ENCODERS, dict):
            encoder = LABEL_ENCODERS.get(field)
            if encoder is None:
                encoder = LABEL_ENCODERS.get(str(field))

        row[field] = encode_category(row.get(field, ""), encoder)

    feature_row = {column: 0 for column in MODEL_COLUMNS}
    for column in MODEL_COLUMNS:
        if column in row:
            feature_row[column] = row[column]

    return feature_row


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    status = "ok" if app.config.get("ARTIFACTS_LOADED") else "degraded"
    return jsonify(
        {
            "status": status,
            "artifacts_loaded": bool(app.config.get("ARTIFACTS_LOADED")),
            "year_context": CURRENT_YEAR,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not app.config.get("ARTIFACTS_LOADED"):
        return jsonify({"error": app.config.get("ARTIFACTS_ERROR", "Model artifacts not loaded.")}), 500

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing = [field for field in REQUIRED_FIELDS if field not in payload or payload.get(field) in (None, "")]
    if missing:
        return jsonify({"error": "Missing required fields: " + ", ".join(missing)}), 400

    year = safe_float(payload.get("year"), default=CURRENT_YEAR)
    year_error = validate_range("Year", year, min_value=1886, max_value=CURRENT_YEAR + 1)
    if year_error:
        return jsonify({"error": year_error}), 400

    cylinders = safe_float(payload.get("cylinders"), default=0)
    cylinders_error = validate_range("Cylinders", cylinders, min_value=1, max_value=24)
    if cylinders_error:
        return jsonify({"error": cylinders_error}), 400

    odometer = safe_float(payload.get("odometer"), default=0)
    odometer_error = validate_range("Odometer", odometer, min_value=0, max_value=2_000_000)
    if odometer_error:
        return jsonify({"error": odometer_error}), 400

    computed_age = payload.get("car_age")
    if computed_age is None or computed_age == "":
        computed_age = max(0, CURRENT_YEAR - int(year))
    else:
        computed_age = max(0, int(safe_float(computed_age, default=max(0, CURRENT_YEAR - int(year)))))

    payload["car_age"] = computed_age

    feature_row = build_feature_row(payload)
    features = pd.DataFrame([feature_row], columns=MODEL_COLUMNS)
    prediction = float(MODEL.predict(features)[0])

    return jsonify({"predicted_price": round(prediction, 2)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
