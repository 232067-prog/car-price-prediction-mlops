# car-price-prediction-mlops

Production-grade used car valuation web platform powered by a Scikit-Learn Random Forest model with **8,400 engineered features**, served through a Flask REST API and automated with GitHub Actions CI/CD.

## API

- `GET /health` - service health and model metadata
- `POST /predict` - predict used-car valuation

### Example prediction request

```json
{
  "features": {
    "year": 2022,
    "mileage": 28000,
    "make": "Toyota",
    "model": "Sedan",
    "fuel_type": "hybrid",
    "transmission": "automatic",
    "engine_size_l": 2.0,
    "owner_count": 1
  }
}
```

## Local run

```bash
python -m pip install -r requirements.txt
python app.py
```

## Test

```bash
pytest -q
```
