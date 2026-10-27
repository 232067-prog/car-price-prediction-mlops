# Car Price Prediction App

This project provides a simple Flask web app for predicting a vehicle price from car details.

## Requirements

Make sure Python is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

The app expects the trained model artifacts to be present in the `model` folder:

- `car_price_model.pkl`
- `label_encoders.pkl`
- `model_columns.pkl`

## Run locally

From the project root, start the application with:

```bash
python app.py
```

Then open your browser to:

```text
http://127.0.0.1:5000/
```

## API

The app exposes a POST endpoint at `/predict` that accepts JSON with the car details and returns the predicted price.

Example request:

```json
{
  "year": 2018,
  "manufacturer": "ford",
  "model": "focus",
  "condition": "good",
  "cylinders": 4,
  "fuel": "gas",
  "odometer": 89000,
  "title_status": "clean",
  "transmission": "automatic",
  "drive": "front wheel drive",
  "size": "compact",
  "type": "sedan",
  "paint_color": "blue",
  "state": "ca"
}
```

The application computes `car_age` automatically from the vehicle year before predicting.
