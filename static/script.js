const form = document.getElementById('car-form');
const predictedPrice = document.getElementById('predicted-price');
const statusMessage = document.getElementById('status-message');
const submitButton = form.querySelector('button[type="submit"]');
const currentYear = new Date().getFullYear();

const formatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  payload.manufacturer = String(payload.manufacturer || '').trim().toLowerCase();
  payload.model = String(payload.model || '').trim().toLowerCase();
  payload.state = String(payload.state || '').trim().toLowerCase();

  payload.year = Number(payload.year);
  payload.cylinders = Number(payload.cylinders);
  payload.odometer = Number(payload.odometer);
  payload.car_age = Math.max(0, currentYear - payload.year);

  submitButton.disabled = true;
  submitButton.textContent = 'Predicting...';
  statusMessage.textContent = 'Generating estimate...';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    predictedPrice.textContent = formatter.format(Number(data.predicted_price));
    statusMessage.textContent = 'Estimated based on the selected vehicle details.';
  } catch (error) {
    statusMessage.textContent = error.message || 'Unable to generate a price estimate.';
    predictedPrice.textContent = '$0';
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Predict Price';
  }
});
