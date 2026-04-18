from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_TRIP = {
    'trip_distance': 3.5,
    'pickup_hour': 14,
    'fare_amount': 15.0,
    'trip_duration_minutes': 20.0,
    'passenger_count': 1,
    'pickup_day_of_week': 2,
    'pickup_borough_Manhattan': 1,
    'dropoff_borough_Manhattan': 1,
}

# ────────────────────────────────────────────────────────────────
# Task 2.3 – Required test cases (at least 5)
# ────────────────────────────────────────────────────────────────

# 1. Successful single prediction with valid input
def test_predict_valid_input():
    '''POST /predict with a well-formed payload returns 200 and a tip amount.'''
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    data = response.json()
    assert 'tip_amount' in data
    assert isinstance(data['tip_amount'], float)
    assert 'prediction_id' in data
    assert 'model_version' in data


# 2. Successful batch prediction
def test_predict_batch_valid():
    '''POST /predict/batch with 3 records returns 200 and the correct count.'''
    payload = {'records': [VALID_TRIP] * 3}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['count'] == 3
    assert len(data['predictions']) == 3
    assert 'processing_time_ms' in data


# 3. Invalid input – missing required fields
def test_predict_missing_fields():
    '''A payload missing required fields should return HTTP 422.'''
    response = client.post("/predict", json={'pickup_hour': 10})
    assert response.status_code == 422

# 4. Health check endpoint returns correct status
def test_health_check():
    '''GET /health should return 200 with status='healthy' and model_loaded=True.'''
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert 'uptime_seconds' in data


# 5. Edge case – zero distance trip (violates gt=0 constraint)
def test_predict_zero_distance():
    '''trip_distance=0 must be rejected because the constraint is gt=0.'''
    bad = {**VALID_TRIP, 'trip_distance': 0.0}
    response = client.post('/predict', json=bad)
    assert response.status_code == 422