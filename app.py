import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Global state – loaded eagerly at module import time so both uvicorn and
# TestClient (which skips the lifespan) always have real objects.
# ---------------------------------------------------------------------------
MODEL_NAME = 'taxi-tip-regressor'
MODEL_VERSION = '1'
DATASET_VERSION = 'yellow_tripdata_2024-01'

MODEL_METRICS = {
    "mae": 1.198,
    "rmse": 2.285,
    "r2": 0.643,
}

START_TIME: float = time.time()

def _load_artifacts():
    model = joblib.load('models/rf_reg_tuned.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/feature_columns.json') as fh:
        features = json.load(fh)
    print(f'Model loaded – {len(features)} features ready.')
    return model, scaler, features

MODEL, SCALER, FEATURE_COLUMNS = _load_artifacts()


# ---------------------------------------------------------------------------
# Lifespan handler – kept for correctness / future use but artifacts are
# already loaded above, so this is effectively a no-op at runtime.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Startup/shutdown hook (artifacts pre-loaded at import time).'''
    global START_TIME
    START_TIME = time.time()
    yield
    print("Shutting down API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title='Taxi Tip Prediction API',
    description='Predicts NYC yellow-taxi tip amounts using a tuned Random Forest.',
    version='1.0.0',
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TripInput(BaseModel):
    '''Raw trip features supplied by the caller.

    The six 'core' fields are always required and constrained.
    Borough flags default to 0 (Unknown) so callers can omit them.
    '''

    # Core numeric features (must be positive / in valid ranges)
    trip_distance: float = Field(
        ..., gt=0, le=200, description='Trip distance in miles (must be > 0)'
    )
    pickup_hour: int = Field(
        ..., ge=0, le=23, description='Hour of pickup (0-23)'
    )
    fare_amount: float = Field(
        ..., ge=0, le=500, description='Metered fare in USD (0–500)'
    )
    trip_duration_minutes: float = Field(
        ..., gt=0, le=600, description='Trip duration in minutes (must be > 0)'
    )
    passenger_count: int = Field(
        default=1, ge=1, le=9, description='Number of passengers (1–9)'
    )
    pickup_day_of_week: int = Field(
        default=0, ge=0, le=6, description='Day of week (0=Mon … 6=Sun)'
    )

    # ---- Pickup borough flags (one-hot; default 0 = unknown/other) ----
    pickup_borough_Bronx: int = Field(default=0, ge=0, le=1)
    pickup_borough_Brooklyn: int = Field(default=0, ge=0, le=1)
    pickup_borough_EWR: int = Field(default=0, ge=0, le=1)
    pickup_borough_Manhattan: int = Field(default=0, ge=0, le=1)
    pickup_borough_Queens: int = Field(default=0, ge=0, le=1)
    pickup_borough_Staten_Island: int = Field(
        default=0, ge=0, le=1, alias='pickup_borough_Staten Island'
    )

    # ---- Dropoff borough flags ----
    dropoff_borough_Bronx: int = Field(default=0, ge=0, le=1)
    dropoff_borough_Brooklyn: int = Field(default=0, ge=0, le=1)
    dropoff_borough_EWR: int = Field(default=0, ge=0, le=1)
    dropoff_borough_Manhattan: int = Field(default=0, ge=0, le=1)
    dropoff_borough_Queens: int = Field(default=0, ge=0, le=1)
    dropoff_borough_Staten_Island: int = Field(
        default=0, ge=0, le=1, alias='dropoff_borough_Staten Island'
    )

    model_config = {
        'populate_by_name': True,
        'json_schema_extra': {
            'examples': [
                {
                    'trip_distance': 3.5,
                    'pickup_hour': 14,
                    'fare_amount': 15.0,
                    'trip_duration_minutes': 20.0,
                    'passenger_count': 1,
                    'pickup_day_of_week': 2,
                    'pickup_borough_Manhattan': 1,
                    'dropoff_borough_Manhattan': 1,
                }
            ]
        },
    }


class PredictionResponse(BaseModel):
    tip_amount: float
    model_version: str
    prediction_id: str


class BatchInput(BaseModel):
    records: List[TripInput] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Feature-engineering helper
# ---------------------------------------------------------------------------

def _build_feature_vector(trip: TripInput) -> np.ndarray:
    '''Convert a TripInput to a scaled numpy row matching FEATURE_COLUMNS.'''

    # Derived features that mirror the training pipeline
    trip_speed_mph = trip.trip_distance / max(trip.trip_duration_minutes / 60, 1e-6)
    log_trip_distance = float(np.log1p(trip.trip_distance))
    fare_per_mile = trip.fare_amount / max(trip.trip_distance, 1e-6)
    fare_per_minute = trip.fare_amount / max(trip.trip_duration_minutes, 1e-6)
    is_weekend = int(trip.pickup_day_of_week >= 5)

    # Build a dict covering every expected column (zero-fill unknowns)
    row: dict = {col: 0 for col in FEATURE_COLUMNS}

    raw_values = {
        'trip_distance': trip.trip_distance,
        'pickup_hour': trip.pickup_hour,
        'fare_amount': trip.fare_amount,
        'trip_duration_minutes': trip.trip_duration_minutes,
        'passenger_count': trip.passenger_count,
        'pickup_day_of_week': trip.pickup_day_of_week,
        'is_weekend': is_weekend,
        'trip_speed_mph': trip_speed_mph,
        'log_trip_distance': log_trip_distance,
        'fare_per_mile': fare_per_mile,
        'fare_per_minute': fare_per_minute,
        # Borough one-hot flags
        'pickup_borough_Bronx': trip.pickup_borough_Bronx,
        'pickup_borough_Brooklyn': trip.pickup_borough_Brooklyn,
        'pickup_borough_EWR': trip.pickup_borough_EWR,
        'pickup_borough_Manhattan': trip.pickup_borough_Manhattan,
        'pickup_borough_Queens': trip.pickup_borough_Queens,
        'pickup_borough_Staten Island': trip.pickup_borough_Staten_Island,
        'dropoff_borough_Bronx': trip.dropoff_borough_Bronx,
        'dropoff_borough_Brooklyn': trip.dropoff_borough_Brooklyn,
        'dropoff_borough_EWR': trip.dropoff_borough_EWR,
        'dropoff_borough_Manhattan': trip.dropoff_borough_Manhattan,
        'dropoff_borough_Queens': trip.dropoff_borough_Queens,
        'dropoff_borough_Staten Island': trip.dropoff_borough_Staten_Island,
    }

    for key, val in raw_values.items():
        if key in row:
            row[key] = val

    vector = np.array([row[col] for col in FEATURE_COLUMNS], dtype=float).reshape(1, -1)
    return SCALER.transform(vector)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post('/predict', response_model=PredictionResponse, tags=["Predictions"])
def predict(trip: TripInput):
    '''Return a single tip-amount prediction for the supplied trip features.'''
    features = _build_feature_vector(trip)
    raw_pred = MODEL.predict(features)[0]
    return PredictionResponse(
        tip_amount=round(float(raw_pred), 2),
        model_version=MODEL_VERSION,
        prediction_id=str(uuid.uuid4()),
    )


@app.post('/predict/batch', response_model=BatchResponse, tags=["Predictions"])
def predict_batch(batch: BatchInput):
    '''Return tip predictions for a list of trips (max 100 records).'''
    t0 = time.time()
    results: List[PredictionResponse] = []

    for trip in batch.records:
        features = _build_feature_vector(trip)
        raw_pred = MODEL.predict(features)[0]
        results.append(
            PredictionResponse(
                tip_amount=round(float(raw_pred), 2),
                model_version=MODEL_VERSION,
                prediction_id=str(uuid.uuid4()),
            )
        )

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    return BatchResponse(predictions=results, count=len(results), processing_time_ms=elapsed_ms)


@app.get('/health', tags=['Monitoring'])
def health_check():
    '''Return API liveness and model-load status.'''
    return {
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'model_version': MODEL_VERSION,
        'uptime_seconds': round(time.time() - START_TIME, 1),
    }


@app.get('/model/info', tags=['Monitoring'])
def model_info():
    '''Return metadata about the currently loaded model.'''
    return {
        'model_name': MODEL_NAME,
        'version': MODEL_VERSION,
        'dataset_version': DATASET_VERSION,
        'feature_names': FEATURE_COLUMNS,
        'feature_count': len(FEATURE_COLUMNS),
        'training_metrics': MODEL_METRICS,
    }


# ---------------------------------------------------------------------------
# Global exception handler – never expose raw tracebacks
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            'error': "Internal server error",
            'detail': "An unexpected error occurred. Please try again later.",
        },
    )