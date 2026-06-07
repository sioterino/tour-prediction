"""Prediction service: loads the most recent exported model and runs inference."""

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.api.schemas import PredictionRequest

logger = logging.getLogger(__name__)


def _load_latest_model():
    """Return the most recently exported .joblib model from MODELS_DIR."""
    models = sorted(MODELS_DIR.glob("*.joblib"))
    if not models:
        logger.error("No .joblib models found in %s", MODELS_DIR)
        raise FileNotFoundError(f"No trained model found in {MODELS_DIR}")

    latest = models[-1]  # sorted by name → timestamp suffix picks the newest
    logger.info("Loading model: %s", latest.name)
    return joblib.load(latest)


# Loaded once at startup, reused across requests
_model = None


def get_model():
    """Return the cached model, loading it on first call."""
    global _model
    if _model is None:
        _model = _load_latest_model()
    return _model


def predict(request: PredictionRequest) -> float:
    """Run the model on the incoming request and return the predicted attendance."""
    model = get_model()

    # Build a single-row DataFrame matching the feature names from training.
    # Update this dict as you replace placeholder fields with real ones.
    features = pd.DataFrame([{
        "venue_capacity": request.venue_capacity,
        "feature_b":      request.feature_b,
        "feature_c":      request.feature_c,
    }])

    prediction: np.ndarray = model.predict(features)
    return float(prediction[0])