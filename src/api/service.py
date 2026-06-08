"""Prediction service: loads the most recent exported model bundle and runs inference."""

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, LEAKY_COLS
from src.api.schemas import PredictionRequest

logger = logging.getLogger(__name__)

_bundle = None  # {"model": ..., "scaler": ..., "columns": [...]}


def _load_latest_bundle() -> dict:
    models = sorted(MODELS_DIR.glob("*.joblib"))
    if not models:
        logger.error("No .joblib models found in %s", MODELS_DIR)
        raise FileNotFoundError(f"No trained model found in {MODELS_DIR}")
    latest = models[-1]
    logger.info("Loading model: %s", latest.name)
    bundle = joblib.load(latest)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            f"{latest.name} is not a valid bundle. Re-run export_model.py to regenerate."
        )
    return bundle


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        _bundle = _load_latest_bundle()
    return _bundle


def _preprocess(request: PredictionRequest, scaler, columns: list[str]) -> np.ndarray:
    """Apply the same preprocessing the training pipeline used.

    Steps:
    1. Build a single-row DataFrame from the request.
    2. Drop leaky columns (none expected here, but kept for consistency).
    3. One-hot encode with get_dummies, then reindex to training columns
       (fills any unseen dummy columns with 0).
    4. Scale with the fitted scaler.
    """
    row = pd.DataFrame([{
        "venue_name":        request.venue_name,
        "venue_type":        request.venue_type,
        "venue_capacity":    request.venue_capacity,
        "continent":         request.continent,
        "country":           request.country,
        "city":              request.city,
        "artist_name":       request.artist_name,
        "gender":            request.gender,
        "generation":        request.generation,
        "members":           request.members,
        "company":           request.company,
        "years_since_debut": request.years_since_debut,
        "artist_age_bucket": request.artist_age_bucket,
        "show_nights":       request.show_nights,
    }])

    # Drop leaky cols if somehow present
    row = row.drop(columns=[c for c in LEAKY_COLS if c in row.columns])

    # One-hot encode then align to training columns (unseen categories → 0)
    row = pd.get_dummies(row, drop_first=True)
    row = row.reindex(columns=columns, fill_value=0)

    return scaler.transform(row)


def predict(request: PredictionRequest) -> int:
    bundle = get_bundle()
    X = _preprocess(request, bundle["scaler"], bundle["columns"])
    prediction = bundle["model"].predict(X)
    return max(0, round(float(prediction[0])))