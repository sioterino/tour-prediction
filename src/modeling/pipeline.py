"""Shared modeling pipeline used by both experiments.py and predict.py.

Responsibilities
----------------
- Feature matrix construction (drop leaky cols, one-hot encode, scale)
- Model training and prediction
- Loading/filtering a dataset into train and predict splits
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import LEAKY_COLS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Feature engineering ───────────────────────────────────────────────────────

def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with any leaky columns removed (silently skips missing ones)."""
    present = [c for c in LEAKY_COLS if c in df.columns]
    if present:
        logger.debug("Dropping leaky columns: %s", present)
    return df.drop(columns=present)


def encode_and_align(X_train: pd.DataFrame, X_other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode both frames jointly so columns are guaranteed to align.
 
    Parameters
    ----------
    X_train:
        Feature frame used for fitting (train split or labelled rows).
    X_other:
        Feature frame to transform (test split or rows to predict).
 
    Returns
    -------
    Encoded (X_train, X_other) with identical column sets.
    """
    combined = pd.concat([X_train, X_other], axis=0)
    combined = pd.get_dummies(combined, drop_first=True)
    return (
        combined.loc[X_train.index].copy(),
        combined.loc[X_other.index].copy(),
    )


def scale_features(X_train: pd.DataFrame | np.ndarray, X_other: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a StandardScaler on *X_train* and transform both frames."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_other)


def build_feature_matrices(X_train: pd.DataFrame, X_other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: drop leaky cols > encode > scale.
 
    Parameters
    ----------
    X_train, X_other:
        Raw feature DataFrames (must NOT include the target column).
 
    Returns
    -------
    Scaled numpy arrays ``(X_train_scaled, X_other_scaled)``.
    """
    X_train = drop_leaky_columns(X_train)
    X_other = drop_leaky_columns(X_other)
 
    X_train_enc, X_other_enc = encode_and_align(X_train, X_other)
    return scale_features(X_train_enc, X_other_enc)


def build_train_matrix(X_train: pd.DataFrame) -> np.ndarray:
    """Preprocessing pipeline for a single frame (no second split to align with)."""
    X = drop_leaky_columns(X_train)
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that have a valid *attendance* value and *venue_capacity*."""
    before = len(df)

    df = df[df["attendance"].notna()].copy()
    df = df.dropna(subset=["venue_capacity"])

    if "reporting_status" in df.columns:
        df = df[df["reporting_status"] == "reported"]

    dropped = before - len(df)

    if dropped:
        logger.debug("Dropped %d rows with null venue_capacity", dropped)

    return df


def load_unlabelled(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where *attendance* is null (rows that need a prediction)."""
    return df[df["attendance"].isna()].copy()


# ── Model training ────────────────────────────────────────────────────────────

def fit_model(model: Any,X_train: np.ndarray,y_train: pd.Series,) -> Any:
    """Fit *model* on the training data and return it.
 
    Returning the fitted model keeps callers flexible: they can call
    ``.predict()`` for inference or serialize the object for export.
    """
    model.fit(X_train, y_train)
    return model