"""Utilities for reading and selecting from experiment results."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def choose_best_model(metrics_path: Path) -> tuple[str, str]:
    """Return ``(model_name, dataset_key)`` for the best-performing model.

    "Best" is defined as lowest MAE, breaking ties by highest R2, then lowest RMSE.
    """
    if not metrics_path.exists():
        logger.error("Metrics file not found: %s", metrics_path)
        exit(1)

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        logger.error("Metrics file is empty: %s", metrics_path)
        exit(1)

    best = metrics.sort_values(
        by=["MAE", "R2", "RMSE"],
        ascending=[True, False, True],
        ignore_index=True,
    ).iloc[0]

    logger.info(
        "Best experiment > model=%s  dataset=%s  R2=%.4f  RMSE=%.4f  MAE=%.4f",
        best["model"], best["dataset"], best["R2"], best["RMSE"], best["MAE"],
    )
    return best["model"], best["dataset"]