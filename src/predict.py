"""Predict attendance for rows without a known value.
 
Workflow
--------
1. Read metrics.csv and pick the best experiment (lowest MAE, highest R2).
2. Re-train that model on *all* labelled rows from its dataset.
3. Predict attendance for every row where attendance is null.
4. Save the result as a timestamped CSV.
"""

import re
from datetime import datetime
from pathlib import Path

from src.config import DATASETS, METRICS_PATH, PREDICTIONS_DIR
from src.modeling.models import get_regression_models
from src.modeling.pipeline import (
    build_feature_matrices,
    fit_model,
    load_labelled,
    load_unlabelled,
)
from src.utils.data import load_dataset
from src.utils.models import choose_best_model
from src.utils.logger import setup_logger
from src.utils.save import save_dataset
from src.utils.sanitizer import sanitize_file_name


logger = setup_logger(__name__)


def main() -> None:
    logger.info("Starting prediction pipeline")
 
    model_name, dataset_key = choose_best_model(METRICS_PATH)
 
    if dataset_key not in DATASETS:
        logger.error("Unknown dataset key in metrics: %s", dataset_key)
        raise KeyError(dataset_key)
 
    dataset_path = DATASETS[dataset_key]
    dataset = load_dataset(dataset_path)
 
    if "attendance" not in dataset.columns:
        logger.error("Dataset missing 'attendance' column: %s", dataset_path)
        raise KeyError("attendance column missing")
 
    train_df   = load_labelled(dataset)
    predict_df = load_unlabelled(dataset)
 
    if predict_df.empty:
        logger.warning("No rows with null attendance in %s", dataset_path)
        raise ValueError("No attendance-null rows found")
 
    logger.info("Training rows: %d  |  Prediction rows: %d", len(train_df), len(predict_df))
 
    models = get_regression_models()
    if model_name not in models:
        logger.error("Unknown model: %s", model_name)
        raise KeyError(model_name)
 
    X_train, X_pred = build_feature_matrices(
        train_df.drop(columns=["attendance"]),
        predict_df.drop(columns=["attendance"]),
    )
 
    logger.info("Training %s on %s", model_name, dataset_path.name)
    model = fit_model(models[model_name], X_train, train_df["attendance"])
 
    output_df = predict_df.copy()
    output_df["attendance"] = model.predict(X_pred)
 
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    file_name = f"{sanitize_file_name(model_name)}-{dataset_key}-{timestamp}.csv"
    save_dataset(output_df, PREDICTIONS_DIR, file_name)
    logger.info("Predictions saved > %s/%s", PREDICTIONS_DIR, file_name)


if __name__ == "__main__":
    main()