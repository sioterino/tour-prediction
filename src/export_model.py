"""Export the best trained model as a .pkl file for use in an API.

Workflow
--------
1. Read metrics.csv and pick the best experiment (lowest MAE, highest R2).
2. Train that model on *all* labelled rows from its dataset.
3. Serialize the fitted model to a timestamped .pkl file.
"""

import re
from datetime import datetime

import joblib

from src.config import DATASETS, METRICS_PATH, MODELS_DIR, MODEL_EXT
from src.modeling.models import get_regression_models
from src.modeling.pipeline import fit_model, load_labelled
from src.modeling.pipeline import build_train_matrix, fit_model, load_labelled
from src.utils.data import load_dataset
from src.utils.models import choose_best_model
from src.utils.logger import setup_logger
from src.utils.sanitizer import sanitize_file_name


logger = setup_logger(__name__)


def main() -> None:
    logger.info("Starting model export pipeline")

    model_name, dataset_key = choose_best_model(METRICS_PATH)

    if dataset_key not in DATASETS:
        logger.error("Unknown dataset key in metrics: %s", dataset_key)
        raise KeyError(dataset_key)

    dataset_path = DATASETS[dataset_key]
    dataset = load_dataset(dataset_path)

    if "attendance" not in dataset.columns:
        logger.error("Dataset missing 'attendance' column: %s", dataset_path)
        raise KeyError("attendance column missing")

    train_df = load_labelled(dataset)
    if train_df.empty:
        logger.error("No labelled rows found in %s", dataset_path)
        raise ValueError("No valid training rows")

    logger.info("Training rows: %d", len(train_df))

    X_train = build_train_matrix(train_df.drop(columns=["attendance"]))

    models = get_regression_models()
    if model_name not in models:
        logger.error("Unknown model: %s", model_name)
        raise KeyError(model_name)

    logger.info("Training %s on %s", model_name, dataset_path.name)
    trained_model = fit_model(models[model_name], X_train, train_df["attendance"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    file_name = f"{sanitize_file_name(model_name)}-{dataset_key}-{timestamp}{MODEL_EXT}"
    output_path = MODELS_DIR / file_name

    joblib.dump(trained_model, output_path)

    logger.info("Model exported → %s", output_path)


if __name__ == "__main__":
    main()
