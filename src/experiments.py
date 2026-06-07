"""Experiment runner: train all 18 model-dataset combinations and save results."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATASETS, METRICS_PATH, RESULTS_DIR
from src.modeling.metrics import calculate_regression_metrics
from src.modeling.models import get_regression_models
from src.modeling.pipeline import build_feature_matrices, load_labelled
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def run() -> None:
    """Run all 18 experiments (6 datasets x 3 models) and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    models_dict = get_regression_models()

    for dataset_name, dataset_path in DATASETS.items():
        logger.info("Processing dataset: %s", dataset_name)

        df = pd.read_csv(dataset_path)
        df = load_labelled(df) 

        y = df["attendance"]
        X = df.drop(columns=["attendance"])

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train, X_test = build_feature_matrices(X_train_raw, X_test_raw)

        for model_name, model in models_dict.items():
            logger.debug("  Training: %s ...", model_name)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = calculate_regression_metrics(y_test, y_pred)

            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )

            logger.info(
                "  %-35s  MAE=%.2f  RMSE=%.2f  R2=%.4f",
                model_name,
                metrics["MAE"],
                metrics["RMSE"],
                metrics["R2"],
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_PATH, index=False)
    logger.info("Metrics saved > %s", METRICS_PATH)

    summary = results_df.sort_values("R2", ascending=False)
    logger.info("\n%s\nSUMMARY: All Results Sorted by R2 (Descending)\n%s\n%s",
                "=" * 80, "=" * 80, summary.to_string(index=False))


if __name__ == "__main__":
    run()
