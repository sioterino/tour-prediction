import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.modeling.models import get_regression_models
from src.utils.data import load_dataset
from src.utils.logger import setup_logger
from src.utils.save import save_dataset


logger = setup_logger(__name__)

METRICS_PATH = Path("E:\\code\\tour-prediction\\data\\results\\metrics.csv")
OUTPUT_DIR = Path("E:\\code\\tour-prediction\\data\\predictions")
PROCESSED_DATASETS = {
    "v1_baseline": Path("E:\\code\\tour-prediction\\data\\processed\\attendance_v1_baseline.csv"),
    "v2_artist": Path("E:\\code\\tour-prediction\\data\\processed\\attendance_v2_artist.csv"),
    "v3_artist_geo": Path("E:\\code\\tour-prediction\\data\\processed\\attendance_v3_artist_geo.csv"),
    "v4_artist_geo_time": Path("E:\\code\\tour-prediction\\data\\processed\\attendance_v4_artist_geo_time.csv"),
    "v5_full": Path("E:\\code\\tour-prediction\\data\\processed\\attendance_v5_full.csv"),
    "v6_no_artist": Path("E:\\code\\tour-prediction\\data\\processed\\attendance_v6_no_artist.csv"),
}
LEAKY_COLS = ["fill_rate", "box_score", "avg_ticket_price"]


def sanitize_file_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value.replace(" ", "_"))


def choose_best_experiment(metrics_path: Path) -> tuple[str, str]:
    if not metrics_path.exists():
        logger.error("Metrics file not found: %s", metrics_path)
        raise FileNotFoundError(metrics_path)

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        logger.error("Metrics file is empty: %s", metrics_path)
        raise ValueError("Metrics file is empty")

    best = metrics.sort_values(
        by=["MAE", "R2", "RMSE"],
        ascending=[True, False, True],
        ignore_index=True,
    ).iloc[0]

    logger.info(
        "Selected best experiment: model=%s, dataset=%s, R2=%.4f, RMSE=%.4f, MAE=%.4f",
        best["model"],
        best["dataset"],
        best["R2"],
        best["RMSE"],
        best["MAE"],
    )
    return best["model"], best["dataset"]


def preprocess_features(
    train_df: pd.DataFrame, predict_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df.drop(columns=["attendance"])
    X_predict = predict_df.drop(columns=["attendance"])

    X_train = X_train.drop(columns=[col for col in LEAKY_COLS if col in X_train.columns], errors="ignore")
    X_predict = X_predict.drop(columns=[col for col in LEAKY_COLS if col in X_predict.columns], errors="ignore")

    combined = pd.concat([X_train, X_predict], axis=0)
    combined = pd.get_dummies(combined, drop_first=True)

    X_train_processed = combined.loc[X_train.index].copy()
    X_predict_processed = combined.loc[X_predict.index].copy()

    return X_train_processed, X_predict_processed


def train_and_predict(
    model_name: str,
    dataset_path: Path,
) -> pd.DataFrame:
    dataset = load_dataset(dataset_path)

    if "attendance" not in dataset.columns:
        logger.error("Processed dataset missing 'attendance' column: %s", dataset_path)
        raise KeyError("attendance column missing")

    train_df = dataset[dataset["attendance"].notna()].copy()
    predict_df = dataset[dataset["attendance"].isna()].copy()

    if predict_df.empty:
        logger.warning("No rows with null attendance found in %s", dataset_path)
        raise ValueError("No attendance-null rows found")

    logger.info("Training rows: %d, prediction rows: %d", len(train_df), len(predict_df))

    train_df = train_df.dropna(subset=["venue_capacity"])
    if train_df.empty:
        logger.error("No training rows remain after dropping rows with null venue_capacity")
        raise ValueError("No valid training rows")

    X_train, X_predict = preprocess_features(train_df, predict_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    models = get_regression_models()
    if model_name not in models:
        logger.error("Model name not found in model registry: %s", model_name)
        raise KeyError(model_name)

    model = models[model_name]
    y_train = train_df["attendance"]

    logger.info("Training model %s on %s", model_name, dataset_path.name)
    model.fit(X_train_scaled, y_train)

    logger.info("Predicting attendance for %d rows", len(X_predict_scaled))
    predictions = model.predict(X_predict_scaled)

    output_df = predict_df.copy()
    output_df["attendance"] = predictions

    return output_df


def main() -> None:
    logger.info("Starting prediction using best experiment results")

    model_name, dataset_key = choose_best_experiment(METRICS_PATH)

    if dataset_key not in PROCESSED_DATASETS:
        logger.error("Unknown dataset key from metrics: %s", dataset_key)
        raise KeyError(dataset_key)

    dataset_path = PROCESSED_DATASETS[dataset_key]
    output_df = train_and_predict(model_name, dataset_path)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    file_name = f"{sanitize_file_name(model_name)}-{dataset_key}-{timestamp}.csv"

    save_dataset(output_df, OUTPUT_DIR, file_name)

    logger.info("Prediction output saved to %s/%s", OUTPUT_DIR, file_name)


if __name__ == "__main__":
    main()
