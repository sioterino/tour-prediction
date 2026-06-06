"""Experiment runner: train all 18 model-dataset combinations and save results."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.modeling.models import get_regression_models
from src.modeling.metrics import calculate_regression_metrics


DATASETS = {
    "v1_baseline": "E:\\code\\tour-prediction\\data\\processed\\attendance_v1_baseline.csv",
    "v2_artist": "E:\\code\\tour-prediction\\data\\processed\\attendance_v2_artist.csv",
    "v3_artist_geo": "E:\\code\\tour-prediction\\data\\processed\\attendance_v3_artist_geo.csv",
    "v4_artist_geo_time": "E:\\code\\tour-prediction\\data\\processed\\attendance_v4_artist_geo_time.csv",
    "v5_full": "E:\\code\\tour-prediction\\data\\processed\\attendance_v5_full.csv",
    "v6_no_artist": "E:\\code\\tour-prediction\\data\\processed\\attendance_v6_no_artist.csv",
}

LEAKY_COLS = [
    "fill_rate",
    "box_score",
    "avg_ticket_price"
]


def run() -> None:
    """Run all 18 experiments (6 datasets x 3 models) and save results."""
    os.makedirs("results", exist_ok=True)

    results = []
    models_dict = get_regression_models()

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\nProcessing dataset: {dataset_name}")

        # Load dataset
        df = pd.read_csv(dataset_path)

        # Drop rows where attendance is null
        df = df.dropna(subset=["attendance"])

        # Drop rows where venue_capacity is null
        df = df.dropna(subset=["venue_capacity"])

        # Extract target and features
        y = df["attendance"]
        X = df.drop(columns=["attendance"])

        # Drop leaky columns (only if they exist)
        cols_to_drop = [col for col in LEAKY_COLS if col in X.columns]
        X = X.drop(columns=cols_to_drop)

        # One-hot encode categorical columns
        X = pd.get_dummies(X, drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and evaluate each model
        for model_name, model in models_dict.items():
            print(f"  Training: {model_name}...", end=" ")

            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = calculate_regression_metrics(y_test, y_pred)

            # Store results
            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )

            print(f"MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/metrics.csv", index=False)

    # Print summary table sorted by R2 descending
    print("\n" + "="*80)
    print("SUMMARY: All Results Sorted by R2 (Descending)")
    print("="*80)
    summary = results_df.sort_values("R2", ascending=False)
    print(summary.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    run()
