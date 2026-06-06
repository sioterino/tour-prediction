"""Dataset reporting utilities for analyzing tour attendance datasets."""

import pandas as pd
from pandas import DataFrame


def print_basic_info(df: DataFrame, dataset_name: str) -> None:
    """
    Print basic information about the dataset.

    Includes shape, columns, data types, and summary statistics.

    Args:
        df: The DataFrame to analyze.
        dataset_name: Name of the dataset for display purposes.
    """
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns ({df.shape[1]}):")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    print(f"\nDataFrame Info:")
    df.info()
    print(f"\nBasic Statistics:")
    print(df.describe())


def print_missing_values(df: DataFrame) -> None:
    """
    Print missing value counts for columns that have missing values.

    Args:
        df: The DataFrame to analyze.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("\nNo missing values found.")
    else:
        print(f"\nMissing Values ({len(missing)} columns with gaps):")
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} ({pct:.2f}%)")


def print_target_summary(df: DataFrame, target: str = "attendance") -> None:
    """
    Print summary statistics of the target column.

    Args:
        df: The DataFrame to analyze.
        target: Name of the target column (default: "attendance").
    """
    if target not in df.columns:
        print(f"\nWarning: '{target}' column not found in dataset.")
        return

    target_data = df[target]

    print(f"\nTarget Column Summary: '{target}'")
    print(f"  - Count: {target_data.count()}")
    print(f"  - Mean: {target_data.mean():.2f}")
    print(f"  - Median: {target_data.median():.2f}")
    print(f"  - Std Dev: {target_data.std():.2f}")
    print(f"  - Min: {target_data.min():.2f}")
    print(f"  - Max: {target_data.max():.2f}")
    print(f"  - Q1 (25%): {target_data.quantile(0.25):.2f}")
    print(f"  - Q3 (75%): {target_data.quantile(0.75):.2f}")
    print(f"  - Missing: {target_data.isnull().sum()}")
