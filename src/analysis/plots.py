"""Plotting utilities for analyzing tour attendance datasets."""

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


def plot_target_distribution(
    df: DataFrame, dataset_name: str, target: str = "attendance"
) -> None:
    """
    Plot histogram and KDE distribution of the target column.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
        target: Name of the target column (default: "attendance").
    """
    if target not in df.columns:
        print(f"Warning: '{target}' column not found in dataset.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(df[target], bins=30, kde=True, edgecolor="black", alpha=0.7)
    plt.title(f"Distribution of {target} - {dataset_name}", fontsize=14, fontweight="bold")
    plt.xlabel(target, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: DataFrame, dataset_name: str) -> None:
    """
    Plot a correlation heatmap of numeric columns.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    
    if len(numeric_cols) < 2:
        print("Warning: Less than 2 numeric columns available for correlation analysis.")
        return

    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Correlation Heatmap - {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_attendance_by_category(
    df: DataFrame, dataset_name: str, column: str
) -> None:
    """
    Plot boxplot of attendance grouped by a categorical column.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
        column: Name of the categorical column to group by.
    """
    if column not in df.columns:
        print(f"Warning: '{column}' column not found in dataset.")
        return

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=column, y="attendance", palette="Set2")
    plt.title(
        f"Attendance Distribution by {column} - {dataset_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Attendance", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_attendance_by_year(df: DataFrame, dataset_name: str) -> None:
    """
    Plot mean attendance per show year as a line plot.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
    """
    if "show_year" not in df.columns:
        print("Warning: 'show_year' column not found in dataset.")
        return

    yearly_attendance = df.groupby("show_year")["attendance"].agg(["mean", "count"])
    
    plt.figure(figsize=(10, 6))
    plt.plot(
        yearly_attendance.index,
        yearly_attendance["mean"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )
    plt.title(
        f"Mean Attendance by Year - {dataset_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Mean Attendance", fontsize=12)
    plt.grid(axis="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
