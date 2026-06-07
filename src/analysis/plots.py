"""Plotting utilities for regression analysis of tour attendance datasets."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from pandas import DataFrame

# ── Shared style ────────────────────────────────────────────────────────────

_PALETTE = "steelblue"
_CAT_PALETTE = "Set2"


def _section_title(ax, dataset_name: str, title: str) -> None:
    """Stamp dataset name + plot title on an axes object."""
    ax.set_title(f"{title}\n[{dataset_name}]", fontsize=13, fontweight="bold", pad=12)


# ── Target analysis ──────────────────────────────────────────────────────────


def plot_target_distribution(
    df: DataFrame, dataset_name: str, target: str = "attendance"
) -> None:
    """
    Histogram + KDE of the target, plus a log-scale version side-by-side.

    Attendance data is almost always right-skewed; the log panel reveals
    whether a log-transform of the target would help the models.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
        target: Name of the target column (default: "attendance").
    """
    if target not in df.columns:
        print(f"Warning: '{target}' column not found in dataset.")
        return

    data = df[target].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    sns.histplot(data, bins=40, kde=True, ax=axes[0], color=_PALETTE, edgecolor="white", alpha=0.8)
    axes[0].set_xlabel(target, fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _section_title(axes[0], dataset_name, "Target Distribution (linear)")

    # Log scale – useful to spot if log-transform would normalise the target
    log_data = np.log1p(data)
    sns.histplot(log_data, bins=40, kde=True, ax=axes[1], color="coral", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel(f"log(1 + {target})", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    _section_title(axes[1], dataset_name, "Target Distribution (log scale)")

    plt.tight_layout()
    plt.show()


# ── Feature–target relationships ─────────────────────────────────────────────


def plot_numeric_vs_target(
    df: DataFrame, dataset_name: str, target: str = "attendance"
) -> None:
    """
    Scatter plots of every numeric feature against the target.

    This is the most important regression EDA plot: it shows whether
    relationships are linear, curved, or noisy, and flags outliers.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
        target: Name of the target column (default: "attendance").
    """
    if target not in df.columns:
        print(f"Warning: '{target}' column not found in dataset.")
        return

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target
    ]
    if not numeric_cols:
        print("No numeric feature columns found.")
        return

    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.scatter(df[col], df[target], alpha=0.35, s=18, color=_PALETTE, edgecolors="none")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel(target, fontsize=10)
        ax.set_title(f"{col} vs {target}", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Numeric Features vs Target  [{dataset_name}]", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_attendance_by_category(
    df: DataFrame, dataset_name: str, column: str, target: str = "attendance"
) -> None:
    """
    Box plot of attendance grouped by a categorical feature.

    Reveals which categories drive high/low attendance and how much
    within-group variance exists — key context for tree-based splits.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
        column: Name of the categorical column to group by.
        target: Name of the target column (default: "attendance").
    """
    if column not in df.columns:
        return  # silently skip missing columns (e.g. artist_name not in v1)

    order = (
        df.groupby(column)[target]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.6 + 2), 6))
    sns.boxplot(data=df, x=column, y=target, order=order, palette=_CAT_PALETTE, ax=ax)
    ax.set_xlabel(column, fontsize=11)
    ax.set_ylabel(target, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    _section_title(ax, dataset_name, f"Attendance by {column}")
    plt.tight_layout()
    plt.show()


def plot_attendance_by_year(df: DataFrame, dataset_name: str, target: str = "attendance") -> None:
    """
    Mean attendance per year with a sample-count annotation.

    Useful for spotting temporal trends the time-based dataset versions
    (v4, v5) are trying to capture.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
        target: Name of the target column (default: "attendance").
    """
    if "show_year" not in df.columns:
        return

    yearly = df.groupby("show_year")[target].agg(["mean", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly["show_year"], yearly["mean"], marker="o", linewidth=2,
            markersize=7, color="steelblue", label="Mean attendance")

    # Annotate sample counts so thin years are obvious
    for _, row in yearly.iterrows():
        ax.annotate(
            f"n={int(row['count'])}",
            xy=(row["show_year"], row["mean"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="gray",
        )

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Mean Attendance", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(alpha=0.25)
    ax.legend()
    _section_title(ax, dataset_name, "Mean Attendance by Year")
    plt.tight_layout()
    plt.show()


# ── Correlation ───────────────────────────────────────────────────────────────


def plot_correlation_heatmap(df: DataFrame, dataset_name: str, target: str = "attendance") -> None:
    """
    Correlation heatmap of numeric columns, with the target column highlighted.

    Args:
        df: The DataFrame containing the data.
        dataset_name: Name of the dataset for the plot title.
        target: Name of the target column (default: "attendance").
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return

    # Put target first so it's easy to scan
    if target in numeric_cols:
        numeric_cols = [target] + [c for c in numeric_cols if c != target]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols) * 1.1), max(6, len(numeric_cols) * 0.9)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    _section_title(ax, dataset_name, "Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ── Post-training diagnostics ─────────────────────────────────────────────────


def plot_predicted_vs_actual(
    y_true, y_pred, model_name: str, dataset_name: str, target: str = "attendance"
) -> None:
    """
    Scatter of predicted vs. actual values with a perfect-prediction line.

    Points on the diagonal = perfect predictions. Systematic curves or fans
    reveal bias or heteroscedasticity you should address.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        model_name: Name of the model for the plot title.
        dataset_name: Name of the dataset for the plot title.
        target: Label for axis names.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color=_PALETTE, edgecolors="none", label="Predictions")

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel(f"Actual {target}", fontsize=11)
    ax.set_ylabel(f"Predicted {target}", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    ax.grid(alpha=0.2)
    _section_title(ax, f"{dataset_name} | {model_name}", "Predicted vs Actual")
    plt.tight_layout()
    plt.show()


def plot_residuals(
    y_true, y_pred, model_name: str, dataset_name: str
) -> None:
    """
    Residual plot (residuals vs. predicted) + residual distribution side-by-side.

    A good model shows residuals scattered randomly around zero.
    Patterns (fans, curves, clusters) signal problems worth fixing.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        model_name: Name of the model for the plot title.
        dataset_name: Name of the dataset for the plot title.
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=18, color="coral", edgecolors="none")
    axes[0].axhline(0, color="black", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Residual (actual − predicted)", fontsize=11)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[0].grid(alpha=0.2)
    _section_title(axes[0], f"{dataset_name} | {model_name}", "Residuals vs Predicted")

    # Residual distribution
    sns.histplot(residuals, bins=40, kde=True, ax=axes[1], color="coral", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Residual", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _section_title(axes[1], f"{dataset_name} | {model_name}", "Residual Distribution")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    feature_names, importances, model_name: str, dataset_name: str, top_n: int = 20
) -> None:
    """
    Horizontal bar chart of feature importances (top N).

    Works with any model that exposes a `.feature_importances_` attribute
    (Decision Tree, Random Forest). Not applicable to MLP.

    Args:
        feature_names: List of feature names.
        importances: Array of importance scores (from model.feature_importances_).
        model_name: Name of the model for the plot title.
        dataset_name: Name of the dataset for the plot title.
        top_n: How many top features to show (default: 20).
    """
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    bars = ax.barh(names[::-1], values[::-1], color=_PALETTE, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    # Annotate bar values
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="dimgray")

    _section_title(ax, f"{dataset_name} | {model_name}", f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()