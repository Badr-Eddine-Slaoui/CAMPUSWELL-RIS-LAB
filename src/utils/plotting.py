import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

PALETTE = "Set2"
FIGURE_DPI = 150


def save_figure(fig, path: Path, dpi: int = FIGURE_DPI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved to: {path}")


def plot_target_distribution(y: pd.Series, save_path: Path = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = y.value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Mental Health Risk — Class Distribution")
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Count")
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height())}",
            (bar.get_x() + bar.get_width() / 2., bar.get_height()),
            ha="center", va="bottom", fontsize=10,
        )
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Numerical Feature Correlation Matrix")
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = "mean_weighted_f1", save_path: Path = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=comparison_df, x="model", y=metric, hue="model", palette=PALETTE, legend=False, ax=ax)
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_feature_importance(importance_series: pd.Series, top_n: int = 15, save_path: Path = None) -> plt.Figure:
    top = importance_series.nlargest(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    if save_path:
        save_figure(fig, save_path)
    return fig

