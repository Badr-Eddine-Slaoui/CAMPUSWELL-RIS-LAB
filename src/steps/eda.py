import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from src.data.load_data import load_dataset
from src.data.validate_data import (
    check_missing_values,
    check_duplicates,
    check_target_categories,
    check_categorical_values,
    check_numerical_ranges,
)
from src.utils.metrics import save_metrics
from src.utils.plotting import (
    plot_target_distribution,
    plot_correlation_heatmap,
    save_figure,
)
from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    BINARY_FEATURES,
    REPORTS_FIGURES_DIR,
    REPORTS_TABLES_DIR,
    REPORTS_METRICS_DIR,
    STUDENT_ID_COLUMN,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def load_and_inspect(path: Path = DATA_PATH) -> pd.DataFrame:
    df = load_dataset(path)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    report = check_missing_values(df)
    logger.info(f"Columns with missing values:\n{report}")
    return report


def analyze_duplicates(df: pd.DataFrame) -> dict:
    report = check_duplicates(df, id_column=STUDENT_ID_COLUMN)
    logger.info(f"Duplicate analysis: {report}")
    return report


def analyze_target_distribution(df: pd.DataFrame) -> dict:
    report = check_target_categories(df)
    logger.info(f"Target distribution: {report['class_counts']}")
    return report


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    stats_df = df[numeric_cols].describe().T
    stats_df["median"] = df[numeric_cols].median()
    stats_df["iqr"] = stats_df["75%"] - stats_df["25%"]
    stats_df["skewness"] = df[numeric_cols].skew()
    stats_df["kurtosis"] = df[numeric_cols].kurt()
    return stats_df


def compute_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    pearson_corr = df[numeric_cols].corr(method="pearson")
    spearman_corr = df[numeric_cols].corr(method="spearman")
    return pearson_corr, spearman_corr


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    clean_df = df[numeric_cols].dropna()
    vif_records = []
    for col in numeric_cols:
        x_other = clean_df.drop(columns=[col])
        y_col = clean_df[col]
        lr = LinearRegression().fit(x_other, y_col)
        r2 = lr.score(x_other, y_col)
        vif_val = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        vif_records.append({"feature": col, "vif": round(float(vif_val), 4), "r_squared": round(float(r2), 4)})
    return pd.DataFrame(vif_records).sort_values("vif", ascending=False)


def compute_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    results = []
    for col in numeric_cols:
        groups = [grp[col].dropna() for _, grp in df.groupby(TARGET_COLUMN)]
        stat, pval = stats.kruskal(*groups)
        results.append({"feature": col, "kruskal_stat": round(float(stat), 4), "p_value": float(pval)})
    return pd.DataFrame(results).sort_values("kruskal_stat", ascending=False)


def compute_cramers_v(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ORDINAL_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES
    results = []
    for col in cat_cols:
        if col in df.columns:
            valid = df[[col, TARGET_COLUMN]].dropna()
            ct = pd.crosstab(valid[col], valid[TARGET_COLUMN])
            chi2, pval, dof, _ = stats.chi2_contingency(ct)
            n = ct.sum().sum()
            r, k = ct.shape
            v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
            results.append({"categorical_feature": col, "chi2": round(float(chi2), 4), "p_value": float(pval), "cramers_v": round(float(v), 4)})
    return pd.DataFrame(results).sort_values("cramers_v", ascending=False)


def generate_eda_figures(df: pd.DataFrame, figures_dir: Path = REPORTS_FIGURES_DIR) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_target_distribution(df[TARGET_COLUMN], save_path=figures_dir / "target_distribution.png")
    plt.close(fig)

    fig = plot_correlation_heatmap(df, save_path=figures_dir / "correlation_heatmap.png")
    plt.close(fig)

    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    spearman_corr = df[numeric_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Spearman Rank Correlation Heatmap")
    plt.tight_layout()
    save_figure(fig, figures_dir / "spearman_correlation_heatmap.png")
    plt.close(fig)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:10]):
        df[col].hist(ax=axes[i], bins=25, edgecolor="black", color="#2b5c8f", alpha=0.75)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    save_figure(fig, figures_dir / "numerical_distributions.png")
    plt.close(fig)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:10]):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=col, order=["Low", "Medium", "High"], palette="Set2", ax=axes[i], hue=TARGET_COLUMN, legend=False)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Mental Health Risk")
    plt.tight_layout()
    save_figure(fig, figures_dir / "features_by_risk_category.png")
    plt.close(fig)

    cat_cols = ORDINAL_FEATURES + BINARY_FEATURES
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols[:4]):
        ct = pd.crosstab(df[col], df[TARGET_COLUMN], normalize="index")[["Low", "Medium", "High"]] * 100
        ct.plot(kind="bar", stacked=True, ax=axes[i], colormap="viridis", edgecolor="black")
        axes[i].set_title(f"Risk Distribution by {col}", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Percentage (%)")
        axes[i].legend(title="Risk Level")
        axes[i].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    save_figure(fig, figures_dir / "categorical_by_risk.png")
    plt.close(fig)

    logger.info(f"EDA figures saved to: {figures_dir}")


def generate_eda_report(df: pd.DataFrame, tables_dir: Path = REPORTS_TABLES_DIR, metrics_dir: Path = REPORTS_METRICS_DIR) -> dict:
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    missing_report = analyze_missing_values(df)
    duplicate_report = analyze_duplicates(df)
    target_report = analyze_target_distribution(df)
    stats_df = compute_descriptive_stats(df)
    pearson_corr, spearman_corr = compute_correlations(df)
    vif_df = compute_vif(df)
    kruskal_df = compute_kruskal_wallis(df)
    cramers_df = compute_cramers_v(df)

    missing_report.to_csv(tables_dir / "missing_values_report.csv")
    stats_df.to_csv(tables_dir / "descriptive_statistics.csv")
    pearson_corr.to_csv(tables_dir / "pearson_correlations.csv")
    spearman_corr.to_csv(tables_dir / "spearman_correlations.csv")
    vif_df.to_csv(tables_dir / "vif_report.csv", index=False)
    kruskal_df.to_csv(tables_dir / "kruskal_wallis_report.csv", index=False)
    cramers_df.to_csv(tables_dir / "cramers_v_report.csv", index=False)

    discrepancy_social_screen = int((df["Social_Media_Hours"] > df["Screen_Time_Hours"]).sum())
    age_17_count = int((df["Age"] == 17).sum())

    summary = {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "exact_duplicates": int(duplicate_report.get("exact_duplicates", 0)),
        "duplicate_ids": int(duplicate_report.get("duplicate_ids", 0)),
        "target_distribution": {k: int(v) for k, v in target_report.get("class_counts", {}).items()},
        "top_vif_features": vif_df.head(3).to_dict(orient="records"),
        "top_kruskal_separators": kruskal_df.head(5).to_dict(orient="records"),
        "top_categorical_associations": cramers_df.to_dict(orient="records"),
        "survey_self_reporting_inconsistencies": {
            "social_media_exceeds_screen_time_count": discrepancy_social_screen,
            "students_age_17_count": age_17_count
        }
    }
    with open(metrics_dir / "eda_statistical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Comprehensive EDA statistical artifacts exported.")
    return summary


def pipeline() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    logger.info("=== Advanced EDA Pipeline Started ===")

    df = load_and_inspect()
    generate_eda_figures(df)
    summary = generate_eda_report(df)

    logger.info("=== Advanced EDA Pipeline Completed ===")
    return {"status": "completed", "summary": summary}


