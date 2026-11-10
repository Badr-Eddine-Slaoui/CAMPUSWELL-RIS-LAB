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

