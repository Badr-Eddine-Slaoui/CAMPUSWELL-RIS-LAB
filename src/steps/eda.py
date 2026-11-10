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
