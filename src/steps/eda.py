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


