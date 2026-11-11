import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from src.data.load_data import load_dataset
from src.features.preprocessing import (
    build_model_pipeline,
    FeatureEngineer,
)
from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    STUDENT_ID_COLUMN,
    NUMERICAL_FEATURES,
    ENGINEERED_NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    BINARY_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    TEST_FEATURES_PATH,
    TEST_TARGET_PATH,
    DATA_PROCESSED_DIR,
    REPORTS_METRICS_DIR,
)

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


def drop_identifier(df: pd.DataFrame) -> pd.DataFrame:
    if STUDENT_ID_COLUMN in df.columns:
        df = df.drop(columns=[STUDENT_ID_COLUMN])
        logger.info(f"Dropped identifier column: {STUDENT_ID_COLUMN}")
    return df


def separate_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    logger.info(f"Features: {X.shape}, Target: {y.shape}")
    return X, y


def split_dataset(X: pd.DataFrame, y: pd.Series) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


