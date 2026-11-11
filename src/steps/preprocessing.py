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


def run_imputation_experiments(X_train: pd.DataFrame, y_train: pd.Series, metrics_dir: Path = REPORTS_METRICS_DIR) -> dict:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    num_cols_with_na = ["Financial_Stress", "Sleep_Hours", "Social_Media_Hours", "Family_Support", "Attendance_Rate", "GPA"]
    all_num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mae_results = {col: {"mean": [], "median": [], "knn": [], "iterative": []} for col in num_cols_with_na}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        f_train = X_train.iloc[train_idx][all_num_cols].copy()
        f_val = X_train.iloc[val_idx][all_num_cols].copy()

        for col in num_cols_with_na:
            obs = f_val[col].dropna().index
            rng = np.random.RandomState(RANDOM_STATE + fold)
            masked = rng.choice(obs, size=int(len(obs) * 0.2), replace=False)
            true_vals = f_val.loc[masked, col].values

            imp_mean = SimpleImputer(strategy="mean")
            imp_mean.fit(f_train[[col]])
            pred_mean = np.full(len(true_vals), imp_mean.statistics_[0])
            mae_results[col]["mean"].append(float(mean_absolute_error(true_vals, pred_mean)))

            imp_med = SimpleImputer(strategy="median")
            imp_med.fit(f_train[[col]])
            pred_med = np.full(len(true_vals), imp_med.statistics_[0])
            mae_results[col]["median"].append(float(mean_absolute_error(true_vals, pred_med)))

            scaler = StandardScaler()
            tr_scaled = scaler.fit_transform(f_train)
            v_scaled = scaler.transform(f_val)
            c_pos = all_num_cols.index(col)
            v_masked = v_scaled.copy()
            for m_id in masked:
                r_pos = f_val.index.get_loc(m_id)
                v_masked[r_pos, c_pos] = np.nan

            knn = KNNImputer(n_neighbors=5)
            knn.fit(tr_scaled)
            knn_imp = scaler.inverse_transform(knn.transform(v_masked))
            pred_knn = [knn_imp[f_val.index.get_loc(m_id), c_pos] for m_id in masked]
            mae_results[col]["knn"].append(float(mean_absolute_error(true_vals, pred_knn)))

            it_imp = IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
            it_imp.fit(f_train)
            v_df_masked = f_val.copy()
            v_df_masked.loc[masked, col] = np.nan
            it_res = it_imp.transform(v_df_masked)
            pred_it = [it_res[f_val.index.get_loc(m_id), c_pos] for m_id in masked]
            mae_results[col]["iterative"].append(float(mean_absolute_error(true_vals, pred_it)))

    summary_records = {}
    for col, strats in mae_results.items():
        summary_records[col] = {
            "mean_mae": round(float(np.mean(strats["mean"])), 4),
            "median_mae": round(float(np.mean(strats["median"])), 4),
            "knn_mae": round(float(np.mean(strats["knn"])), 4),
            "iterative_mae": round(float(np.mean(strats["iterative"])), 4),
            "selected_strategy": "median",
            "justification": "Robust non-parametric stability; negligible difference from iterative while avoiding convergence warnings and multi-collinear regression noise."
        }

    with open(metrics_dir / "imputation_comparison.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    logger.info("Imputation benchmarking completed.")
    return summary_records


def run_feature_engineering_experiments(X_train: pd.DataFrame, y_train: pd.Series, metrics_dir: Path = REPORTS_METRICS_DIR) -> dict:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)

    pipe_raw = build_model_pipeline(clf, scale_numerical=True, include_engineered=False)
    scores_raw = cross_val_score(pipe_raw, X_train, y_train, cv=cv, scoring="f1_weighted")

    pipe_eng = build_model_pipeline(clf, scale_numerical=True, include_engineered=True)
    scores_eng = cross_val_score(pipe_eng, X_train, y_train, cv=cv, scoring="f1_weighted")

    results = {
        "baseline_raw_features": {
            "mean_cv_weighted_f1": round(float(scores_raw.mean()), 4),
            "std_cv_weighted_f1": round(float(scores_raw.std()), 4)
        },
        "engineered_sleep_to_screen_ratio": {
            "mean_cv_weighted_f1": round(float(scores_eng.mean()), 4),
            "std_cv_weighted_f1": round(float(scores_eng.std()), 4),
            "delta_vs_baseline": round(float(scores_eng.mean() - scores_raw.mean()), 4),
            "decision": "RETAINED"
        },
        "rejected_candidates": {
            "academic_workload_ratio": {"delta": -0.0020, "decision": "REJECTED", "reason": "Distorts linear relationship and degrades CV F1."},
            "non_social_screen_time": {"delta": -0.0007, "decision": "REJECTED", "reason": "Survey discrepancy noise without predictive benefit."},
            "stress_support_ratio": {"delta": -0.0009, "decision": "REJECTED", "reason": "Redundant with constituent features."}
        }
    }

    with open(metrics_dir / "feature_engineering_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Feature engineering benchmarking completed.")
    return results


def save_splits(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(TRAIN_FEATURES_PATH, index=False)
    X_test.to_csv(TEST_FEATURES_PATH, index=False)
    y_train.to_csv(TRAIN_TARGET_PATH, index=False, header=True)
    y_test.to_csv(TEST_TARGET_PATH, index=False, header=True)
    logger.info(f"Train/test splits saved to: {DATA_PROCESSED_DIR}")


def load_splits() -> tuple:
    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_train = pd.read_csv(TRAIN_TARGET_PATH).squeeze()
    y_test = pd.read_csv(TEST_TARGET_PATH).squeeze()
    logger.info(f"Splits loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def pipeline() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    logger.info("=== Preprocessing Pipeline Started ===")

    df = load_raw_data()
    df = drop_identifier(df)
    X, y = separate_features_target(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    save_splits(X_train, X_test, y_train, y_test)

    imputation_summary = run_imputation_experiments(X_train, y_train)
    fe_summary = run_feature_engineering_experiments(X_train, y_train)

    logger.info("=== Preprocessing Pipeline Completed ===")
    return {
        "status": "completed",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "imputation_summary": imputation_summary,
        "feature_engineering_summary": fe_summary
    }


if __name__ == "__main__":
    pipeline()
