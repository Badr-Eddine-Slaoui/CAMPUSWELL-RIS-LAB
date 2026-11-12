import logging
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from src.features.preprocessing import build_model_pipeline
from src.steps.preprocessing import load_splits
from src.utils.metrics import save_metrics, load_metrics
from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    BASELINE_RESULTS_PATH,
    OPTIMIZATION_RESULTS_PATH,
    TARGET_CLASSES,
)

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "f1_weighted"


def load_baseline_candidates() -> list:
    if not BASELINE_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Baseline results not found at: {BASELINE_RESULTS_PATH}")
    baseline = load_metrics(BASELINE_RESULTS_PATH)
    candidates = baseline.get("selected_candidates", [])
    logger.info(f"Loaded candidates from baseline: {candidates}")
    return candidates


def get_param_distributions(model_name: str) -> dict:
    if model_name == "Logistic Regression":
        return {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0],
            "classifier__solver": ["lbfgs", "saga"],
            "classifier__max_iter": [500, 1000, 2000],
        }
    if model_name == "HistGradientBoosting":
        return {
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_iter": [50, 100, 150, 200],
            "classifier__max_leaf_nodes": [15, 31, 63],
            "classifier__min_samples_leaf": [10, 20, 30],
            "classifier__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        }
    if model_name == "Random Forest":
        return {
            "classifier__n_estimators": [50, 100, 200, 300],
            "classifier__max_depth": [None, 5, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        }
    if model_name == "XGBoost":
        return {
            "classifier__n_estimators": [100, 200, 300, 500],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "classifier__max_depth": [3, 4, 5, 6, 8],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__gamma": [0, 0.1, 0.2, 0.5],
        }
    raise ValueError(f"No parameter distributions defined for: {model_name}")


def get_grid_params(model_name: str, best_params: dict) -> dict:
    if model_name == "Logistic Regression":
        c_val = float(best_params.get("classifier__C", 1.0))
        c_candidates = sorted(list(set([max(0.005, round(c_val * 0.5, 4)), c_val, round(c_val * 1.5, 4)])))
        return {
            "classifier__C": c_candidates,
            "classifier__solver": [best_params.get("classifier__solver", "lbfgs")],
            "classifier__max_iter": [best_params.get("classifier__max_iter", 1000)],
        }
    if model_name == "HistGradientBoosting":
        lr = float(best_params.get("classifier__learning_rate", 0.1))
        m_iter = int(best_params.get("classifier__max_iter", 100))
        return {
            "classifier__learning_rate": [max(0.01, round(lr * 0.7, 3)), lr, round(lr * 1.3, 3)],
            "classifier__max_iter": [m_iter],
            "classifier__max_leaf_nodes": [best_params.get("classifier__max_leaf_nodes", 31)],
            "classifier__min_samples_leaf": [best_params.get("classifier__min_samples_leaf", 20)],
            "classifier__l2_regularization": [best_params.get("classifier__l2_regularization", 0.0)],
        }
    if model_name == "Random Forest":
        n = best_params.get("classifier__n_estimators", 100)
        return {
            "classifier__n_estimators": [max(50, n - 50), n, n + 50],
            "classifier__max_depth": [best_params.get("classifier__max_depth", None)],
            "classifier__min_samples_split": [best_params.get("classifier__min_samples_split", 2)],
            "classifier__min_samples_leaf": [best_params.get("classifier__min_samples_leaf", 1)],
            "classifier__max_features": [best_params.get("classifier__max_features", "sqrt")],
        }
    if model_name == "XGBoost":
        lr = best_params.get("classifier__learning_rate", 0.1)
        n = best_params.get("classifier__n_estimators", 100)
        return {
            "classifier__n_estimators": [max(50, n - 50), n, n + 100],
            "classifier__learning_rate": [lr / 2, lr, lr * 2],
            "classifier__max_depth": [best_params.get("classifier__max_depth", 5)],
            "classifier__subsample": [best_params.get("classifier__subsample", 0.8)],
            "classifier__colsample_bytree": [best_params.get("classifier__colsample_bytree", 0.8)],
            "classifier__gamma": [best_params.get("classifier__gamma", 0)],
        }
    raise ValueError(f"No grid params defined for: {model_name}")


def _make_base_classifier(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=RANDOM_STATE)
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE)
    if model_name == "XGBoost":
        return XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _needs_label_encoding(model_name: str) -> bool:
    return model_name == "XGBoost"


def run_randomized_search(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15,
) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scale = model_name == "Logistic Regression"
    pipeline = build_model_pipeline(_make_base_classifier(model_name), scale_numerical=scale)

    if _needs_label_encoding(model_name):
        le = LabelEncoder()
        le.fit(TARGET_CLASSES)
        y_fit = pd.Series(le.transform(y_train), index=y_train.index)
    else:
        y_fit = y_train

    param_distributions = get_param_distributions(model_name)
    start = time.time()
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=PRIMARY_METRIC,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_fit)
    elapsed = time.time() - start
    logger.info(
        f"[{model_name}] RandomizedSearchCV done in {elapsed:.1f}s — "
        f"Best CV {PRIMARY_METRIC}: {search.best_score_:.4f}"
    )
    logger.info(f"[{model_name}] Best params: {search.best_params_}")
    return search


def run_grid_search(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scale = model_name == "Logistic Regression"
    pipeline = build_model_pipeline(_make_base_classifier(model_name), scale_numerical=scale)

    if _needs_label_encoding(model_name):
        le = LabelEncoder()
        le.fit(TARGET_CLASSES)
        y_fit = pd.Series(le.transform(y_train), index=y_train.index)
    else:
        y_fit = y_train

    grid = get_grid_params(model_name, best_params)
    start = time.time()
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        cv=cv,
        scoring=PRIMARY_METRIC,
        refit=True,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_fit)
    elapsed = time.time() - start
    logger.info(
        f"[{model_name}] GridSearchCV done in {elapsed:.1f}s — "
        f"Best CV {PRIMARY_METRIC}: {search.best_score_:.4f}"
    )
    return search


def compare_with_baseline(candidates: list, optimized_results: dict) -> pd.DataFrame:
    baseline_data = load_metrics(BASELINE_RESULTS_PATH)
    baseline_cv = baseline_data.get("cv_results", {})
    rows = []
    for name in candidates:
        baseline_f1 = baseline_cv.get(name, {}).get("test_weighted_f1", {}).get("mean", None)
        opt = optimized_results.get(name, {})
        rows.append({
            "model": name,
            "baseline_cv_weighted_f1": baseline_f1,
            "optimized_cv_weighted_f1": opt.get("best_cv_score"),
            "best_params": opt.get("best_params"),
            "improvement": (
                opt.get("best_cv_score") - baseline_f1
                if baseline_f1 and opt.get("best_cv_score") else None
            ),
        })
    return pd.DataFrame(rows)


def select_production_candidate(comparison_df: pd.DataFrame) -> str:
    best = comparison_df.loc[comparison_df["optimized_cv_weighted_f1"].idxmax(), "model"]
    logger.info(f"Production candidate selected: {best}")
    return best


