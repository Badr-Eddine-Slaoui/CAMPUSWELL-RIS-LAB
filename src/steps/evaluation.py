import logging
import datetime
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from src.features.preprocessing import build_model_pipeline
from src.steps.preprocessing import load_splits
from src.utils.metrics import compute_classification_metrics, save_metrics, load_metrics
from src.utils.serialization import save_pipeline, save_metadata
from src.utils.plotting import plot_confusion_matrix, plot_feature_importance, save_figure
from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    FINAL_METRICS_PATH,
    OPTIMIZATION_RESULTS_PATH,
    REPORTS_FIGURES_DIR,
    TARGET_CLASSES,
)

logger = logging.getLogger(__name__)

SCORING = {
    "accuracy": "accuracy",
    "weighted_f1": "f1_weighted",
    "macro_f1": "f1_macro",
}


def load_optimization_results() -> dict:
    if not OPTIMIZATION_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Optimization results not found at: {OPTIMIZATION_RESULTS_PATH}")
    return load_metrics(OPTIMIZATION_RESULTS_PATH)


def _build_production_pipeline(model_name: str, best_params: dict):
    if model_name == "Logistic Regression":
        clf = LogisticRegression(
            random_state=RANDOM_STATE,
            C=float(best_params.get("classifier__C", 1.0)),
            solver=best_params.get("classifier__solver", "lbfgs"),
            max_iter=int(best_params.get("classifier__max_iter", 1000)),
        )
        return build_model_pipeline(clf, scale_numerical=True)

    if model_name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            learning_rate=float(best_params.get("classifier__learning_rate", 0.1)),
            max_iter=int(best_params.get("classifier__max_iter", 100)),
            max_leaf_nodes=int(best_params.get("classifier__max_leaf_nodes", 31)),
            min_samples_leaf=int(best_params.get("classifier__min_samples_leaf", 20)),
            l2_regularization=float(best_params.get("classifier__l2_regularization", 0.0)),
        )
        return build_model_pipeline(clf, scale_numerical=False)

    if model_name == "Random Forest":
        clf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=int(best_params.get("classifier__n_estimators", 100)),
            max_depth=best_params.get("classifier__max_depth", None),
            min_samples_split=int(best_params.get("classifier__min_samples_split", 2)),
            min_samples_leaf=int(best_params.get("classifier__min_samples_leaf", 1)),
            max_features=best_params.get("classifier__max_features", "sqrt"),
        )
        return build_model_pipeline(clf, scale_numerical=False)

    if model_name == "XGBoost":
        clf = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            verbosity=0,
            n_estimators=int(best_params.get("classifier__n_estimators", 100)),
            learning_rate=float(best_params.get("classifier__learning_rate", 0.1)),
            max_depth=int(best_params.get("classifier__max_depth", 5)),
            subsample=float(best_params.get("classifier__subsample", 0.8)),
            colsample_bytree=float(best_params.get("classifier__colsample_bytree", 0.8)),
            gamma=float(best_params.get("classifier__gamma", 0.0)),
        )
        return build_model_pipeline(clf, scale_numerical=False)

    raise ValueError(f"Unknown model: {model_name}")


def train_final_pipeline(model_name: str, best_params: dict, X_train: pd.DataFrame, y_train: pd.Series):
    production_pipeline = _build_production_pipeline(model_name, best_params)

    if model_name == "XGBoost":
        le = LabelEncoder()
        le.fit(TARGET_CLASSES)
        y_fit = pd.Series(le.transform(y_train), index=y_train.index)
    else:
        y_fit = y_train

    logger.info(f"Training final pipeline: {model_name}")
    production_pipeline.fit(X_train, y_fit)
    return production_pipeline


def evaluate_on_test_set(pipeline, model_name: str, X_test: pd.DataFrame, y_test: pd.Series):
    if model_name == "XGBoost":
        le = LabelEncoder()
        le.fit(TARGET_CLASSES)
        y_pred_encoded = pipeline.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None
    else:
        y_pred = pipeline.predict(X_test)
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    logger.info(f"Final test evaluation — Weighted F1: {metrics['weighted_f1']:.4f}")
    return metrics, y_pred


