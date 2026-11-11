import logging
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from src.features.preprocessing import build_model_pipeline
from src.steps.preprocessing import load_splits
from src.utils.metrics import cross_val_metrics_summary, save_metrics
from src.utils.plotting import plot_model_comparison, save_figure
from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    BASELINE_RESULTS_PATH,
    REPORTS_FIGURES_DIR,
    TARGET_CLASSES,
)

logger = logging.getLogger(__name__)

SCORING = {
    "accuracy": "accuracy",
    "weighted_f1": "f1_weighted",
    "macro_f1": "f1_macro",
    "weighted_precision": "precision_weighted",
    "weighted_recall": "recall_weighted",
    "roc_auc_ovr": "roc_auc_ovr",
}


def define_baseline_classifiers() -> dict:
    return {
        "Logistic Regression": {
            "classifier": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight=None,
            ),
            "scale_numerical": True,
        },
        "HistGradientBoosting": {
            "classifier": HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "scale_numerical": False,
        },
        "Random Forest": {
            "classifier": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                class_weight=None,
            ),
            "scale_numerical": False,
        },
        "XGBoost": {
            "classifier": XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
                verbosity=0,
            ),
            "scale_numerical": False,
        },
    }


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    classifiers: dict,
    cv_folds: int = CV_FOLDS,
) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    le = LabelEncoder()
    le.fit(TARGET_CLASSES)

    for name, config in classifiers.items():
        logger.info(f"Cross-validating: {name}")
        pipeline = build_model_pipeline(
            config["classifier"],
            scale_numerical=config["scale_numerical"],
            include_engineered=True,
        )

        if name == "XGBoost":
            y_encoded = pd.Series(le.transform(y_train), index=y_train.index)
            cv_res = cross_validate(
                pipeline, X_train, y_encoded,
                cv=cv, scoring=SCORING, return_train_score=False, n_jobs=-1,
            )
        else:
            cv_res = cross_validate(
                pipeline, X_train, y_train,
                cv=cv, scoring=SCORING, return_train_score=False, n_jobs=-1,
            )

        summary = cross_val_metrics_summary(cv_res)
        results[name] = summary
        logger.info(
            f"{name} — Weighted F1: {summary['test_weighted_f1']['mean']:.4f} "
            f"(±{summary['test_weighted_f1']['std']:.4f}), ROC-AUC: {summary['test_roc_auc_ovr']['mean']:.4f}"
        )

    return results


