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


