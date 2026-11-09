import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from src.config import TARGET_CLASSES

logger = logging.getLogger(__name__)


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES).tolist(),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            ))
        except Exception as exc:
            logger.warning(f"Could not compute ROC-AUC: {exc}")
    return metrics


def cross_val_metrics_summary(cv_results: dict) -> dict:
