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


def analyze_errors(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    error_df = X_test.copy()
    error_df["true_label"] = y_test.values
    error_df["predicted_label"] = y_pred
    error_df["is_error"] = error_df["true_label"] != error_df["predicted_label"]
    errors = error_df[error_df["is_error"]].copy()
    logger.info(f"Total misclassified samples: {len(errors)} / {len(X_test)}")
    logger.info(f"Error breakdown:\n{errors.groupby(['true_label', 'predicted_label']).size()}")
    return errors


def generate_evaluation_figures(test_metrics: dict, y_pred: np.ndarray) -> None:
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = np.array(test_metrics["confusion_matrix"])
    fig = plot_confusion_matrix(cm, class_names=TARGET_CLASSES)
    save_figure(fig, REPORTS_FIGURES_DIR / "final_confusion_matrix.png")


def pipeline() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    logger.info("=== Evaluation Pipeline Started ===")

    X_train, X_test, y_train, y_test = load_splits()
    opt_results = load_optimization_results()

    production_candidate = opt_results["selected_production_candidate"]
    best_params = opt_results["optimized_results"][production_candidate]["best_params"]
    optimized_cv_f1 = opt_results["optimized_results"][production_candidate]["best_cv_score"]

    logger.info(f"Production candidate: {production_candidate}")
    logger.info(f"Best params: {best_params}")

    final_pipeline = train_final_pipeline(production_candidate, best_params, X_train, y_train)
    test_metrics, y_pred = evaluate_on_test_set(final_pipeline, production_candidate, X_test, y_test)
    errors = analyze_errors(X_test, y_test, y_pred)
    generate_evaluation_figures(test_metrics, y_pred)

    save_pipeline(final_pipeline, MODEL_PATH)

    metadata = {
        "model_name": production_candidate,
        "best_hyperparameters": best_params,
        "optimized_cv_weighted_f1": optimized_cv_f1,
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_roc_auc_ovr": test_metrics.get("roc_auc_ovr"),
        "target_classes": TARGET_CLASSES,
        "training_date": datetime.datetime.now().isoformat(),
        "total_errors": len(errors),
        "total_test_samples": len(X_test),
    }
    save_metadata(metadata, MODEL_METADATA_PATH)
    save_metrics(test_metrics, FINAL_METRICS_PATH)

    logger.info("=== Evaluation Pipeline Completed ===")
    return {
        "status": "completed",
        "production_model": production_candidate,
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "model_saved_to": str(MODEL_PATH),
    }


if __name__ == "__main__":
    pipeline()
