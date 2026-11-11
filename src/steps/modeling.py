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


def build_comparison_dataframe(cv_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in cv_results.items():
        row = {"model": model_name}
        for metric_key, stats in metrics.items():
            short_key = metric_key.replace("test_", "mean_")
            row[short_key] = round(stats["mean"], 4)
            row[short_key.replace("mean_", "std_")] = round(stats["std"], 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_weighted_f1", ascending=False)


def select_candidate_models(comparison_df: pd.DataFrame, top_n: int = 2) -> list:
    candidates = comparison_df.nlargest(top_n, "mean_weighted_f1")["model"].tolist()
    logger.info(f"Selected candidates for optimization: {candidates}")
    return candidates


def save_baseline_results(cv_results: dict, comparison_df: pd.DataFrame, candidates: list) -> None:
    BASELINE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cv_results": cv_results,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "selected_candidates": candidates,
    }
    save_metrics(payload, BASELINE_RESULTS_PATH)

    comparison_df.to_csv(BASELINE_RESULTS_PATH.parent / "baseline_comparison_table.csv", index=False)

    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_model_comparison(comparison_df, metric="mean_weighted_f1")
    save_figure(fig, REPORTS_FIGURES_DIR / "baseline_model_comparison.png")
    logger.info("Baseline results and figures saved.")


def pipeline() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    logger.info("=== Modeling Pipeline Started ===")

    X_train, X_test, y_train, y_test = load_splits()
    classifiers = define_baseline_classifiers()
    cv_results = run_cross_validation(X_train, y_train, classifiers)
    comparison_df = build_comparison_dataframe(cv_results)
    candidates = select_candidate_models(comparison_df)
    save_baseline_results(cv_results, comparison_df, candidates)

    logger.info("=== Modeling Pipeline Completed ===")
    return {
        "status": "completed",
        "models_evaluated": list(classifiers.keys()),
        "selected_candidates": candidates,
        "best_model": comparison_df.iloc[0]["model"],
        "best_weighted_f1": comparison_df.iloc[0]["mean_weighted_f1"],
    }


if __name__ == "__main__":
    pipeline()
