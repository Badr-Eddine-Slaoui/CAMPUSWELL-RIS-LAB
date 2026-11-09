import logging
import pandas as pd
from src.config import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TARGET_CLASSES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    BINARY_FEATURES,
    PHYSICAL_ACTIVITY_ORDER,
    DIET_QUALITY_ORDER,
)

logger = logging.getLogger(__name__)


def check_required_columns(df: pd.DataFrame) -> dict:
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return {"present": present, "missing": missing, "extra": extra}


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    })
    return report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def check_duplicates(df: pd.DataFrame, id_column: str = "Student_ID") -> dict:
    total_duplicates = df.duplicated().sum()
    id_duplicates = df[id_column].duplicated().sum() if id_column in df.columns else None
    return {
        "total_duplicate_rows": int(total_duplicates),
        "duplicate_ids": int(id_duplicates) if id_duplicates is not None else "N/A",
    }


def check_target_categories(df: pd.DataFrame) -> dict:
    if TARGET_COLUMN not in df.columns:
        return {"error": f"{TARGET_COLUMN} not found"}
    actual_classes = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    expected_classes = sorted(TARGET_CLASSES)
    unexpected = [c for c in actual_classes if c not in expected_classes]
    missing_classes = [c for c in expected_classes if c not in actual_classes]
    return {
        "actual_classes": actual_classes,
        "expected_classes": expected_classes,
        "unexpected_classes": unexpected,
        "missing_classes": missing_classes,
        "class_counts": df[TARGET_COLUMN].value_counts().to_dict(),
    }

