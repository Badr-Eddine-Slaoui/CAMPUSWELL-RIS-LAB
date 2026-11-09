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


def check_categorical_values(df: pd.DataFrame) -> dict:
    issues = {}
    if "Gender" in df.columns:
        expected_gender = {"Male", "Female", "Non-binary"}
        actual_gender = set(df["Gender"].dropna().unique())
        unexpected = actual_gender - expected_gender
        if unexpected:
            issues["Gender"] = {"unexpected_values": list(unexpected)}

    if "Physical_Activity" in df.columns:
        expected_pa = set(PHYSICAL_ACTIVITY_ORDER)
        actual_pa = set(df["Physical_Activity"].dropna().unique())
        unexpected = actual_pa - expected_pa
        if unexpected:
            issues["Physical_Activity"] = {"unexpected_values": list(unexpected)}

    if "Diet_Quality" in df.columns:
        expected_dq = set(DIET_QUALITY_ORDER)
        actual_dq = set(df["Diet_Quality"].dropna().unique())
        unexpected = actual_dq - expected_dq
        if unexpected:
            issues["Diet_Quality"] = {"unexpected_values": list(unexpected)}

    for col in ["Part_Time_Job", "Previous_Mental_Health_History"]:
        if col in df.columns:
            expected = {"Yes", "No"}
            actual = set(df[col].dropna().unique())
            unexpected = actual - expected
            if unexpected:
                issues[col] = {"unexpected_values": list(unexpected)}
    return issues


def check_numerical_ranges(df: pd.DataFrame) -> dict:
    expected_ranges = {
        "Age": (17, 35),
        "Year_of_Study": (1, 5),
        "Sleep_Hours": (3.0, 12.0),
        "Study_Hours_Per_Week": (0, 60),
        "GPA": (0.0, 4.0),
        "Social_Media_Hours": (0.0, 15.0),
        "Screen_Time_Hours": (0.0, 20.0),
        "Financial_Stress": (1.0, 10.0),
        "Family_Support": (1.0, 10.0),
        "Attendance_Rate": (0.0, 100.0),
    }
    flags = {}
    for col, (low, high) in expected_ranges.items():
        if col in df.columns:
            series = df[col].dropna()
            below = int((series < low).sum())
            above = int((series > high).sum())
            if below > 0 or above > 0:
                flags[col] = {
                    "expected_range": (low, high),
                    "actual_min": float(series.min()),
                    "actual_max": float(series.max()),
                    "below_min_count": below,
                    "above_max_count": above,
                }
    return flags


def run_full_validation(df: pd.DataFrame) -> dict:
    logger.info("Running full data validation...")
    report = {
        "column_check": check_required_columns(df),
