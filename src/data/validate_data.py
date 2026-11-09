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


