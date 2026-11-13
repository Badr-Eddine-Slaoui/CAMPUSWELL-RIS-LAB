from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from airflow.sdk import dag, task
except ImportError:
    from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "Badr Eddine Slaoui",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="student_mental_health_pipeline",
    description=(
        "CampusWell RiskLab — End-to-end ML pipeline for student mental health "
        "risk classification. Covers EDA, preprocessing, baseline modeling, "
        "hyperparameter optimization, and final evaluation."
    ),
    schedule="@weekly",
    start_date=datetime(2026, 11, 13),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["campuswell", "ml", "mental-health", "classification"],
)
