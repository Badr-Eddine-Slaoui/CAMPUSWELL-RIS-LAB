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
def student_mental_health_dag():

    @task(task_id="eda_task")
    def eda_task() -> dict:
        print("Running EDA task...")
        from src.steps.eda import pipeline as eda_pipeline
        logger.info("Starting EDA task...")
        result = eda_pipeline()
        logger.info(f"EDA task completed: {result}")
        return result

    @task(task_id="preprocessing_task")
    def preprocessing_task() -> dict:
        from src.steps.preprocessing import pipeline as preprocessing_pipeline
        logger.info("Starting Preprocessing task...")
        result = preprocessing_pipeline()
        logger.info(f"Preprocessing task completed: {result}")
        return result

    @task(task_id="modeling_task")
    def modeling_task() -> dict:
        from src.steps.modeling import pipeline as modeling_pipeline
        logger.info("Starting Modeling task...")
        result = modeling_pipeline()
        logger.info(f"Modeling task completed: {result}")
        return result

    @task(task_id="optimization_task")
    def optimization_task() -> dict:
        from src.steps.optimization import pipeline as optimization_pipeline
        logger.info("Starting Optimization task...")
        result = optimization_pipeline()
        logger.info(f"Optimization task completed: {result}")
        return result

    @task(task_id="evaluation_task")
    def evaluation_task() -> dict:
        from src.steps.evaluation import pipeline as evaluation_pipeline
        logger.info("Starting Evaluation task...")
        result = evaluation_pipeline()
        logger.info(f"Evaluation task completed: {result}")
        return result

    eda_result = eda_task()
    preprocessing_result = preprocessing_task()
    modeling_result = modeling_task()
    optimization_result = optimization_task()
    evaluation_result = evaluation_task()

    eda_result >> preprocessing_result >> modeling_result >> optimization_result >> evaluation_result


dag_instance = student_mental_health_dag()

