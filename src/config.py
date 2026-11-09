from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_NAME = os.getenv("PROJECT_NAME", "CampusWell-RiskLab")

DATA_PATH = PROJECT_ROOT / os.getenv("DATA_PATH", "data/raw/student_mental_health_dataset.csv")

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Mental_Health_Risk")

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))

MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/production_pipeline.joblib")

REPORTS_DIR = PROJECT_ROOT / os.getenv("REPORTS_DIR", "reports")

AIRFLOW_HOME = PROJECT_ROOT / os.getenv("AIRFLOW_HOME", ".airflow")

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"

