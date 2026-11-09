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

TRAIN_FEATURES_PATH = DATA_PROCESSED_DIR / "X_train.csv"
TRAIN_TARGET_PATH = DATA_PROCESSED_DIR / "y_train.csv"
TEST_FEATURES_PATH = DATA_PROCESSED_DIR / "X_test.csv"
TEST_TARGET_PATH = DATA_PROCESSED_DIR / "y_test.csv"

BASELINE_RESULTS_PATH = REPORTS_METRICS_DIR / "baseline_results.json"
OPTIMIZATION_RESULTS_PATH = REPORTS_METRICS_DIR / "optimization_results.json"
FINAL_METRICS_PATH = REPORTS_METRICS_DIR / "final_evaluation_metrics.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

STUDENT_ID_COLUMN = "Student_ID"

NUMERICAL_FEATURES = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

ENGINEERED_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ["sleep_to_screen_ratio"]

ORDINAL_FEATURES = ["Physical_Activity", "Diet_Quality"]

NOMINAL_FEATURES = ["Gender"]

BINARY_FEATURES = ["Part_Time_Job", "Previous_Mental_Health_History"]

PHYSICAL_ACTIVITY_ORDER = ["Low", "Moderate", "High"]

DIET_QUALITY_ORDER = ["Poor", "Average", "Good"]

TARGET_CLASSES = ["Low", "Medium", "High"]

REQUIRED_COLUMNS = (
    [STUDENT_ID_COLUMN]
    + NUMERICAL_FEATURES
    + ORDINAL_FEATURES
    + NOMINAL_FEATURES
    + BINARY_FEATURES
    + [TARGET_COLUMN]
)


def validate_config():
    missing = []
    if not DATA_PATH.exists():
        missing.append(f"Dataset not found at: {DATA_PATH}")
    if missing:
        return False, missing
    return True, []