import logging
import joblib
import json
from pathlib import Path
from src.config import MODEL_PATH

logger = logging.getLogger(__name__)


def save_pipeline(pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Pipeline saved to: {path}")


def load_pipeline(path: Path):
    print(path)
    if not path.exists():
        raise FileNotFoundError(f"Model pipeline not found at: {path}")
    logger.info(f"Loading pipeline from: {path}")
    return joblib.load(path)


