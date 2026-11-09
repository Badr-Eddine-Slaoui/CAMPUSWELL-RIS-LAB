import logging
from pathlib import Path
import pandas as pd
from src.config import DATA_PATH

logger = logging.getLogger(__name__)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Please place the CSV file at the configured DATA_PATH.\n"
            "See data/README.md for instructions."
        )
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

