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


def get_dataset_info(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 ** 2),
    }

