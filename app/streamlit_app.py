import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    MODEL_PATH,
    DATA_PATH,
    FINAL_METRICS_PATH,
    MODEL_METADATA_PATH,
    BASELINE_RESULTS_PATH,
    OPTIMIZATION_RESULTS_PATH,
    REPORTS_FIGURES_DIR,
    TARGET_CLASSES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    BINARY_FEATURES,
    PHYSICAL_ACTIVITY_ORDER,
    DIET_QUALITY_ORDER,
)

DISCLAIMER = (
    "⚠️ **Educational Disclaimer:** This application uses a **synthetic dataset** "
    "generated for educational and portfolio purposes only. All predictions are simulations "
    "based on artificial data. They do **not** represent real individuals and must **never** "
    "be used for clinical diagnosis, medical screening, disciplinary action, or any "
    "real-world decision-making about students' mental health."
)


@st.cache_data
def load_dataset_cached():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_pipeline_cached():
    try:
        from src.utils.serialization import load_pipeline
        return load_pipeline(MODEL_PATH)
    except Exception:
        return None


@st.cache_data
def load_json_metrics(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def render_disclaimer():
    st.warning(DISCLAIMER)


