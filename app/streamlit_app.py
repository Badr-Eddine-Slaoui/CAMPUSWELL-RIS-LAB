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


def page_overview():
    st.title("🧠 CampusWell RiskLab")
    st.subheader("Student Mental Health Risk Analysis & Prediction")

    render_disclaimer()

    st.markdown(
        "**CampusWell RiskLab** is an end-to-end Machine Learning classification project that analyzes "
        "student academic, behavioral, lifestyle, and psychosocial indicators to classify their "
        "**synthetic** mental health risk level into three categories: **Low**, **Medium**, and **High**.\n\n"
        "This project was developed as a YouCode assignment to demonstrate the complete ML workflow: "
        "Data → EDA → Preprocessing → Modeling → Optimization → Evaluation → Deployment."
    )

    df = load_dataset_cached()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", str(len(df.columns) - 2))
        with col3:
            st.metric("Target Classes", "3")
        with col4:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.subheader("Target Class Distribution")
        counts = df["Mental_Health_Risk"].value_counts().reset_index()
        counts.columns = ["Risk Category", "Count"]
        counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
        fig = px.pie(
            counts, names="Risk Category", values="Count",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Mental Health Risk Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(counts, use_container_width=True)
    else:
        st.error(f"Dataset not found at: `{DATA_PATH}`. Please follow the instructions in `data/README.md`.")


def page_eda():
    st.title("📊 Exploratory Data Analysis")
    render_disclaimer()

    df = load_dataset_cached()
    if df is None:
        st.error("Dataset not available.")
        return

    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 5 Records**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("**Data Types**")
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Column", "Type"]
        st.dataframe(dtype_df, use_container_width=True)

    st.subheader("Missing Value Analysis")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    if not missing_df.empty:
        fig = px.bar(
            missing_df.reset_index(), x="index", y="Missing %",
            title="Missing Values by Column (%)",
            color="Missing %",
            color_continuous_scale="Reds",
            labels={"index": "Column"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("No missing values found.")

    st.subheader("Numerical Feature Distributions")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if "Student_ID" in num_cols:
        num_cols.remove("Student_ID")
    selected_num = st.selectbox("Select feature:", num_cols)
    fig = px.histogram(
        df, x=selected_num, color="Mental_Health_Risk",
        barmode="overlay", opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Distribution of {selected_num} by Risk Category",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_df = df[num_cols].copy()
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Feature Correlation Matrix", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)


