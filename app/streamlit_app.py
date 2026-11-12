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


def page_model_performance():
    st.title("🏆 Model Performance")
    render_disclaimer()

    metadata = load_json_metrics(MODEL_METADATA_PATH)
    final_metrics = load_json_metrics(FINAL_METRICS_PATH)
    baseline_results = load_json_metrics(BASELINE_RESULTS_PATH)
    opt_results = load_json_metrics(OPTIMIZATION_RESULTS_PATH)

    if metadata is None or final_metrics is None:
        st.warning(
            "Model evaluation results not yet available. "
            "Please run the complete pipeline first by executing the four notebooks "
            "in order, or running `src/steps/evaluation.py`."
        )
        return

    st.subheader(f"Production Model: **{metadata['model_name']}**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{final_metrics['accuracy']:.4f}")
    with col2:
        st.metric("Weighted F1", f"{final_metrics['weighted_f1']:.4f}")
    with col3:
        st.metric("Macro F1", f"{final_metrics['macro_f1']:.4f}")
    with col4:
        if "roc_auc_ovr" in final_metrics:
            st.metric("ROC-AUC (OvR)", f"{final_metrics['roc_auc_ovr']:.4f}")

    if baseline_results and opt_results:
        st.subheader("Baseline vs. Optimized Comparison")
        comparison_data = opt_results.get("comparison_table", [])
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)

    st.subheader("Classification Report")
    cls_report = final_metrics.get("classification_report", {})
    if cls_report:
        report_rows = []
        for label, stats in cls_report.items():
            if isinstance(stats, dict):
                report_rows.append({
                    "Class": label,
                    "Precision": round(stats.get("precision", 0), 4),
                    "Recall": round(stats.get("recall", 0), 4),
                    "F1-Score": round(stats.get("f1-score", 0), 4),
                    "Support": int(stats.get("support", 0)),
                })
        if report_rows:
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

    st.subheader("Confusion Matrix")
    cm = final_metrics.get("confusion_matrix")
    if cm:
        cm_array = np.array(cm)
        fig = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=TARGET_CLASSES,
            y=TARGET_CLASSES,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="Confusion Matrix — Final Test Set",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best Hyperparameters")
    best_params = metadata.get("best_hyperparameters", {})
    if best_params:
        params_df = pd.DataFrame(list(best_params.items()), columns=["Parameter", "Value"])
        st.dataframe(params_df, use_container_width=True)


def page_prediction_playground():
    st.title("🎮 Prediction Playground")
    render_disclaimer()

    st.markdown(
        "Enter values for a hypothetical student profile and the trained model will predict "
        "their **synthetic** mental health risk category.\n\n"
        "> ⚠️ This is an educational simulation only. These predictions must not be used "
        "> for any real diagnostic, clinical, or administrative purposes."
    )

    pipeline = load_pipeline_cached()
    if pipeline is None:
        st.warning(
            "The trained model is not yet available. "
            "Please complete the full pipeline execution first."
        )
        return

    with st.form("prediction_form"):
        st.subheader("Student Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=17, max_value=35, value=21, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            year_of_study = st.number_input("Year of Study", min_value=1, max_value=5, value=2, step=1)
            sleep_hours = st.slider("Sleep Hours (nightly)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
            study_hours = st.slider("Study Hours Per Week", min_value=0.0, max_value=60.0, value=20.0, step=0.5)

        with col2:
            gpa = st.slider("GPA (4.0 scale)", min_value=0.0, max_value=4.0, value=2.7, step=0.01)
            social_media = st.slider("Social Media Hours (daily)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
            screen_time = st.slider("Screen Time Hours (daily)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
            physical_activity = st.selectbox("Physical Activity", PHYSICAL_ACTIVITY_ORDER)
            financial_stress = st.slider("Financial Stress (1–10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

        with col3:
            family_support = st.slider("Family Support (1–10)", min_value=1.0, max_value=10.0, value=6.0, step=0.5)
            attendance_rate = st.slider("Attendance Rate (%)", min_value=30.0, max_value=100.0, value=85.0, step=0.5)
            part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
            diet_quality = st.selectbox("Diet Quality", DIET_QUALITY_ORDER)
            prev_mental_health = st.selectbox("Previous Mental Health History", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Risk Category", type="primary")

    if submitted:
        input_data = pd.DataFrame([{
            "Age": float(age),
            "Gender": gender,
            "Year_of_Study": float(year_of_study),
            "Sleep_Hours": float(sleep_hours),
            "Study_Hours_Per_Week": float(study_hours),
            "GPA": float(gpa),
            "Social_Media_Hours": float(social_media),
            "Screen_Time_Hours": float(screen_time),
            "Physical_Activity": physical_activity,
            "Financial_Stress": float(financial_stress),
            "Family_Support": float(family_support),
            "Attendance_Rate": float(attendance_rate),
            "Part_Time_Job": part_time_job,
            "Diet_Quality": diet_quality,
            "Previous_Mental_Health_History": prev_mental_health,
        }])

        try:
            prediction = pipeline.predict(input_data)
            predicted_class = str(prediction[0])

            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(input_data)[0]
                classes = pipeline.classes_
            else:
                probabilities = None
                classes = None

            st.divider()
            st.subheader("Prediction Result")

            color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            emoji = color_map.get(predicted_class, "⚪")

            st.markdown(f"### {emoji} Predicted Risk Category: **{predicted_class}**")

            if probabilities is not None and classes is not None:
                st.subheader("Class Probabilities")
                proba_df = pd.DataFrame({
                    "Risk Category": [str(c) for c in classes],
                    "Probability": probabilities,
                })
                fig = px.bar(
                    proba_df, x="Risk Category", y="Probability",
                    color="Risk Category",
                    color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                    title="Prediction Probability by Risk Category",
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            st.info(
                "🔬 **Reminder:** This prediction is generated from a model trained on "
                "**synthetic data** and is for **educational demonstration only**. "
                "It is not a clinical assessment and should never be used to make "
                "decisions about a real student's mental health."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def main():
    st.set_page_config(
        page_title="CampusWell RiskLab",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🧠 CampusWell RiskLab")
    st.sidebar.markdown("---")

    pages = {
        "🏠 Overview": page_overview,
        "📊 Exploratory Data Analysis": page_eda,
        "🏆 Model Performance": page_model_performance,
        "🎮 Prediction Playground": page_prediction_playground,
    }

    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project:** YouCode 2026–2027")
    st.sidebar.markdown("**Dataset:** Synthetic — Educational Only")

    pages[selected_page]()


if __name__ == "__main__":
    main()

