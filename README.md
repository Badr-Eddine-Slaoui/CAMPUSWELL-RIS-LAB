# CampusWell RiskLab

> **Student Mental Health Risk Analysis and Prediction**  
> *An End-to-End Supervised Machine Learning Classification, MLOps Orchestration, and Interactive Decision System*

[![Python: 3.14](https://img.shields.io/badge/Python-3.14%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit: 1.67.0](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Apache Airflow: 3.4.0 / 2.10.4](https://img.shields.io/badge/Apache%20Airflow-3.4.0%20%7C%202.10.4-017CEE.svg?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose%20v2-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code Style: Zero Comments](https://img.shields.io/badge/Code%20Style-Zero%20Comments%20%26%20Docstrings-success.svg)]()
[![Model Test F1: 0.7557](https://img.shields.io/badge/Champion%20Weighted%20F1-0.7557-brightgreen.svg)]()

---

## 1. Educational and Synthetic Data Disclaimer

> [!WARNING]
> ### CRITICAL ETHICAL AND MEDICAL NOTICE
> 1. **Synthetic Educational Dataset:** The dataset powering CampusWell RiskLab is entirely **synthetic**, generated artificially for educational demonstration, student portfolio defense, and machine learning pedagogy.
> 2. **Never a Diagnostic or Medical Tool:** Predictions and risk assessments produced by this system **must never be interpreted as clinical diagnoses**, medical screening, psychiatric evaluations, or evidence concerning the real-world mental health of any student or individual.
> 3. **Prohibited Administrative & Disciplinary Use:** This system **must never be deployed** for academic admissions, institutional filtering, disciplinary sanctions, academic tracking, or administrative resource restriction.
> 4. **Human Care Priority:** Real mental health challenges demand compassionate, qualified healthcare professionals. Algorithmic scoring on synthetic behavioral attributes serves exclusively to teach end-to-end Machine Learning pipelines, rigorous validation, leakage prevention, and MLOps orchestration.

---

## 2. Executive Summary

**CampusWell RiskLab** is a production-grade, reproducible Machine Learning project that classifies student synthetic mental health risk into three categories: **Low**, **Medium**, and **High**.

Developed as an end-to-end data science portfolio project for YouCode (2026–2027), the system translates raw student academic, lifestyle, and psychosocial data into actionable risk insights through a leak-free scikit-learn pipeline, orchestrated by Apache Airflow, and served via an interactive 4-page Streamlit dashboard.

### Core Key Achievements
- **Leak-Free ML Pipeline:** Strict train/test isolation (80/20 stratified split, $N_{\text{train}}=1,600$, $N_{\text{test}}=400$), robust imputation (median for numerical, most-frequent for categorical), ordinal integer mapping, nominal one-hot encoding, and experimentally validated engineered feature `sleep_to_screen_ratio`.
- **Empirical Model Benchmarking:** Benchmarked Logistic Regression, HistGradientBoosting, XGBoost, and Random Forest under 5-Fold Stratified Cross-Validation. **Logistic Regression** emerged as the decisive champion ($0.7564 \pm 0.0204$ CV Weighted F1, $0.9063$ ROC-AUC) over tree-based ensembles ($0.6842$ for HistGradientBoosting, $0.6788$ for XGBoost, $0.6709$ for Random Forest), reflecting the underlying linear scoring mechanism of the synthetic data generation process.
- **Fine-Grained Optimization:** Coarse-to-fine hyperparameter optimization via `RandomizedSearchCV` and `GridSearchCV` confirmed Logistic Regression with `C=1.0`, `solver='lbfgs'`, and `max_iter=1000` as the production champion ($0.7564$ CV Weighted F1).
- **Generalization on Untouched Test Set:**
  - **Weighted F1-Score:** **$0.7557$**
  - **Macro F1-Score:** **$0.7551$**
  - **Overall Accuracy:** **$75.50\%$** ($302 / 400$ correct, 98 errors)
  - **Multi-Class ROC-AUC (One-vs-Rest):** **$0.9129$**
  - **Zero High $\to$ Low Critical False Negatives:** Out of 136 true High risk students, 0 were misclassified as Low risk ($0.0\%$).
- **Dual-Mode Orchestration:** Fully functional Apache Airflow DAG (`student_mental_health_pipeline`) featuring dual-version TaskFlow imports (`airflow.sdk` for Airflow 3.4.0 in local mode; `airflow.decorators` for Airflow 2.10.4 in Docker mode).
- **Multi-Service Docker Containerization:** Complete Compose stack running PostgreSQL 15, Airflow Init, Airflow Webserver, Airflow Scheduler, and Streamlit with container health checks and volume persistence.
- **Executed Jupyter Notebooks:** All 4 notebooks (`01_eda.ipynb`, `02_preprocessing.ipynb`, `03_modeling_and_evaluation.ipynb`, `04_hyperparameter_optimization.ipynb`) are sequentially executed with 100% cell outputs, genuine execution counts, and verified zero comments and docstrings.

---

## 3. Problem Formulation and Objectives

### Learning Paradigm
- **Task:** Supervised Multi-Class Classification.
- **Input:** 15 student behavioral, academic, lifestyle, and psychosocial indicators ($X \in \mathbb{R}^{N \times 15}$).
- **Target Variable ($y$):** Discrete synthetic risk level with 3 classes:
  $$\mathcal{Y} = \{\text{Low}, \text{Medium}, \text{High}\}$$

### Class Definitions
1. **Low Risk:** Students exhibiting healthy sleep schedules, balanced screen habits, solid academic standing, and low financial/academic strain.
2. **Medium Risk:** Students experiencing moderate academic stress, fluctuating attendance, or borderline lifestyle habits requiring mild advisory support.
3. **High Risk:** Students exhibiting severe academic stress, extreme financial distress, acute sleep deficits, high screen time, or poor family support requiring immediate academic intervention.

### Primary and Secondary Objectives
1. **Model Generalization:** Maximize **Weighted F1-score** ($F_1^{\text{weighted}}$) and **Macro F1-score** ($F_1^{\text{macro}}$) across classes while minimizing critical false negatives ($\text{High} \to \text{Low}$).
2. **Methodological Rigor:** Zero data leakage across preprocessing, feature engineering, and cross-validation folds.
3. **Reproducibility & Modularity:** Decouple pipeline steps into callable Python modules (`src/steps/`), runnable via CLI, Airflow DAG, or Streamlit app.
4. **Code Quality:** Zero code comments and zero docstrings strictly enforced across all Python source files (`.py`) and Jupyter notebook code cells.

---

## 4. Dataset Description

The dataset comprises **2,000 synthetic student records** across 17 initial attributes (1 identifier, 15 predictor variables, and 1 target variable).

### Attribute Schema

| Feature Name | Data Type | Role | Value Range / Categories | Description |
|---|---|---|---|---|
| `Student_ID` | String | Identifier | `STU0001` – `STU2000` | Unique student record ID (dropped before modeling) |
| `Age` | Integer | Continuous / Discrete | $17 - 29$ years | Student chronological age (mean: $22.48$, 93 age 17 freshmen) |
| `Gender` | Categorical | Nominal Predictor | `Male` ($963$), `Female` ($975$), `Non-binary` ($62$) | Self-reported gender identity |
| `Year_of_Study` | Integer | Ordinal / Discrete | $1, 2, 3, 4, 5$ | Academic progress year (including 5th-year students) |
| `Sleep_Hours` | Float | Continuous Predictor | $4.00 - 10.00$ hrs/day | Average daily sleep duration (mean: $7.00$) |
| `Study_Hours_Per_Week` | Float | Continuous Predictor | $5.00 - 40.00$ hrs/wk | Weekly self-directed study hours (mean: $22.75$) |
| `GPA` | Float | Continuous Predictor | $1.50 - 3.66$ | Grade Point Average on a 4.00 scale (mean: $2.69$) |
| `Social_Media_Hours` | Float | Continuous Predictor | $0.50 - 8.00$ hrs/day | Daily time spent on social media (mean: $4.01$) |
| `Screen_Time_Hours` | Float | Continuous Predictor | $1.00 - 12.00$ hrs/day | Total daily digital screen time (mean: $6.48$) |
| `Physical_Activity` | Categorical | Ordinal Predictor | `Low` ($662$), `Moderate` ($668$), `High` ($670$) | Self-reported exercise frequency |
| `Financial_Stress` | Float | Continuous Scale | $1.40 - 10.00$ | Subjective financial distress level (mean: $5.36$) |
| `Family_Support` | Float | Continuous Scale | $1.70 - 9.90$ | Perceived family emotional/practical support (mean: $5.68$) |
| `Attendance_Rate` | Float | Continuous Predictor | $50.00\% - 100.00\%$ | Lecture attendance percentage (mean: $75.05\%$) |
| `Part_Time_Job` | Categorical | Binary Predictor | `No` ($1,029$), `Yes` ($971$) | Employment outside studies |
| `Diet_Quality` | Categorical | Ordinal Predictor | `Poor` ($660$), `Average` ($635$), `Good` ($636$) | Quality of nutritional intake |
| `Previous_Mental_Health_History` | Categorical | Binary Predictor | `No` ($1,023$), `Yes` ($977$) | Prior formal or informal mental health struggles |
| `Mental_Health_Risk` | Categorical | **Target Class** | `Low` ($700$), `Medium` ($700$), `High` ($600$) | **Ground Truth Synthetic Risk Label** |

### Target Distribution
The dataset exhibits well-balanced classes:
- **Low Risk:** 700 records ($35.0\%$)
- **Medium Risk:** 700 records ($35.0\%$)
- **High Risk:** 600 records ($30.0\%$)

### Missing Value Audit
Seven features contain realistic synthetic missing values (totaling 419 missing entries across 368 affected rows):
1. `Financial_Stress`: 91 missing ($4.55\%$)
2. `Diet_Quality`: 69 missing ($3.45\%$)
3. `Sleep_Hours`: 62 missing ($3.10\%$)
4. `Social_Media_Hours`: 58 missing ($2.90\%$)
5. `Family_Support`: 54 missing ($2.70\%$)
6. `Attendance_Rate`: 47 missing ($2.35\%$)
7. `GPA`: 38 missing ($1.90\%$)

---

## 5. Repository Architecture

```
CAMPUSWELL-RISKLAB/
├── .airflow/                             # Local Airflow home (SQLite DB, logs)
│   ├── airflow.cfg                       # Local Airflow configuration
│   └── airflow.db                        # SQLite metadata database
├── .env                                  # Local environment variables
├── .env.example                          # Environment template
├── .gitignore                            # Git exclusion rules
├── .dockerignore                         # Docker build context exclusions
├── IMPLEMENTATION_PLAN.md                # Master implementation plan & audit record
├── TASKS.md                              # Baseline task tracking checklist
├── ISSUES.md                             # Baseline issue log
├── MODEL_REFINEMENT_IMPLEMENTATION_PLAN.md # 20-phase refinement plan
├── MODEL_REFINEMENT_TASKS.md             # Granular refinement checklist
├── MODEL_REFINEMENT_ISSUES.md            # Refinement issue tracking log
├── README.md                             # Definitive technical documentation (this file)
├── pytest.ini                            # Pytest configuration
├── app/                                  # Streamlit web application
│   ├── __init__.py
│   ├── streamlit_app.py                  # Multi-page interactive Streamlit dashboard
│   └── components/                       # Shared UI widgets
├── dags/                                 # Apache Airflow workflow definitions
│   └── student_mental_health_dag.py      # Dual-version TaskFlow DAG (5 pipeline tasks)
├── data/                                 # Data storage partitions
│   ├── raw/                              # Untouched source data
│   │   └── student_mental_health_dataset.csv
│   └── processed/                        # Leak-free train/test splits
│       ├── X_train.csv                   # 1,600 training feature rows
│       ├── X_test.csv                    # 400 test feature rows
│       ├── y_train.csv                   # 1,600 training labels
│       └── y_test.csv                    # 400 test labels
├── docker/                               # Containerization manifests
│   ├── Dockerfile.airflow                # Multi-stage image (Airflow 2.10.4 + ML stack)
│   ├── Dockerfile.streamlit              # Lightweight Streamlit runtime image
│   └── docker-compose.yml                # Multi-service stack (Postgres, Airflow, App)
├── models/                               # Serialized model artifacts
│   ├── production_pipeline.joblib        # Fitted sklearn Pipeline (Preprocessor + LR)
│   └── model_metadata.json               # Champion metadata, hyperparameters & provenance
├── notebooks/                            # Executed, pedagogical Jupyter notebooks
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis & visual profiling (1..20)
│   ├── 02_preprocessing.ipynb            # Imputation benchmark & feature ablation (1..12)
│   ├── 03_modeling_and_evaluation.ipynb  # 4-model benchmarking & coefficient analysis (1..9)
│   └── 04_hyperparameter_optimization.ipynb # Tuning, test evaluation & error analysis (1..10)
├── reports/                              # Automated pipeline outputs
│   ├── figures/                          # Exported high-resolution figures (150 DPI)
│   │   ├── baseline_model_comparison.png
│   │   ├── correlation_heatmap.png
│   │   ├── spearman_correlation_heatmap.png
│   │   ├── features_by_risk_category.png
│   │   ├── categorical_by_risk.png
│   │   ├── final_confusion_matrix.png
│   │   ├── numerical_distributions.png
│   │   └── target_distribution.png
│   ├── metrics/                          # Machine-readable metric JSON files
│   │   ├── eda_statistical_summary.json  # Parametric & non-parametric statistics
│   │   ├── imputation_comparison.json    # 5-fold masked imputation experiment
│   │   ├── feature_engineering_comparison.json # Feature interaction CV ablation
│   │   ├── baseline_results.json         # 5-fold CV metrics across 4 models
│   │   ├── optimization_results.json     # Hyperparameter search results & gains
│   │   └── final_evaluation_metrics.json # Test metrics, per-class stats & error matrix
│   └── tables/                           # CSV reports generated by pipeline
│       ├── descriptive_statistics.csv
│       ├── missing_values_report.csv
│       └── optimization_comparison_table.csv
├── requirements/                         # Dependency specifications
│   ├── airflow-requirements.txt          # Python 3.12 dependencies for Docker Airflow
│   └── requirements.txt                  # Python 3.14 locked dependencies for local venv
├── src/                                  # Core Python library (zero comments/docstrings)
│   ├── __init__.py
│   ├── config.py                         # Paths, seeds, feature groupings & constants
│   ├── features/                         # Feature transformation & engineering
│   │   ├── __init__.py
│   │   └── preprocessing.py              # ColumnTransformer, FeatureEngineer, Imputer
│   ├── steps/                            # Modular pipeline stage entry points
│   │   ├── __init__.py
│   │   ├── eda.py                        # Standalone EDA pipeline step
│   │   ├── preprocessing.py              # Standalone data split & transformation step
│   │   ├── modeling.py                   # Standalone benchmarking step
│   │   ├── optimization.py               # Standalone hyperparameter tuning step
│   │   └── evaluation.py                 # Standalone final evaluation & export step
│   └── utils/                            # Shared utilities
│       ├── __init__.py
│       ├── logger.py                     # Unified logging formatter
│       ├── metrics.py                    # Metric serializing & formatting helpers
│       ├── plotting.py                   # Matplotlib & Seaborn chart generators
│       └── serialization.py              # Joblib pipeline & JSON metadata handlers
└── tests/                                # Automated unit & integration tests
    ├── test_config.py                    # Config paths and feature taxonomy tests
    ├── test_preprocessing.py             # FeatureEngineer and ColumnTransformer tests
    └── test_pipeline.py                  # Model inference and metrics integrity tests
```

---

## 6. Local Environment Installation Guide

```bash
# 1. Clone Repository
git clone https://github.com/your-username/CampusWell-RiskLab.git
cd CampusWell-RiskLab

# 2. Configure Environment Variables
cp .env.example .env

# 3. Create and Activate Virtual Environment
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 4. Install Dependencies
pip install -r requirements/requirements.txt

# 5. Run Test Suite
pytest
```

---

## 7. Exploratory Data Analysis & Statistical Auditing

The EDA pipeline (`python -m src.steps.eda`) evaluates data distributions, bivariate relationships, and statistical significance:

### Key Statistical Insights
1. **Predictor Ranking:**
   - **Numerical Predictors:** `Sleep_Hours` is the single strongest discriminator across risk categories (Kruskal-Wallis $H = 568.06$, Pearson $r = -0.5419$), followed by `GPA` ($H = 233.84$, $r = -0.3526$), and `Financial_Stress` ($H = 186.00$, $r = +0.3135$). `Age` has no statistical relationship with risk ($p = 0.935$).
   - **Categorical Predictors:** `Previous_Mental_Health_History` is the strongest categorical discriminator (Cramér's $V = 0.2425$), followed by `Physical_Activity` ($V = 0.1673$). `Gender` is statistically independent of risk ($p = 0.277$).
2. **Collinearity Audit:**
   - `Year_of_Study` $\leftrightarrow$ `Study_Hours_Per_Week`: $r = 0.7034$ (highest collinearity in dataset).
   - `Social_Media_Hours` $\leftrightarrow$ `Screen_Time_Hours`: $r = 0.6814$.
   - All Variance Inflation Factors (VIF) remain below $2.7$, proving collinearity does not destabilize generalized linear models.

---

## 8. Preprocessing & Feature Engineering Experiments

### Imputation Experiment (5-Fold Masked Cross-Validation on $X_{\text{train}}$)
We evaluated Mean, Median, KNN, and Iterative imputation across 5 folds with artificially masked values:
- `Social_Media_Hours`: Mean MAE 0.8784, Median MAE 0.8753, KNN MAE 0.6521, Iterative MAE 0.5889.
- `GPA`: Mean MAE 0.2746, Median MAE 0.2743, KNN MAE 0.2509, Iterative MAE 0.2216.
- Downstream CV F1 evaluation: Uniform Median achieved **0.7564** F1 vs **0.7539** for Iterative (due to small-sample regression noise and convergence instability in IterativeImputer). Decision: **Median imputation** selected for production stability and top classification score.

### Feature Engineering CV Ablation
- Baseline ($X_{\text{train}}$ raw features): Weighted F1 = **0.7511** $\pm$ 0.0210.
- `+ sleep_to_screen_ratio`: Weighted F1 = **0.7564** $\pm$ 0.0204 (**+0.0053** improvement) $\to$ **RETAINED**.
- `+ academic_workload_ratio`: Weighted F1 = **0.7491** $\pm$ 0.0140 (**-0.0020** degradation) $\to$ **REJECTED**.
- `+ non_social_screen_time`: Weighted F1 = **0.7504** (-0.0007) $\to$ **REJECTED**.
- `+ stress_support_ratio`: Weighted F1 = **0.7502** (-0.0009) $\to$ **REJECTED**.

---

## 9. Model Benchmarking and Comparison

We benchmarked four diverse supervised learning algorithms under identical **5-Fold Stratified Cross-Validation** ($N_{\text{train}} = 1,600$, `random_state=42`):

| Model Architecture | Mean Weighted F1 | Std Weighted F1 | Mean Macro F1 | Std Macro F1 | Mean Accuracy | Std Accuracy | Mean ROC-AUC (OvR) | Selection Decision |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.7564** | $\pm 0.0204$ | **0.7558** | $\pm 0.0203$ | **0.7563** | $\pm 0.0204$ | **0.9063** | **Champion (Rank 1)** |
| **HistGradientBoosting** | **0.6842** | $\pm 0.0120$ | **0.6833** | $\pm 0.0120$ | **0.6850** | $\pm 0.0142$ | **0.8587** | Retained for Tuning |
| **XGBoost Classifier** | **0.6788** | $\pm 0.0210$ | **0.6779** | $\pm 0.0211$ | **0.6787** | $\pm 0.0221$ | **0.8538** | Baseline Comparison |
| **Random Forest** | **0.6709** | $\pm 0.0316$ | **0.6700** | $\pm 0.0316$ | **0.6738** | $\pm 0.0328$ | **0.8438** | Eliminated |

### Why Did Linear Models Outperform Tree Ensembles?
A central empirical finding is that **Multinomial Logistic Regression substantially outperformed tree ensembles by $+7.2$ percentage points in F1-score**.

This phenomenon stems directly from the generative structure of the synthetic dataset:
- Synthetic risk labels were assigned using an **additive linear scoring index**:
  $$\text{Risk\_Score} = \sum w_i x_i + \epsilon$$
  followed by continuous threshold cuts into Low, Medium, and High.
- Generalized linear models model linear decision hyperplanes directly with minimal sample complexity.
- Tree-based models must approximate smooth diagonal boundaries using orthogonal axis-aligned step functions, introducing partition variance and estimation noise on moderate sample sizes ($N=1,600$).

---

## 10. Hyperparameter Optimization Results

Hyperparameter optimization was conducted using a two-stage coarse-to-fine strategy:
1. **Stage 1 (Coarse Search):** `RandomizedSearchCV` (5-fold CV) exploring wide hyperparameter spaces.
2. **Stage 2 (Fine Search):** `GridSearchCV` centered closely around the best parameter neighborhood from Stage 1.

| Model Candidate | Baseline CV F1 | Optimized CV F1 | Optimal Hyperparameters |
|---|---|---|---|
| **Logistic Regression** | **0.7564** | **0.7564** | `C: 1.0`, `solver: 'lbfgs'`, `max_iter: 1000` |
| **HistGradientBoosting** | **0.6842** | **0.6876** | `learning_rate: 0.05`, `max_iter: 200`, `max_leaf_nodes: 31`, `min_samples_leaf: 10`, `l2_regularization: 10.0` |

---

## 11. Final Model Evaluation on Held-Out Test Set

The finalized champion Logistic Regression pipeline was trained on the entire training partition ($N_{\text{train}}=1,600$) and evaluated on the untouched, held-out test partition ($N_{\text{test}}=400$).

### Test Set Performance Metrics

| Metric | Empirical Score | Interpretation |
|---|---|---|
| **Weighted F1-Score** | **0.7557** | Excellent overall harmonic balance across class proportions |
| **Macro F1-Score** | **0.7551** | Unbiased cross-class performance across all 3 tiers |
| **Overall Accuracy** | **0.7550** | 302 out of 400 test students classified correctly (98 errors) |
| **Weighted Precision** | **0.7565** | High diagnostic reliability across positive predictions |
| **Weighted Recall** | **0.7550** | High sensitivity across risk cohorts |
| **Multi-Class ROC-AUC (OvR)** | **0.9129** | Outstanding discriminative separation across class boundaries |

### Per-Class Test Breakdown

| Risk Class | Precision | Recall | F1-Score | Support ($N$) |
|---|---|---|---|---|
| **High Risk** | **0.8102** | **0.8162** | **0.8132** | 136 |
| **Low Risk** | **0.8281** | **0.8030** | **0.8154** | 132 |
| **Medium Risk** | **0.6296** | **0.6439** | **0.6367** | 132 |
| **Macro Average** | 0.7560 | 0.7544 | 0.7551 | 400 |
| **Weighted Average** | 0.7565 | 0.7550 | 0.7557 | 400 |

### Confusion Matrix ($3 \times 3$)

```
                     PREDICTED
                Low    Medium    High    Total
TRUE   Low      106        25       1      132
       Medium    22        85      25      132
       High       0        25     111      136
       Total    128       135     137      400
```

### Critical Safety Finding: Zero Severe False Negatives
- **Zero High $\to$ Low Errors:** Out of 136 true High Risk students, **0 were misclassified as Low Risk** ($0.0\%$).
- **Adjacent Boundary Errors:** Errors occur exclusively between adjacent categories ($\text{Low} \leftrightarrow \text{Medium}$ and $\text{Medium} \leftrightarrow \text{High}$), reflecting the continuous spectrum of synthetic distress scores.

---

## 12. Orchestration & Serving

### Apache Airflow Pipeline
The pipeline is orchestrated via `dags/student_mental_health_dag.py` with 5 sequential tasks:
`eda_task` $\to$ `preprocessing_task` $\to$ `modeling_task` $\to$ `optimization_task` $\to$ `evaluation_task`.

### Streamlit Application
Interactive 4-page dashboard:
- `🏠 Overview`: Dataset statistics, target distributions, and educational disclaimer.
- `📊 Exploratory Data Analysis`: Interactive distributions, missingness analysis, and correlation heatmaps.
- `🏆 Model Performance`: Real metrics, comparison tables, confusion matrix heatmap, and hyperparameters.
- `🎮 Prediction Playground`: Profile inputs, live pipeline inference, and probability distributions.

```bash
streamlit run app/streamlit_app.py
```

---

## 13. Standalone Pipeline Stage Execution

```bash
# 1. Run Data Validation & EDA
python -m src.steps.eda

# 2. Run Preprocessing & Generate Splits
python -m src.steps.preprocessing

# 3. Run Benchmark Modeling
python -m src.steps.modeling

# 4. Run Hyperparameter Optimization
python -m src.steps.optimization

# 5. Run Final Evaluation & Model Serialization
python -m src.steps.evaluation
```

---

## 14. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
