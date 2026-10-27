# RetainIQ — Tailwyndz Propel | Customer Churn Intelligence

> **A machine-learning powered customer churn prediction and retention strategy dashboard for the FreshBasket Loyalty Program.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Data Sources](#4-data-sources)
5. [Components](#5-components)
6. [Dashboard Pages & Working](#6-dashboard-pages--working)
7. [ML Model Pipeline](#7-ml-model-pipeline)
8. [Setup & Running Locally](#8-setup--running-locally)
9. [Key Metrics & Outputs](#9-key-metrics--outputs)

---

## 1. Project Overview

**RetainIQ** is an end-to-end customer churn prediction system built for **FreshBasket**, a retail loyalty program. The project covers:

- **Exploratory Data Analysis (EDA)** on customer profile, monthly activity, and churn label data.
- **Machine Learning model training** (Logistic Regression & Random Forest) with scikit-learn pipelines.
- **Churn prediction export** — a CSV of per-customer churn probabilities and risk levels.
- **Interactive Streamlit dashboard** (`Tailwyndz Propel`) with 5 views: Overview, Metrics, Charts, Customer Risk Explorer, and Retention Strategy Simulator.

The goal is to help business analysts and retention teams identify at-risk customers, understand their risk drivers, and simulate the financial impact of different intervention strategies.

---

## 2. Architecture

```
+----------------------------------------------------------------+
|                        Data Layer                              |
|  FreshBasket_Loyalty_Churn_Dataset.xlsx                        |
|   +-- fb Customer Profile   (demographics, membership tier)    |
|   +-- fb Monthly Activity   (spend, transactions, coupons)     |
|   +-- fb Churn Label        (CHURNED: 0/1)                    |
+------------------------+---------------------------------------+
                         |
                         v
+----------------------------------------------------------------+
|                   EDA & Feature Engineering                    |
|  FreshBasket_Churn_EDA.ipynb                                   |
|   - Exploratory analysis, visualizations                       |
|   - Feature derivation (12-month aggregates)                   |
|   - Churn probability scoring -> churn_predictions_export.csv  |
+------------------------+---------------------------------------+
                         |
                         v
+----------------------------------------------------------------+
|                     Model Training                             |
|  train_model.py                                                |
|   - Loads profile + activity + churn labels                    |
|   - Trains Logistic Regression & Random Forest                 |
|   - Selects best model -> churn_pipeline.pkl                   |
+------------------------+---------------------------------------+
                         |
                         v
+----------------------------------------------------------------+
|                  Streamlit Dashboard                           |
|  app.py + pages/                                               |
|   +-- Overview        -- data preview & distribution           |
|   +-- Metrics         -- KPI cards & live model status         |
|   +-- Charts          -- interactive Plotly visualizations     |
|   +-- Customer Risk   -- searchable, filterable risk table     |
|   +-- Retention Strategy -- ROI simulation & threshold compare |
+----------------------------------------------------------------+
```

---

## 3. Project Structure

```
e:/RetainIQ/
|
+-- README.md                                    <- This file
+-- requirements.txt                             <- Python dependencies
+-- extract_notebook.py                          <- Helper to extract notebook
+-- churn_predictions_export.csv                 <- Root-level predictions copy
+-- FreshBasket_Churn_EDA (2).ipynb              <- Primary EDA notebook
|
+-- Assessment - 4 [...]/
|   +-- Customer_Churn_Prediction_Assessment.docx
|   +-- FreshBasket_Loyalty_Churn_Dataset.xlsx   <- Master data file
|
+-- tailwyndz-churn-dashboard/                   <- Main application
    +-- app.py                                   <- Streamlit entry point
    +-- train_model.py                           <- Full model training script
    +-- save_model.py                            <- Pipeline stub / quick save
    +-- requirements.txt
    +-- data/
    |   +-- churn_predictions_export.csv         <- Model output (predictions)
    +-- models/
    |   +-- churn_pipeline.pkl                   <- Serialized ML pipeline
    +-- pages/
        +-- 2_Customer_Risk.py                   <- Customer Risk Explorer page
        +-- 3_Retention_Strategy.py              <- Retention Simulator page
```

---

## 4. Data Sources

All data originates from **`FreshBasket_Loyalty_Churn_Dataset.xlsx`**, which has three sheets:

| Sheet | Key Columns | Description |
|---|---|---|
| `fb Customer Profile` | `CUSTOMER_ID`, `MEMBERSHIP_TIER`, `PREFERRED_CATEGORY`, `HOME_STORE_PRICE_TIER`, `SIGNUP_DATE` | Static demographic and preference data per customer |
| `fb Monthly Activity` | `CUSTOMER_ID`, `MONTH`, `TOTAL_SPEND`, `TRANSACTIONS`, `COUPONS_REDEEMED`, `SUPPORT_TICKETS` | Monthly transactional behaviour (Jan 2023 – Dec 2023) |
| `fb Churn Label` | `CUSTOMER_ID`, `CHURNED` (0/1) | Ground-truth churn outcome |

The prediction CSV (`churn_predictions_export.csv`) adds:

| Column | Description |
|---|---|
| `CHURN_PROBABILITY` | Model-predicted probability of churn (0.0 – 1.0) |
| `PREDICTED_CHURNED` | Binary prediction (1 = churned, 0 = retained) |
| `RISK_LEVEL` | Categorical tier derived from churn probability |

---

## 5. Components

### 5.1 Jupyter Notebook — EDA & Prediction Export

**File:** `FreshBasket_Churn_EDA (2).ipynb`

This is the analytical starting point. It performs:

- **Data loading & cleaning** from all three Excel sheets.
- **Exploratory Data Analysis** — churn rate by membership tier, spend distributions, activity patterns.
- **Feature engineering** — deriving 12-month aggregates (total spend, avg monthly spend, transactions per month, coupons per month, support tickets per month).
- **Churn probability scoring** — generating per-customer churn probabilities.
- **Export** — outputs `churn_predictions_export.csv` with `CUSTOMER_ID`, `CHURN_PROBABILITY`, `PREDICTED_CHURNED`, and `RISK_LEVEL`.

---

### 5.2 Model Training — `train_model.py`

**File:** `tailwyndz-churn-dashboard/train_model.py`

This standalone script trains the full scikit-learn model pipeline.

#### Steps:

1. **Data Loading** — Reads the prediction CSV, customer profile, monthly activity, and churn label from the Excel file.
2. **Feature Engineering** — Aggregates 12 months of activity per customer (Jan–Dec 2023), computing:
   - `AVG_MONTHLY_SPEND_12M`
   - `TXNS_PER_MONTH`
   - `COUPONS_PER_MONTH`
   - `SUPPORT_TICKETS_PER_MONTH`
   - `MONTHS_RECORDED`
3. **Train/Test Split** — 70/30 stratified split (`random_state=42`).
4. **Preprocessing Pipeline** — Uses `ColumnTransformer`:
   - **Numeric columns** → `SimpleImputer(median)` + `StandardScaler`
   - **Categorical columns** → `SimpleImputer(most_frequent)` + `OneHotEncoder`
5. **Model Training** — Trains two models:
   - `LogisticRegression` (max_iter=1000)
   - `RandomForestClassifier` (n_estimators=100)
6. **Evaluation** — Reports accuracy and ROC-AUC for both.
7. **Model Selection** — Random Forest is selected as the final model.
8. **Serialization** — Saves the fitted pipeline to `models/churn_pipeline.pkl` using `joblib`.
9. **Feature Importance** — Extracts and saves top feature importances from the Random Forest to `models/feature_importances.csv`.

---

### 5.3 Pipeline Stub — `save_model.py`

**File:** `tailwyndz-churn-dashboard/save_model.py`

A minimal utility script that creates and saves an **untrained** Random Forest pipeline skeleton. Used for:

- Bootstrapping the `models/churn_pipeline.pkl` file structure.
- Quick testing of the dashboard's model-loading path without running the full training script.

> **Note:** This pipeline is **not trained** on real data. Run `train_model.py` to produce a production-ready model.

---

### 5.4 Streamlit Dashboard — `app.py`

**File:** `tailwyndz-churn-dashboard/app.py`

The primary Streamlit entry point. It hosts five views accessible from the sidebar radio selector.

#### Data Loading (`load_data`)
- Reads `data/churn_predictions_export.csv` as the base.
- Optionally enriches data by merging Excel sheets (Customer Profile, Monthly Activity, Churn Label).
- Gracefully falls back to zero-filled columns if the Excel file is unavailable.
- Derives `SUPPORT_TICKETS` and `COMPLAINT_FLAG` from activity/profile data.

#### Model Loading (`load_model`)
- Loads `models/churn_pipeline.pkl` using `joblib`.
- Returns `None` if the file does not exist — the dashboard degrades gracefully, displaying a warning.

---

### 5.5 Customer Risk Explorer — `pages/2_Customer_Risk.py`

**File:** `tailwyndz-churn-dashboard/pages/2_Customer_Risk.py`

A dedicated Streamlit page for drilling into individual customer risk profiles.

#### Features:
- **Search by Customer ID** — live text filter.
- **Risk Level Filter** — dropdown to select Safe / Low / Medium / High.
- **Churn Probability Slider** — show only customers above a minimum probability.
- **Sortable Table** — sort by Customer ID, Churn Probability, Predicted Churned, or Risk Level.
- **Key Drivers Summary** — metrics cards showing Avg Churn Probability, Total Support Tickets, Total Complaints.
- **Custom CSV Upload** — upload your own CSV to update churn probabilities.
- **Visualizations:**
  - Donut pie chart — Churn Distribution (Active vs Churned).
  - Histogram — Churn Probability Distribution.
  - Bar chart — Average Churn Probability by Risk Level.
- **CSV Download** — download the filtered table as `filtered_customer_risk.csv`.

---

### 5.6 Retention Strategy Simulator — `pages/3_Retention_Strategy.py`

**File:** `tailwyndz-churn-dashboard/pages/3_Retention_Strategy.py`

A financial modelling tool to simulate the ROI of different retention intervention strategies.

#### Configurable Assumptions (sidebar):

| Parameter | Default | Description |
|---|---|---|
| Churn threshold | 0.50 | Customers above this probability are targeted |
| Cost per customer | $500 | Cost to execute one retention intervention |
| Intervention success rate | 15% | Assumed % of interventions that successfully prevent churn |
| Customer lifetime value (LTV) | $500 | Assumed revenue retained per saved customer |

#### Computed Metrics:

| Metric | Formula |
|---|---|
| Campaign Cost | `n_targeted × cost_per_customer` |
| Gross Value | `true_positives × success_rate × LTV` |
| Net Value | `Gross Value − Campaign Cost` |
| True Positives | Predicted churners who actually churned |
| False Positives | Predicted churners who did not churn |
| Missed Churners | Actual churners below the threshold |

#### Scenario Comparison Table:
Automatically compares three fixed thresholds — **0.3, 0.4, 0.7** — side by side so analysts can evaluate the coverage vs. cost trade-off.

---

## 6. Dashboard Pages & Working

| Page | Access | Description |
|---|---|---|
| **Overview** | Sidebar → Overview | Data preview (top 10 rows), missing value check, churn count summary |
| **Metrics** | Sidebar → Metrics | Four KPI cards: total customers, high-risk count, avg churn probability, intervention count. Adjustable threshold slider |
| **Charts** | Sidebar → Charts | Three Plotly charts: risk level pie, churn probability histogram, predicted churn pie |
| **Customer Risk** | Sidebar → Customer Risk / Page 2 | Full-featured risk explorer with search, filter, sort, visualizations, and CSV export |
| **Retention Strategy** | Sidebar → Retention Strategy / Page 3 | Financial ROI simulator with configurable assumptions and scenario comparison table |

---

## 7. ML Model Pipeline

```
Input Features
     |
     +-- Numeric: AVG_MONTHLY_SPEND_12M, TXNS_PER_MONTH,
     |            COUPONS_PER_MONTH, SUPPORT_TICKETS_PER_MONTH,
     |            MONTHS_RECORDED, TENURE, ...
     |
     +-- Categorical: MEMBERSHIP_TIER, PREFERRED_CATEGORY,
                      HOME_STORE_PRICE_TIER, ...
     |
     v
+----------------------------------+
|        ColumnTransformer         |
|  Numeric  -->  Impute + Scale    |
|  Categorical --> Impute + OHE    |
+----------------------------------+
     |
     v
+----------------------------------+
|   RandomForestClassifier         |
|   n_estimators=100, seed=42      |
+----------------------------------+
     |
     v
  Output: churn probability (0-1) + binary label
```

The pipeline is serialized with `joblib` and loaded at dashboard startup. If the model file is absent, the dashboard falls back to the pre-computed `churn_predictions_export.csv`.

---

## 8. Setup & Running Locally

### Prerequisites
- Python 3.9+
- `pip`

### Installation

```bash
# Navigate to the application directory
cd "e:/RetainIQ/tailwyndz-churn-dashboard"

# Create a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Prepare Data

1. Copy `FreshBasket_Loyalty_Churn_Dataset.xlsx` into `tailwyndz-churn-dashboard/data/`.
2. Ensure `data/churn_predictions_export.csv` exists (generated by the notebook or provided separately).

### Train the Model *(optional — enables live predictions)*

```bash
python train_model.py
```

This produces `models/churn_pipeline.pkl` and `models/feature_importances.csv`.

### Run the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 9. Key Metrics & Outputs

| Output | Location | Description |
|---|---|---|
| `churn_predictions_export.csv` | `data/` | Per-customer churn probability, prediction, and risk level |
| `churn_pipeline.pkl` | `models/` | Serialized scikit-learn Random Forest pipeline |
| `feature_importances.csv` | `models/` | Top feature importances from the trained Random Forest |
| Dashboard | `http://localhost:8501` | Interactive multi-page Streamlit application |

---

> **Disclaimer:** Financial outputs in the Retention Strategy Simulator (campaign cost, gross/net value) are estimated scenario projections based on user-supplied assumptions. They are not guaranteed ROI figures.
