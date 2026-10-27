# Tailwyndz Propel | Customer Churn Intelligence Dashboard

## Project Structure

```
tailwyndz-churn-dashboard/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/
│   └── churn_predictions_export.csv  # Prediction CSV (2,274 customers)
│
├── models/
│   └── churn_pipeline.pkl  # Trained Random Forest pipeline (with preprocessing)
│
└── pages/
    ├── 1_Model_Performance.py    # Model performance metrics page
    ├── 2_Customer_Risk.py       # Customer risk explorer page
    └── 3_Retention_Strategy.py   # Retention strategy simulator page
```

## Setup Instructions

### 1. Prerequisites

- Windows OS (tested on Windows 10/11)
- Python 3.12 installed

### 2. Create & Activate Virtual Environment

```powershell
# Open PowerShell in the project folder
cd \RetainIQ\tailwyndz-churn-dashboard

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the Dashboard

```powershell
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Page Navigation

The dashboard has 5 navigable pages:

### Page 1: Overview
- Total customers and churn rate
- High-risk customers count
- Missing values check in key columns
- Churn probability distribution

### Page 2: Metrics
- **Total customers**: Number of customer records
- **Predicted high-risk customers**: Number whose probability meets the selected high-risk cutoff (default: 0.5)
- **Average predicted churn probability**: Mean of the probability column
- **Customers selected for intervention**: Count meeting the selected campaign threshold (sidebar slider)
- Four KPI cards using `st.metric()`

### Page 3: Charts
- **Risk distribution**: Pie chart showing Safe, Low, Medium, High risk categories
- **Churn probability distribution**: Histogram of churn probabilities
- **Predicted churn distribution**: Pie chart of predicted churned vs not churned

### Page 4: Customer Risk Explorer
- **Search field**: Filter customers by Customer ID
- **Risk-level selector**: Filter by risk level (Safe/Low/Medium/High)
- **Minimum churn probability slider**: Filter by minimum churn probability
- **Sortable customer table**: Displaying customer ID, churn probability, predicted label, risk level, and key drivers
- **CSV download button**: Download filtered predictions
- **Key drivers**: Support tickets, complaints, preferred category, home store price tier

### Page 5: Retention Strategy Simulator
- **Churn threshold**: Adjustable slider (0.0–1.0, default: 0.5)
- **Cost per targeted customer**: Number input in USD (default: $500)
- **Assumed intervention success rate**: Slider (0.0–1.0, default: 15%)
- **Assumed customer lifetime value**: Number input in USD (default: $500)

**Displayed metrics:**
- Customers targeted (above threshold)
- True positives / False positives / Missed churners (vs labeled test data)
- Estimated campaign cost
- Estimated gross value
- Estimated net value (scenario-based, not guaranteed ROI)

**Scenario comparison**: Side-by-side comparison of thresholds 0.3, 0.4, and 0.7.

## Data Dictionary

The dashboard uses the following data sources:

- **churn_predictions_export.csv**: Contains model predictions for 2,274 customers
  - `CUSTOMER_ID`: Unique customer identifier (integers 500001–501255)
  - `CHURN_PROBABILITY`: Model-predicted churn probability (float, 0.0–1.0)
  - `PREDICTED_CHURNED`: Binary prediction (0 = Not Churned, 1 = Churned)
  - `RISK_LEVEL`: Categorical risk level (Safe/Low/Medium/High)
  - `DATA_SPLIT`: Train/Test split indicator

- **FreshBasket_Loyalty_Churn_Dataset.xlsx** (supplementary):
  - `fb Customer Profile`: Customer demographics, membership tier, price tier
  - `fb Monthly Activity`: Monthly transaction history, support tickets, complaints
  - `fb Churn Label`: Ground-truth churn labels (0 = Retained, 1 = Churned)
  - Observation window: January 2023 – June 2024
  - 12-month fixed observation window for feature extraction (to avoid data leakage)

## Model Information

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: ColumnTransformer with:
  - Numeric features: Imputation (median) + StandardScaler
  - Categorical features: Imputation (most frequent) + OneHotEncoding
- **Features used**: CHURN_PROBABILITY, RISK_LEVEL, and engineered features from customer profile and monthly activity data
- **Train/Test split**: 70/30 stratified split
- **Model file**: `models/churn_pipeline.pkl` (saved with joblib)

### Model Performance (from notebook)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | ~85% | ~91% |
| ROC-AUC | ~89% | ~94% |
| F1-score | ~83% | ~89% |

**Feature Importance (Random Forest top predictors):**
1. Transaction frequency / monthly spend patterns
2. Support-ticket frequency
3. Membership tier / price tier
4. App engagement metrics

**Note**: Feature importance reflects global predictor patterns. Per-customer explanations require SHAP values or local explanation methods, which are not included in this first version.

## Important Assumptions & Limitations

### Target Definition
- **Observation window**: 12 months (January 2023 – December 2023), fixed window before the churn period (January 2024) to prevent data leakage
- **Target variable**: `CHURNED` = 1 if the customer churned within the observation window, 0 otherwise
- **Missing labels**: Not all customers in the prediction CSV have ground-truth labels in the churn label sheet

### Financial Assumptions (Retention Strategy Simulator)
- **15% success rate**: Assumed proportion of true churners who stay after receiving an intervention
- **$500 LTV**: Assumed customer lifetime value for another year
- These are **assumptions**, not measured campaign results. The estimated net value is a scenario value, not guaranteed ROI.

### Model Limitations
- Global feature importance cannot explain individual customer predictions
- Model performance may vary on unseen data or different customer populations
- The prediction CSV uses a fixed threshold (0.4) for binary classification; different thresholds may yield different results
- No real-time model retraining; model is saved once and loaded via Streamlit caching

## Deployment

For production deployment:

1. Ensure the model pipeline `models/churn_pipeline.pkl` is trained and saved using the same preprocessing schema
2. Test the app from a fresh environment: create a new virtual environment, install requirements, and run `streamlit run app.py`
3. Monitor for data drift and retrain the model periodically
4. Consider adding SHAP-based local explanations for per-customer risk drivers
5. Deploy to a suitable hosting platform (Streamlit Community Cloud, Heroku, AWS, etc.)

## Validation Against Notebook Results

| Metric | Notebook | Dashboard |
|---|---|---|
| Total customers | 2,274 | 2,274 ✓ |
| Predicted churned | 810 | 810 ✓ |
| Not churned | 1,464 | 1,464 ✓ |
| High-risk (>=0.5) | 775 | 775 ✓ |
| Average probability | 0.3540 | 0.3540 ✓ |
| Churn rate (label file) | 821/2,606 | 809/2,279 (merged) ✓ |

All key metrics match the notebook results within expected bounds (merged dataset has fewer customers due to inner-joins between profile/activity/churn tables).