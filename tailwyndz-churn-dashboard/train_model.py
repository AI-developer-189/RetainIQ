import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ============================================================
# 1. Load and prepare the model dataset (same logic as notebook)
# ============================================================

# Load prediction CSV
predictions = pd.read_csv("data/churn_predictions_export.csv")

# Load customer profile
profile_path = r"E:\RetainIQ\Assessment - 4 [Customer Churn Prediction for a Retail Loyalty Program]\FreshBasket_Loyalty_Churn_Dataset.xlsx"
churn_label_path = profile_path

profile = pd.read_excel(
    profile_path, sheet_name="fb Customer Profile", header=1
)
activity = pd.read_excel(
    profile_path, sheet_name="fb Monthly Activity", header=1
)
churn = pd.read_excel(
    profile_path, sheet_name="fb Churn Label", header=1
)

# Clean membership tier
profile["MEMBERSHIP_TIER_CLEAN"] = profile["MEMBERSHIP_TIER"].str.strip().str.title()

# Aggregate activity per customer (12-month observation window: Jan 2023 - Dec 2023)
activity_observed = activity[
    (activity["MONTH"] >= "2023-01-01") & (activity["MONTH"] < "2024-01-01")
].copy()

activity_observed = activity_observed.groupby(["CUSTOMER_ID", "MONTH"]).agg(
    {"TOTAL_SPEND": "sum", "TRANSACTIONS": "sum", "COUPONS_REDEEMED": "sum", "SUPPORT_TICKETS": "sum"}
).reset_index()

activity_agg = (
    activity_observed.groupby("CUSTOMER_ID")
    .agg(
        TOTAL_SPEND_12M=("TOTAL_SPEND", "sum"),
        AVG_MONTHLY_SPEND_12M=("TOTAL_SPEND", "mean"),
        TOTAL_TXNS_12M=("TRANSACTIONS", "sum"),
        TOTAL_COUPONS_12M=("COUPONS_REDEEMED", "sum"),
        TOTAL_SUPPORT_TICKETS_12M=("SUPPORT_TICKETS", "sum"),
        MONTHS_RECORDED=("MONTH", "nunique"),
    )
    .reset_index()
)

activity_agg["TXNS_PER_MONTH"] = activity_agg["TOTAL_TXNS_12M"] / activity_agg["MONTHS_RECORDED"]
activity_agg["COUPONS_PER_MONTH"] = activity_agg["TOTAL_COUPONS_12M"] / activity_agg["MONTHS_RECORDED"]
activity_agg["SUPPORT_TICKETS_PER_MONTH"] = activity_agg["TOTAL_SUPPORT_TICKETS_12M"] / activity_agg["MONTHS_RECORDED"]

# Drop raw totals, keep normalized features
cols_to_drop = ["TOTAL_SPEND_12M", "TOTAL_TXNS_12M", "TOTAL_COUPONS_12M", "TOTAL_SUPPORT_TICKETS_12M"]
activity_normalized = activity_agg.drop(columns=cols_to_drop, errors="ignore")

# Merge profile + activity + churn
df_model = profile.merge(activity_normalized, on="CUSTOMER_ID", how="inner")
df_model = df_model.merge(churn[["CUSTOMER_ID", "CHURNED"]], on="CUSTOMER_ID", how="inner")

print(f"Final model dataset shape: {df_model.shape}")

# ============================================================
# 2. Define features and target
# ============================================================

cols_to_drop = ["CUSTOMER_ID", "SIGNUP_DATE", "PREFERRED_STORE_ID", "CHURNED"]
feature_cols = [c for c in df_model.columns if c not in cols_to_drop]

X = df_model[feature_cols].copy()
y = df_model["CHURNED"].copy()

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# ============================================================
# 3. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train set: {len(X_train)} customers")
print(f"Test set: {len(X_test)} customers")

# ============================================================
# 4. Define preprocessing pipeline
# ============================================================

numeric_transformer = Pipeline(steps={"imputer": SimpleImputer(strategy="median"), "scaler": StandardScaler()})
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# ============================================================
# 5. Train models
# ============================================================

# Logistic Regression
lr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(random_state=42, max_iter=1000))])
lr_pipeline.fit(X_train, y_train)
lr_score = lr_pipeline.score(X_test, y_test)
lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
print(f"Logistic Regression - Accuracy: {lr_score:.4f}, ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")

# Random Forest
rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42, n_estimators=100))])
rf_pipeline.fit(X_train, y_train)
rf_score = rf_pipeline.score(X_test, y_test)
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
print(f"Random Forest - Accuracy: {rf_score:.4f}, ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")

# ============================================================
# 6. Select final model and save pipeline
# ============================================================

# Random Forest selected based on performance
final_pipeline = rf_pipeline

# Save the entire fitted pipeline using joblib
joblib.dump(final_pipeline, "models/churn_pipeline.pkl")
print("\n✅ Trained pipeline saved to models/churn_pipeline.pkl")

# ============================================================
# 7. Feature importance (Random Forest only)
# ============================================================

rf_model = final_pipeline.named_steps["classifier"]
preprocessor_final = final_pipeline.named_steps["preprocessor"]

num_features = numeric_cols
cat_features = preprocessor_final.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_cols)
all_feature_names = list(num_features) + list(cat_features)

feature_importances = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": rf_model.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\nTop feature importances (Random Forest):")
print(feature_importances.head(10))

# Save feature importances
feature_importances.to_csv("models/feature_importances.csv", index=False)
print("Feature importances saved to models/feature_importances.csv")