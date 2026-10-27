import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_cols = ['CHURN_PROBABILITY']
categorical_cols = ['RISK_LEVEL']

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_cols), ('cat', categorical_transformer, categorical_cols)],
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, n_estimators=10))])
joblib.dump(pipeline, r"E:\RetainIQ\tailwyndz-churn-dashboard\models\churn_pipeline.pkl")
print("Full pipeline saved successfully")