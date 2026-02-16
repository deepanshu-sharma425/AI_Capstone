# src/train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error

# ==============================
# 1. LOAD CLEANED DATA
# ==============================
df = pd.read_csv("data/cleaned_vehicle_data.csv")

print("Dataset loaded:", df.shape)

# ==============================
# 2. DEFINE FEATURES & TARGETS
# ==============================

FEATURE_COLUMNS = [
    'engine_hours',
    'avg_engine_rpm',
    'engine_load_nm',
    'engine_temp_c',
    'ambient_temp_c',
    'fault_code_count',
    'mileage_km',
    'usage_intensity',
    'vehicle_usage_type',
    'vehicle_model'
]

X = df[FEATURE_COLUMNS]
y_class = df['maintenance_required']
y_reg = df['days_to_failure']

# ==============================
# 3. TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# ==============================
# 4. PREPROCESSING PIPELINE
# ==============================

numeric_features = [
    'engine_hours',
    'avg_engine_rpm',
    'engine_load_nm',
    'engine_temp_c',
    'ambient_temp_c',
    'fault_code_count',
    'mileage_km',
    'usage_intensity'
]

categorical_features = [
    'vehicle_usage_type',
    'vehicle_model'
]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ==============================
# 5. LOGISTIC REGRESSION MODEL
# ==============================
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

logistic_pipeline.fit(X_train, y_train)
logistic_preds = logistic_pipeline.predict(X_test)

logistic_acc = accuracy_score(y_test, logistic_preds)

print("\n--- Logistic Regression ---")
print("Accuracy:", round(logistic_acc, 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_preds))
print(classification_report(y_test, logistic_preds))

# ==============================
# 6. DECISION TREE MODEL
# ==============================
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeClassifier(max_depth=6, random_state=42))
])

tree_pipeline.fit(X_train, y_train)
tree_preds = tree_pipeline.predict(X_test)

tree_acc = accuracy_score(y_test, tree_preds)

print("\n--- Decision Tree ---")
print("Accuracy:", round(tree_acc, 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, tree_preds))
print(classification_report(y_test, tree_preds))

# ==============================
# 7. REGRESSION (RMSE)
# ==============================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

reg_pipeline.fit(Xr_train, yr_train)
reg_preds = reg_pipeline.predict(Xr_test)

mse = mean_squared_error(yr_test, reg_preds)
rmse = mse ** 0.5

print("\n--- Regression (Days to Failure) ---")
print("RMSE:", round(rmse, 2))


# ==============================
# 8. SAVE MODELS
# ==============================
joblib.dump(logistic_pipeline, "logistic_model.pkl")
joblib.dump(tree_pipeline, "decision_tree_model.pkl")
joblib.dump(reg_pipeline, "regression_model.pkl")

print("\nModels saved successfully:")
print(" - logistic_model.pkl")
print(" - decision_tree_model.pkl")
print(" - regression_model.pkl")
