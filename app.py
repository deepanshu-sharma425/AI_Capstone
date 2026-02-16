# app.py

import streamlit as st
import pandas as pd
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Vehicle Maintenance Prediction",
    layout="wide"
)

st.title("🚗 Vehicle Maintenance Prediction System")
st.write(
    "Predict maintenance risk and estimate time-to-failure using "
    "vehicle telemetry and usage data (Classical ML)."
)

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    logistic_model = joblib.load("logistic_model.pkl")
    tree_model = joblib.load("decision_tree_model.pkl")
    regression_model = joblib.load("regression_model.pkl")
    return logistic_model, tree_model, regression_model

logistic_model, tree_model, regression_model = load_models()

# ==============================
# FILE UPLOAD
# ==============================
st.subheader("📤 Upload Vehicle Telemetry CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV file (same format as cleaned_vehicle_data.csv)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(data.head())

    # ==============================
    # FEATURE SELECTION
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

    if not all(col in data.columns for col in FEATURE_COLUMNS):
        st.error("Uploaded CSV does not contain required columns.")
    else:
        X = data[FEATURE_COLUMNS]

        # ==============================
        # PREDICTION BUTTON
        # ==============================
        if st.button("🔮 Predict Maintenance Risk"):
            # Classification predictions
            data["maintenance_risk_lr"] = logistic_model.predict(X)
            data["maintenance_risk_dt"] = tree_model.predict(X)

            # Probability from Logistic Regression
            data["risk_probability"] = logistic_model.predict_proba(X)[:, 1]

            # Regression prediction
            data["predicted_days_to_failure"] = regression_model.predict(X)

            # Risk label
            data["risk_label"] = data["maintenance_risk_lr"].map(
                {0: "Low Risk", 1: "High Risk"}
            )

            st.subheader("📊 Prediction Results")
            st.dataframe(
                data[
                    [
                        "risk_label",
                        "risk_probability",
                        "predicted_days_to_failure"
                    ]
                ].head(20)
            )

            # ==============================
            # SUMMARY METRICS
            # ==============================
            high_risk_count = (data["maintenance_risk_lr"] == 1).sum()
            total = len(data)

            st.subheader("📈 Fleet Summary")
            col1, col2 = st.columns(2)
            col1.metric("Total Vehicles", total)
            col2.metric("High-Risk Vehicles", high_risk_count)

            # ==============================
            # DOWNLOAD RESULTS
            # ==============================
            st.subheader("⬇️ Download Predictions")

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Prediction CSV",
                data=csv,
                file_name="maintenance_predictions.csv",
                mime="text/csv"
            )

else:
    st.info("Please upload a CSV file to begin.")



    jdjdjd
    datadj
    datadjd
    datadjd
    datadjd

    datadjd
ddddDD
datadjdD
datadjdD
datadjdD
datadjdD
datadjdD

datadjdD
D
    datadjd