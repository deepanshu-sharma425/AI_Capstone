import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Vehicle Maintenance Prediction",
    layout="centered"
)

st.title("🚗 Vehicle Maintenance Prediction System")
st.write(
    "Enter vehicle usage and sensor parameters to predict **maintenance risk** "
    "and **estimated time-to-failure** using classical machine learning."
)

# ==============================
# LOAD TRAINED MODELS
# ==============================
@st.cache_resource
def load_models():
    logistic_model = joblib.load("logistic_model.pkl")
    tree_model = joblib.load("decision_tree_model.pkl")
    regression_model = joblib.load("regression_model.pkl")
    return logistic_model, tree_model, regression_model

logistic_model, tree_model, regression_model = load_models()

# ==============================
# USER INPUT FORM
# ==============================
st.subheader("🛠 Enter Vehicle Parameters")

with st.form("vehicle_input_form"):

    engine_hours = st.number_input(
        "Total Engine Hours",
        min_value=0.0,
        value=500.0
    )

    mileage_km = st.number_input(
        "Total Distance Driven (km)",
        min_value=0.0,
        value=25000.0
    )

    avg_engine_rpm = st.number_input(
        "Average Engine RPM",
        min_value=500.0,
        value=1500.0
    )

    engine_load_nm = st.number_input(
        "Average Engine Load (Nm)",
        min_value=0.0,
        value=40.0
    )

    engine_temp_c = st.number_input(
        "Engine Temperature (°C)",
        value=90.0
    )

    ambient_temp_c = st.number_input(
        "Ambient Temperature (°C)",
        value=30.0
    )

    fault_code_count = st.number_input(
        "Fault Code Count",
        min_value=0,
        value=0
    )

    vehicle_usage_type = st.selectbox(
        "Vehicle Usage Type",
        options=["L", "M", "H"],
        help="L = Light, M = Medium, H = Heavy usage"
    )

    vehicle_model = st.selectbox(
        "Vehicle Model",
        options=["M14860", "L47181", "H29425", "Other"]
    )

    submit = st.form_submit_button("🔮 Predict Maintenance Risk")

# ==============================
# FEATURE ENGINEERING + PREDICTION
# ==============================
if submit:

    # Feature engineering (same logic as training)
    usage_intensity = engine_load_nm / (engine_hours + 1)

    input_df = pd.DataFrame([{
        "engine_hours": engine_hours,
        "avg_engine_rpm": avg_engine_rpm,
        "engine_load_nm": engine_load_nm,
        "engine_temp_c": engine_temp_c,
        "ambient_temp_c": ambient_temp_c,
        "fault_code_count": fault_code_count,
        "mileage_km": mileage_km,
        "usage_intensity": usage_intensity,
        "vehicle_usage_type": vehicle_usage_type,
        "vehicle_model": vehicle_model
    }])

    # Predictions
    risk_pred = logistic_model.predict(input_df)[0]
    risk_prob = logistic_model.predict_proba(input_df)[0][1]
    days_to_failure = regression_model.predict(input_df)[0]

    # Safety: no negative days
    days_to_failure = max(0, round(days_to_failure, 1))

    # Output formatting
    risk_label = "High Risk" if risk_pred == 1 else "Low Risk"

    # ==============================
    # DISPLAY RESULTS
    # ==============================
    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("Maintenance Risk", risk_label)
    col2.metric("Risk Probability", f"{risk_prob:.2f}")
    col3.metric("Estimated Days to Failure", f"{days_to_failure} days")

    if days_to_failure == 0:
        st.warning("⚠ Immediate maintenance recommended.")

    st.success("Prediction generated successfully.")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption(
    "Milestone-1: Classical ML-based Vehicle Maintenance Prediction "
    "(Logistic Regression & Decision Trees)"
)
