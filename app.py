# app.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Vehicle Maintenance Prediction",
    layout="centered"
)

st.title("Vehicle Maintenance Prediction System")
st.write(
    "Enter vehicle telemetry and usage parameters to predict "
    "maintenance risk and estimated time-to-failure."
)

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    return (
        joblib.load("logistic_model.pkl"),
        joblib.load("decision_tree_model.pkl"),
        joblib.load("regression_model.pkl")
    )

logistic_model, tree_model, regression_model = load_models()

# ==============================
# USER INPUT FORM
# ==============================
st.subheader(" Enter Vehicle Parameters")

with st.form("vehicle_form"):
    engine_hours = st.number_input("Engine Hours", min_value=0.0)
    avg_engine_rpm = st.number_input("Average Engine RPM", min_value=0.0)
    engine_load_nm = st.number_input("Engine Load (Nm)", min_value=0.0)
    engine_temp_c = st.number_input("Engine Temperature (°C)")
    ambient_temp_c = st.number_input("Ambient Temperature (°C)")
    fault_code_count = st.number_input("Fault Code Count", min_value=0)
    mileage_km = st.number_input("Mileage (km)", min_value=0.0)
    usage_intensity = st.number_input("Usage Intensity", min_value=0.0)

    vehicle_usage_type = st.selectbox(
        "Vehicle Usage Type",
        ["L", "M", "H"]
    )

    vehicle_model = st.selectbox(
        "Vehicle Model",
        ["M14860", "L47181", "H29424"]
    )

    submit = st.form_submit_button("🔮 Predict Maintenance Risk")

# ==============================
# PREDICTION
# ==============================
if submit:
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

    risk = logistic_model.predict(input_df)[0]
    probability = logistic_model.predict_proba(input_df)[0][1]
    days_to_failure = max(0, regression_model.predict(input_df)[0])


    st.subheader(" Prediction Result")

    st.metric(
        "Maintenance Risk",
        "High Risk" if risk == 1 else "Low Risk"
    )

    st.metric(
        "Risk Probability",
        f"{probability:.3f}"
    )

    st.metric(
        "Predicted Days to Failure",
        f"{days_to_failure:.1f} days"
    )


