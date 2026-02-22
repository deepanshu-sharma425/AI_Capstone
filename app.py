import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Vehicle Maintenance Prediction", layout="centered")

st.title("🚗 Vehicle Maintenance Prediction System")
st.write("Classical ML-based predictive maintenance (Milestone-1)")

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
# MODEL SELECTION
# ==============================
model_choice = st.radio(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree"]
)

clf = logistic_model if model_choice == "Logistic Regression" else tree_model

# ==============================
# INPUT FORM
# ==============================
with st.form("input_form"):
    engine_hours = st.number_input("Engine Hours", 0.0, value=500.0)
    mileage_km = st.number_input("Mileage (km)", 0.0, value=25000.0)
    avg_engine_rpm = st.number_input("Average RPM", 500.0, value=1500.0)
    engine_load_nm = st.number_input("Engine Load (Nm)", 0.0, value=40.0)
    engine_temp_c = st.number_input("Engine Temperature (°C)", value=90.0)
    ambient_temp_c = st.number_input("Ambient Temperature (°C)", value=30.0)
    fault_code_count = st.number_input("Fault Code Count", 0, value=1)

    vehicle_usage_type = st.selectbox("Usage Type", ["L", "M", "H"])
    vehicle_model = st.selectbox("Vehicle Model", ["M14860", "L47181", "H29425", "Other"])

    submit = st.form_submit_button("🔮 Predict")

# ==============================
# PREDICTION
# ==============================
if submit:
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

    risk_pred = clf.predict(input_df)[0]
    risk_prob = clf.predict_proba(input_df)[0][1]
    days_to_failure = max(0, round(regression_model.predict(input_df)[0], 1))

    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Maintenance Risk", "High" if risk_pred else "Low")
    col2.metric("Risk Probability", f"{risk_prob:.2f}")
    col3.metric("Days to Failure", f"{days_to_failure} days")

    if days_to_failure < 30:
        st.warning("⚠ Maintenance recommended soon")

st.markdown("---")
st.caption("Milestone-1 | Classical ML | No LLMs | No Agentic AI")