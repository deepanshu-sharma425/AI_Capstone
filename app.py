import streamlit as st
import pandas as pd
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Vehicle Maintenance Prediction",
    layout="centered"
)

st.title(" Vehicle Maintenance Prediction System")
st.write(
    "This system predicts **maintenance risk** and **estimated time-to-failure** "
    "using **classical machine learning models** trained on cleaned vehicle telemetry data."
)

# ==============================
# LOAD TRAINED MODELS
# ==============================
@st.cache_resource
def load_models():
    logistic_model = joblib.load("logistic_model.pkl")
    decision_tree_model = joblib.load("decision_tree_model.pkl")
    regression_model = joblib.load("regression_model.pkl")
    return logistic_model, decision_tree_model, regression_model

logistic_model, decision_tree_model, regression_model = load_models()

# ==============================
# MODEL SELECTION
# ==============================
st.subheader(" Select Classification Model")

model_choice = st.radio(
    "Choose a model for maintenance risk prediction",
    ["Logistic Regression", "Decision Tree"]
)

clf = logistic_model if model_choice == "Logistic Regression" else decision_tree_model

# ==============================
# INPUT FORM
# ==============================
st.subheader(" Enter Vehicle Telemetry Details")

with st.form("vehicle_input_form"):

    engine_hours = st.number_input("Engine Hours", min_value=0.0, value=500.0)
    mileage_km = st.number_input("Mileage (km)", min_value=0.0, value=25000.0)
    avg_engine_rpm = st.number_input("Average Engine RPM", min_value=500.0, value=1500.0)
    engine_load_nm = st.number_input("Engine Load (Nm)", min_value=0.0, value=40.0)
    engine_temp_c = st.number_input("Engine Temperature (°C)", value=90.0)
    ambient_temp_c = st.number_input("Ambient Temperature (°C)", value=30.0)
    fault_code_count = st.number_input("Fault Code Count", min_value=0, value=1)

    vehicle_usage_type = st.selectbox(
        "Vehicle Usage Type",
        ["L", "M", "H"],
        help="L = Light, M = Medium, H = Heavy"
    )

    vehicle_model = st.selectbox(
        "Vehicle Model",
        ["M14860", "L47181", "H29425", "Other"]
    )

    submit = st.form_submit_button("🔮 Predict Maintenance")

# ==============================
# PREDICTION LOGIC
# ==============================
if submit:

    # Feature engineering (same as training)
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
    risk_pred = clf.predict(input_df)[0]
    risk_prob = clf.predict_proba(input_df)[0][1]
    days_to_failure = max(0, round(regression_model.predict(input_df)[0], 1))

    risk_label = "High Risk" if risk_pred == 1 else "Low Risk"

    # ==============================
    # DISPLAY RESULTS
    # ==============================
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Maintenance Risk", risk_label)
    col2.metric("Risk Probability", f"{risk_prob:.2f}")
    col3.metric("Estimated Days to Failure", f"{days_to_failure} days")

    if days_to_failure < 30:
        st.warning("⚠ Maintenance recommended soon")

    # ==============================
    # CONTRIBUTING FACTORS (EXPLANATION)
    # ==============================
    if model_choice == "Logistic Regression":
        st.subheader(" Contributing Factors (Feature Importance)")

        coef = clf.named_steps["model"].coef_[0]
        feature_names = clf.named_steps["preprocessor"].get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": coef
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df.head(8))

        st.info(
            "Positive importance values increase maintenance risk, "
            "while negative values reduce it."
        )

    else:
        st.info(
            "Decision Tree predictions are based on rule-based thresholds "
            "in engine temperature, load, and usage intensity."
        )

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption(
    "Milestone-1 | Classical Machine Learning | "
    "Logistic Regression & Decision Tree | No LLMs | No Agentic AI"
)