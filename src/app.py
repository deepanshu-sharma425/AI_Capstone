import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="VehicleCare360",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# Dark Theme CSS
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
}

.main .block-container {
    padding: 3rem 4rem;
    max-width: 1200px;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 3rem;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 600;
    color: #f9fafb;
    margin: 0;
}

.subtitle {
    font-size: 0.95rem;
    color: #9ca3af;
    margin-top: 0.5rem;
}

/* Section titles */
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #93c5fd;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Inputs */
input, select {
    background-color: #020617 !important;
    color: #f9fafb !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 500;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    font-size: 0.95rem;
    width: 100%;
}

/* Result cards */
.result {
    border-radius: 10px;
    padding: 2rem;
    margin-top: 2rem;
}

.risk-low {
    background-color: #022c22;
    border-left: 5px solid #10b981;
}

.risk-high {
    background-color: #3f1d1d;
    border-left: 5px solid #ef4444;
}

.result-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: #f9fafb;
}

.result-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #d1d5db;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.8rem;
    margin-top: 4rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("""
<div class="header">
    <h1 class="main-title">VehicleCare360</h1>
    <p class="subtitle">AI-Powered Predictive Vehicle Maintenance System</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Models
# -------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load("models/scaler.pkl")
        lr_model = joblib.load("models/logistic_model.pkl")
        dt_model = joblib.load("models/decision_tree_model.pkl")
        return scaler, lr_model, dt_model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

scaler, lr_model, dt_model = load_assets()

if scaler is None:
    st.stop()

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
model_choice = st.selectbox(
    "Select Prediction Model",
    ["Logistic Regression", "Decision Tree"]
)

# -------------------------------------------------
# Input Sections
# -------------------------------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<p class="section-title">Temperature</p>', unsafe_allow_html=True)
    air_temp = st.number_input("Air Temperature (K)", 250.0, 400.0, 300.0)
    process_temp = st.number_input("Process Temperature (K)", 250.0, 400.0, 310.0)

with col2:
    st.markdown('<p class="section-title">Mechanical</p>', unsafe_allow_html=True)
    rpm = st.number_input("Rotational Speed (rpm)", 0.0, 3000.0, 1500.0)
    torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0)
    tool_wear = st.number_input("Tool Wear (min)", 0.0, 300.0, 100.0)

with col3:
    st.markdown('<p class="section-title">Product</p>', unsafe_allow_html=True)
    product_type = st.selectbox("Product Type", ["L", "M", "H"])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Analyze Maintenance Risk", use_container_width=True):

    # Encode product type (drop_first logic)
    type_L = 1 if product_type == "L" else 0
    type_M = 1 if product_type == "M" else 0

    input_df = pd.DataFrame({
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type_L": [type_L],
        "Type_M": [type_M]
    })

    num_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    model = lr_model if model_choice == "Logistic Regression" else dt_model
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 0:
        st.markdown(f"""
        <div class="result risk-low">
            <div class="result-title">Low Risk</div>
            <p class="result-text">
                Machine operating normally.<br>
                Failure probability: <b>{probability:.2f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div class="result risk-high">
            <div class="result-title">High Risk</div>
            <p class="result-text">
                Potential machine failure detected.<br>
                Failure probability: <b>{probability:.2f}</b><br><br>
                <b>Recommended Actions:</b><br>
                • Schedule preventive maintenance<br>
                • Inspect torque and tool wear<br>
                • Monitor temperature and RPM
            </p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("""
<div class="footer">
Educational ML system — not a replacement for professional maintenance diagnosis
</div>
""", unsafe_allow_html=True)