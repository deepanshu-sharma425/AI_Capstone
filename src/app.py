import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="VehicleCare360",
    layout="wide"
)

# -------------------------------------------------
# Dark UI Styling
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #020617, #020617);
    color: #e5e7eb;
}
.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #e5e7eb;
    text-align: center;
}
.subtitle {
    color: #94a3b8;
    text-align: center;
    margin-bottom: 2rem;
}
.section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #93c5fd;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem;
    font-size: 1rem;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
}
input, select {
    background-color: #020617 !important;
    color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("<div class='main-title'>VehicleCare360</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-Powered Predictive Vehicle Maintenance System</div>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# Load Models
# -------------------------------------------------
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
    dt = joblib.load(os.path.join(MODEL_DIR, "decision_tree_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return lr, dt, scaler

try:
    lr_model, dt_model, scaler = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
model_choice = st.selectbox(
    "Select Prediction Model",
    ["Logistic Regression", "Decision Tree"]
)

model = lr_model if model_choice == "Logistic Regression" else dt_model

# -------------------------------------------------
# Input Layout
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

# ---------- Temperature ----------
with col1:
    st.markdown("<div class='section-title'>Temperature</div>", unsafe_allow_html=True)

    air_temp = st.slider(
        "Air Temperature (K)", 250.0, 350.0, 300.0, 1.0
    )

    process_temp = st.slider(
        "Process Temperature (K)", 250.0, 350.0, 310.0, 1.0
    )

# ---------- Mechanical ----------
with col2:
    st.markdown("<div class='section-title'>Mechanical</div>", unsafe_allow_html=True)

    rpm = st.slider(
        "Rotational Speed (RPM)", 500, 3000, 1500, 50
    )

    torque = st.slider(
        "Torque (Nm)", 10.0, 100.0, 40.0, 1.0
    )

    tool_wear = st.slider(
        "Tool Wear (min)", 0.0, 300.0, 100.0, 5.0
    )

# ---------- Product ----------
with col3:
    st.markdown("<div class='section-title'>Product</div>", unsafe_allow_html=True)

    product_label = st.selectbox(
        "Product Type",
        ["Light", "Medium", "Heavy"]
    )

# -------------------------------------------------
# Correct One-Hot Encoding (NO Type_H)
# -------------------------------------------------
type_L = 1 if product_label == "Light" else 0
type_M = 1 if product_label == "Medium" else 0
# Heavy = both 0 (implicit)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analyze Maintenance Risk"):
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

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.error(
            f"""
            **High Risk**

            Potential machine failure detected.  
            Failure probability: **{probability:.2f}**

            **Recommended Actions**
            - Schedule preventive maintenance
            - Inspect torque and tool wear
            - Monitor temperature and RPM
            """
        )
    else:
        st.success(
            f"""
            **Low Risk**

            Machine operating normally.  
            Failure probability: **{probability:.2f}**
            """
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:0.8rem; margin-top:2rem;'>"
    "Educational ML system — not a replacement for professional maintenance diagnosis"
    "</div>",
    unsafe_allow_html=True
)