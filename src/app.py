import streamlit as st
import pandas as pd
import joblib
import os
import time

# -------------------------------------------------
# Page Config (Premium Feel)
# -------------------------------------------------
st.set_page_config(
    page_title="VehicleCare360 | AI Maintenance",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom CSS (Elite Styling) - Single block to avoid markdown parsing
# -------------------------------------------------
css_content = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
:root {
    --primary: #2563eb;
    --primary-light: #dbeafe;
    --dark: #0f172a;
    --slate: #475569;
    --bg-light: #f8fafc;
    --success: #10b981;
    --danger: #ef4444;
}
* { font-family: 'Outfit', sans-serif !important; }
.main .block-container { 
    padding: 2rem 5rem; 
    max-width: 1300px;
    background-color: var(--bg-light);
}
.hero-section {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 3rem 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
}
.hero-title { font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: -0.02em; }
.hero-subtitle { font-size: 1.1rem; color: #94a3b8; font-weight: 300; }
[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #e2e8f0;
}
.stButton>button {
    background-color: var(--primary);
    color: white;
    border-radius: 10px;
    padding: 0.75rem 0;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    border: none;
    width: 100%;
}
.stButton>button:hover {
    background-color: #1d4ed8;
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
}
.result-box {
    padding: 2.5rem;
    border-radius: 16px;
    margin-top: 2rem;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: "";
    position: absolute;
    top: 0; left: 0; bottom: 0; width: 6px;
}
.res-low { background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }
.res-low::before { background: var(--success); }
.res-high { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
.res-high::before { background: var(--danger); }
.res-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.75rem; }
.res-desc { line-height: 1.7; font-size: 1.05rem; }
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}
.pulse {
    width: 8px; height: 8px; background: var(--success);
    border-radius: 50%; margin-right: 6px;
    animation: pulse-animation 2s infinite;
}
@keyframes pulse-animation {
    0% { box-shadow: 0 0 0 0px rgba(16, 185, 129, 0.4); }
    100% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
}
</style>
"""
st.markdown(css_content, unsafe_allow_html=True)

# -------------------------------------------------
# Asset Loader
# -------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_path, "models")
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        lr = joblib.load(os.path.join(models_dir, "logistic_model.pkl"))
        dt = joblib.load(os.path.join(models_dir, "decision_tree_model.pkl"))
        return scaler, lr, dt, models_dir
    except Exception as e:
        return None, None, None, None

scaler, lr_model, dt_model, models_dir = load_assets()

# -------------------------------------------------
# Sidebar Content
# -------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/car-service.png", width=100)
    st.markdown("### VehicleCare360")
    st.markdown("---")
    
    st.markdown("#### Configuration")
    model_choice = st.selectbox(
        "AI Engine Selection",
        ["Logistic Regression", "Decision Tree"],
        help="Select the machine learning model to use for prediction."
    )
    
    st.markdown("---")
    st.markdown("#### System Integrity")
    if scaler and lr_model and dt_model:
        st.success("✅ Neural Models Loaded")
    else:
        st.error("❌ Model Sync Failed")
    
    st.markdown("---")
    st.info("💡 **Tip:** Monitor Torque and Tool Wear as they are primary failure indicators.")

# -------------------------------------------------
# Hero Section
# -------------------------------------------------
st.markdown("""
<div class="hero-section">
    <div style="display: flex; justify-content: center; margin-bottom: 2rem;">
        <div class="status-pill"><div class="pulse"></div>SYSTEM OPERATIONAL</div>
    </div>
    <h1 class="hero-title">Predictive Maintenance AI</h1>
    <p class="hero-subtitle">Industrial-grade vehicle health monitoring system powered by advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Input Matrix
# -------------------------------------------------
st.markdown("### <i class='fas fa-sliders-h'></i> Operational Parameters", unsafe_allow_html=True)

input_col1, input_col2 = st.columns([2, 1], gap="large")

with input_col1:
    tab1, tab2 = st.tabs(["Mechanical State", "Thermal Profile"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            rpm = st.slider("Rotational Speed (RPM)", 0, 3000, 1500, step=10)
            torque = st.slider("Torque (Nm)", 0.0, 100.0, 40.0, step=0.5)
        with c2:
            tool_wear = st.slider("Tool Wear (Minutes)", 0, 300, 100)
            product_type = st.radio("Product Grade", ["L (Low)", "M (Medium)", "H (High)"], horizontal=True)
            p_type_code = product_type[0]

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            air_temp = st.number_input("Air Temp (K)", 280.0, 350.0, 300.0)
        with c2:
            process_temp = st.number_input("Process Temp (K)", 280.0, 370.0, 310.0)

with input_col2:
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 1rem;">
        <h4 style="margin-top:0;">Live Telemetry</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #64748b;">Torque Load</span>
            <span style="font-weight: 600;">{torque} Nm</span>
        </div>
        <div style="width: 100%; background: #f1f5f9; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="width: {torque}%; height: 100%; background: #2563eb;"></div>
        </div>
        <div style="margin-top: 1.5rem; display: flex; justify-content: space-between;">
            <span style="color: #64748b;">Wear Rate</span>
            <span style="font-weight: 600;">{(tool_wear/300)*100:.1f}%</span>
        </div>
        <div style="width: 100%; background: #f1f5f9; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="width: {(tool_wear/300)*100}%; height: 100%; background: {"#ef4444" if tool_wear > 200 else "#10b981"};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# Execution Hub
# -------------------------------------------------
if st.button("RUN SYSTEM DIAGNOSTIC", use_container_width=True):
    with st.spinner("Processing sensor matrix..."):
        time.sleep(1)
        
        if scaler and lr_model and dt_model:
            # Prepare Input (Drop Type_H as model was trained with L and M only)
            type_L = 1 if p_type_code == "L" else 0
            type_M = 1 if p_type_code == "M" else 0

            input_df = pd.DataFrame({
                "Air temperature [K]": [air_temp],
                "Process temperature [K]": [process_temp],
                "Rotational speed [rpm]": [rpm],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear],
                "Type_L": [type_L],
                "Type_M": [type_M]
            })

            # Scale Numerical Features
            num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
            try:
                input_df[num_cols] = scaler.transform(input_df[num_cols])
                
                # Predict
                model = lr_model if model_choice == "Logistic Regression" else dt_model
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                # Display Results
                if prediction == 0:
                    st.markdown(f"""
                    <div class="result-box res-low">
                        <div class="res-title"><i class="fas fa-shield-check"></i> NORMAL OPERATION</div>
                        <div class="res-desc">
                            System diagnostic indicates <b>Optimal Health</b>. Currently operating within safety bounds.<br>
                            <span style="font-size: 0.9rem; opacity: 0.8;">Calculated risk of failure: <b>{probability*100:.2f}%</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class="result-box res-high">
                        <div class="res-title"><i class="fas fa-engine-warning"></i> CRITICAL RISK DETECTED</div>
                        <div class="res-desc">
                            Warning: High probability of mechanical failure (<b>{probability*100:.2f}%</b>).<br><br>
                            <b>STRATEGIC INTERVENTIONS:</b><br>
                            • Immediate shutdown of high-torque operations requested.<br>
                            • Inspect <b>Tool Wear</b> sensors (Current: {tool_wear} min).<br>
                            • Verify spindle alignment and thermal stability.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Execution Error: {e}")
        else:
            st.error("System Core Offline: Models not found.")

# -------------------------------------------------
# Analytics Footer
# -------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📊 Data Analytics & Model Insights"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Feature Correlation")
        st.info("The model prioritizes **Torque** and **Tool Wear** as the two most decisive factors for vehicle health.")
        if models_dir and os.path.exists(os.path.join(models_dir, "confusion_matrix.png")):
            st.image(os.path.join(models_dir, "confusion_matrix.png"), caption="Model Performance Matrix")
    with col_b:
        st.markdown("#### Model Architecture")
        st.json({
            "Engine": model_choice,
            "State": "Active",
            "Input Features": 7,
            "Output": "Binary Classification (Fail/No-Fail)",
            "Last Sync": time.strftime("%H:%M:%S")
        })

st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 4rem; padding-bottom: 2rem;">
    © 2024 VehicleCare360 AI Systems • Version 2.0.6-Stable
</div>
""", unsafe_allow_html=True)
