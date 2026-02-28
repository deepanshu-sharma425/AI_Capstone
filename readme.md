# VehicleCare360  
### From Predictive Maintenance to Agentic Fleet Support  

An AI-driven platform to **monitor, predict, and support vehicle maintenance decisions** using **classical Machine Learning** and **Agentic AI workflows**.

---

##  Overview  

**VehicleCare360** enables proactive fleet and machine maintenance by leveraging sensor data and intelligent automation.

The system helps organizations:
- Predict potential machine failures before breakdowns occur  
- Reduce downtime through preventive maintenance  
- Transition from static ML predictions to autonomous decision-making agents  

---

## 🔍 Key Capabilities  

### 1️⃣ Predictive Maintenance  
- Uses sensor and operational data (temperature, torque, RPM, tool wear)  
- Identifies machines at **risk of failure**  
- Supports early intervention and cost optimization  

### 2️⃣ Autonomous Fleet Support (Planned – Phase 2)  
- Agentic AI analyzes ML predictions  
- Retrieves maintenance guidelines  
- Generates structured service recommendations  

---

##  System Architecture  

### Phase 1 – Predictive Risk Engine (Mid-Sem)  

**Objective:**  
Binary classification of machine failure risk (Failure / No Failure)

**Tech Stack:**  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  

**ML Workflow:**  

**Models Used:**  
- Logistic Regression (interpretable baseline model)  
- Decision Tree Classifier (captures non-linear behavior)  

---

### Phase 2 – Agentic Fleet Management Assistant (End-Sem)  

**Objective:**  
Convert failure predictions into **actionable maintenance plans**

**Tech Stack:**  
- LangGraph (agent workflow & state management)  
- ChromaDB / FAISS (retrieval of maintenance guidelines)  
- Open-source / free-tier LLMs  

**Agent Logic:**  

---

## 📊 Dataset  

**AI4I 2020 Predictive Maintenance Dataset**

### Features  

**Sensor & Operational Data**
- Air Temperature (K)  
- Process Temperature (K)  
- Rotational Speed (rpm)  
- Torque (Nm)  
- Tool Wear (min)  

**Categorical**
- Product Type (L, M, H)  

**Target Variable**
- `Machine failure` (0 = No Failure, 1 = Failure)

---

## 📈 Exploratory Data Analysis  

Key insights from EDA:
- Failure events are **imbalanced**, requiring stratified sampling  
- Torque, rotational speed, and tool wear strongly influence failures  
- High temperatures often correlate with increased risk  

EDA notebooks are included in the `notebooks/` directory.

---

## 🗂️ Repository Structure  
VehicleCare360/
├── data/
│   ├── raw/                     # Original AI4I 2020 dataset
│   │   └── ai4i2020.csv
│   └── processed/               # Cleaned & feature-engineered data
│       └── cleaned_ai4i_data.csv
│
├── models/                      # Trained ML models & preprocessing artifacts
│   ├── logistic_model.pkl
│   ├── decision_tree_model.pkl
│   └── scaler.pkl
│
├── notebooks/                   # Research & experimentation notebooks
│   ├── Data_Cleaning.ipynb      # Data preprocessing & EDA
│   └── Model_Training.ipynb     # Model training & evaluation
│
├── docs/                        # Documentation & evaluation artifacts
│   ├── 
│   ├── confusion_matrix_decision_tree.png
│   └── feature_importance_decision_tree.png
│
├── src/
│   └── app.py                   # Streamlit application (prediction UI)
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation