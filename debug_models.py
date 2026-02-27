import joblib
import pandas as pd
import sys

try:
    lr = joblib.load('models/logistic_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    print("LR Features:", lr.feature_names_in_)
    print("Scaler Features:", scaler.get_feature_names_out() if hasattr(scaler, 'get_feature_names_out') else "N/A")
except Exception as e:
    print(f"Error: {e}")
