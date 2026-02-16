import pandas as pd

df = pd.read_csv("data/ai4i2020.csv")

print("First 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())
