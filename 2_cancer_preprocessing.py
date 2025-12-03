import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Read data
df = pd.read_csv('Breast_Cancer_Survival.csv')
print(f"Original shape: {df.shape}")

# Separate label column
y = df['Alive']
df = df.drop('Alive', axis=1)

# Encode categorical variables
# Binary encoding for Estrogen and Progesterone Status
if 'Estrogen Status' in df.columns:
    df['Estrogen Status'] = (df['Estrogen Status'] == 'Positive').astype(int)
if 'Progesterone Status' in df.columns:
    df['Progesterone Status'] = (df['Progesterone Status'] == 'Positive').astype(int)

# Label encoding for ordinal variables
ordinal_cols = ['Grade', 'T Stage ', 'N Stage', '6th Stage', 'differentiate', 'A Stage']
for col in ordinal_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# One-hot encoding for nominal variables
nominal_cols = ['Race', 'Marital Status']
for col in nominal_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=int)

print(f"Shape after encoding: {df.shape}")

# Identify continuous numerical columns
continuous_cols = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
continuous_cols = [col for col in continuous_cols if col in df.columns]

# Remove outliers from continuous columns only
df_clean = df.copy()
outlier_index = set()

for column in continuous_cols:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df_clean[column] < lower_bound) | (df_clean[column] >upper_bound)
        outlier_index.update(df_clean[outlier_mask].index)

# Remove all outliers rows at once
df_clean = df_clean.drop(index=outlier_index).reset_index(drop=True)
y = y.drop(index=outlier_index).reset_index(drop=True)

print(f"Shape after outlier removal: {df_clean.shape}")
print(f"Removed {len(outlier_index)} rows with outliers")

# Impute missing values with
for column in df_clean.columns:
    missing_before = df_clean[column].isnull().sum()
    if missing_before > 0:
        if df_clean[column].dtype in ['int64', 'float64']:
            median_val = df_clean[column].median()
            df_clean[column].fillna(median_val, inplace=True)
            print(f"{column}: filled {missing_before} missing values with median={median_val:.2f}")
        else:
            mode_vals = df_clean[column].mode()
            if len(mode_vals) > 0:
                df_clean[column].fillna(mode_vals[0],inplace=True)
                print(f"{column}: filled {missing_before} missing values with mode={mode_vals[0]}")
            else:
                df_clean[column].fillna("Unknown", inplace=True)
                print(f"{column}: filled {missing_before} missing values with 'Unknown'")

# Normalize only continuous numerical columns
scaler = StandardScaler()
df_clean[continuous_cols] = scaler.fit_transform(df_clean[continuous_cols])

# Add label column back
df_clean['Alive'] = y.values

# Save results
df_clean.to_csv('cancer_cleaned.csv', index=False)
with open('cancer_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"Final shape: {df_clean.shape}")
print(f"Final columns: {df_clean.columns.tolist()}")
print("Saved: cancer_cleaned.csv and cancer_scaler.pkl")


