import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Read data
df = pd.read_csv('diabetes_project.csv')
print(f"Original shape: {df.shape}")

# Replace 0 with NaN for biologically implausible columns
zero_not_acceptable = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
df_clean = df.copy()
for col in zero_not_acceptable:
    df_clean[col] = df_clean[col].replace(0, np.nan)

# Remove outliers
iqr_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
outlier_index = set()

for column in iqr_columns:
    if column not in df_clean.columns:
        continue

    Q1 = df_clean[column].quantile(0.25)
    Q3 = df_clean[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
    outlier_index.update(df_clean[outlier_mask].index)

# Remove all outlier rows at once
df_clean = df_clean.drop(index=outlier_index).reset_index(drop=True)
print(f'Shape after outlier removal: {df_clean.shape}')
print(f"Removed {len(outlier_index)} rows with outliers")

# Impute missing values with median
for column in df_clean.columns:
    missing_before = df_clean[column].isnull().sum()
    if missing_before > 0:
        median_val = df_clean[column].median()
        df_clean[column].fillna(median_val, inplace=True)
        print(f"{column}: filled {missing_before} missing values with median {median_val:.2f}")

# Normalize
scaler = StandardScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=df_clean.columns
)

# Save results
df_normalized.to_csv('diabetes_cleaned.csv', index=False)
with open('diabetes_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f'Final shape: {df_normalized.shape}')
print("Saved: diabetes_cleaned.csv and diabetes_scaler.pkl")