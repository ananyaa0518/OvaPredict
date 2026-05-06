# ================================
# PCOS PREDICTION
# ================================

import joblib
import pandas as pd

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("data/random_forest_model.pkl")

# -------------------------------
# Load training dataset
# -------------------------------
df = pd.read_csv("data/cleaned_pcos.csv")

# Remove target column
X_columns = df.drop("PCOS (Y/N)", axis=1).columns

# -------------------------------
# Sample patient data
# MUST match feature count/order
# -------------------------------
sample_data = [[
    25, 60, 165, 22.0,
    13, 72, 18,
    12.5, 2, 30,
    3, 0,
    1, 1.2,
    1.3, 36, 30,
    0.83, 2.5, 4.2,
    15, 25, 1.1,
    90, 1, 1,
    0, 1,
    1, 0,
    1, 120,
    80, 8,
    7, 14,
    15, 9
]]

# -------------------------------
# Convert to DataFrame
# -------------------------------
sample_df = pd.DataFrame(sample_data, columns=X_columns)

# -------------------------------
# Predict
# -------------------------------
prediction = model.predict(sample_df)

# -------------------------------
# Output
# -------------------------------
if prediction[0] == 1:
    print("\n⚠️ High Risk of PCOS")
else:
    print("\n✅ Low Risk of PCOS")