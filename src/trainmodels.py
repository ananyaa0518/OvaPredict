# ================================
# PCOS PREDICTION (FULL FIXED)
# ================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv("data/cleaned_pcos.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")
print(df['PCOS (Y/N)'].value_counts(normalize=True))


# ================================
# 2. CLEAN NON-NUMERIC VALUES 🚨
# ================================

# Replace common garbage values with NaN
df.replace(['a', 'A', '?', ' ', ''], np.nan, inplace=True)

# Convert all columns to numeric (force)
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values
df.fillna(df.mean(), inplace=True)


# ================================
# 3. SPLIT FEATURES & TARGET
# ================================
X = df.drop('PCOS (Y/N)', axis=1)
y = df['PCOS (Y/N)']


# ================================
# 4. TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ================================
# 5. APPLY SMOTE
# ================================
print("\nBefore SMOTE:\n", y_train.value_counts())

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n", y_train_smote.value_counts())


# ================================
# 6. DEFINE MODELS
# ================================
models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        max_iter=5000,
        class_weight='balanced',
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
}


# ================================
# 7. TRAIN & EVALUATE
# ================================
results = []

for name, model in models.items():
    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)

    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)

    # ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = "N/A"

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })


# ================================
# 8. FINAL RESULTS
# ================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

import joblib

# Save results table
results_df.to_csv("data/model_results.csv", index=False)

# Save best model (Random Forest)
best_model = models["Random Forest"]
joblib.dump(best_model, "data/random_forest_model.pkl")

print("\n✅ Results and model saved successfully!")