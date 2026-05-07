# OvaPredict: PCOS Risk Analysis & Prediction

OvaPredict is a machine learning-based web application designed to predict the risk of Polycystic Ovary Syndrome (PCOS) using clinical and lifestyle-related health parameters.

##  Features
- PCOS risk prediction using Machine Learning
- Random Forest Classifier with ~91% accuracy
- Interactive Streamlit web interface
- Data preprocessing and SMOTE balancing
- Multiple model comparison:
  - Logistic Regression
  - Linear SVM
  - Decision Tree
  - Random Forest

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## ML Workflow
1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Handling Missing Values
4. SMOTE for class balancing
5. Model Training & Evaluation
6. Model Saving using `.pkl`

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|------|------|------|------|------|
| Logistic Regression | 0.8257 | 0.6977 | 0.8333 | 0.7595 | 0.9159 |
| Linear SVM | 0.8716 | 0.7619 | 0.8889 | 0.8205 | N/A |
| Random Forest | 0.9083 | 0.8824 | 0.8333 | 0.8571 | 0.9140 |
| Decision Tree | 0.8532 | 0.7941 | 0.7500 | 0.7714 | 0.8834 |

## Best Model
**Random Forest Classifier**
- Accuracy: **90.8%**
- F1 Score: **0.8571**
- ROC-AUC: **0.9140**

## Run the App

Install dependencies:

```bash
pip install -r requirements.txt