# Loan Approval Prediction (XGBoost)

## Project Overview
This project predicts loan approval status based on applicant financial, credit, and asset-related information.
The goal is to build a robust classification pipeline that mimics real-world banking decision systems.

## Model Used
- XGBoost Classifier
- Handles non-linear relationships well
- Robust to feature scaling
- Performs strongly on tabular financial data

## Features Used
- Applicant details (dependents, education, self-employment)
- Financials (annual income, loan amount, loan term)
- Credit history (CIBIL score)
- Asset values (residential, commercial, luxury, bank)

## Pipeline Flow

- Data loading & validation
- Categorical encoding (binary features)
- Feature selection
- Model training using XGBoost
- Prediction & evaluation

## Output
- Binary prediction: Loan Approved / Not Approved
- Can be extended to probability-based risk scoring

## Why This Project?
- Demonstrates end-to-end ML workflow
- Realistic finance use case
- Easily extensible to deployment (API / Streamlit)

## Tech Stack
- Python
- Pandas, NumPy
- XGBoost
- Scikit-learn