import pandas as pd
from common.inference import ModelInferencer

def main():
    # Load configuration
    config_path = "D:\Github Sanity\ML Garden\projects\loan_approval\config.yaml"
    predictor = ModelInferencer(config_path)

    columns = [
        "no_of_dependents", "education", "self_employed",
        "income_annum", "loan_amount", "loan_term",
        "cibil_score", "residential_assets_value",
        "commercial_assets_value", "luxury_assets_value",
        "bank_asset_value"
    ]

    df = pd.DataFrame([[
        2, 1, 0, 850000, 2400000, 20, 748,
        3500000, 1200000, 450000, 620000
    ]], columns=columns)

    predictions = predictor.predict(df)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()