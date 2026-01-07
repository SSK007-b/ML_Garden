import pandas as pd
from common.preprocessing import DataPreprocessor
from common.inference import ModelInferencer


def main():
    # Load configuration
    config_path = "D:\Github Sanity\ML Garden\projects\heart_failure\config.yaml"
    preprocessor = DataPreprocessor()
    predictor = ModelInferencer(config_path)

    columns = [
        "age", "anaemia", "creatinine_phosphokinase",
        "ejection_fraction", "high_blood_pressure",
        "platelets", "serum_creatinine",
        "serum_sodium", "smoking", "time"
    ]   

    df = pd.DataFrame([[58, 0, 582, 38, 1, 263000, 1.1, 137, 0, 120]], columns=columns)

    data = preprocessor.data_scaling(df)

    predictions = predictor.predict(data)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()