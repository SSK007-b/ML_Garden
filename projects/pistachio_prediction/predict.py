import pandas as pd
from common.inference import ModelInferencer

def main():
    # Load configuration
    config_path = "D:\Github Sanity\ML Garden\projects\pistachio_prediction\config.yaml"

    predictor = ModelInferencer(config_path)

    columns = [
        "AREA", "PERIMETER", "MINOR_AXIS", "ECCENTRICITY",
        "EQDIASQ", "CONVEX_AREA", "ASPECT_RATIO",
        "COMPACTNESS", "SHAPEFACTOR_1", "SHAPEFACTOR_3"
    ]

    df = pd.DataFrame(
        [[31245.6, 845.3, 178.4, 0.76, 199.5, 32890.1, 1.85, 0.62, 0.045, 0.71]],
        columns=columns
    )

    predictions = predictor.predict(df)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()