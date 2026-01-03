import yaml
import pandas as pd
from common.preprocessing import DataPreprocessor
from common.training import DataTrainer
from common.evaluation import ModelEvaluator
from xgboost import XGBClassifier

def main():
    # Load configuration
    config_path = 'D:\\Github Sanity\\ML Garden\\projects\\loan_approval\\config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load dataset
    data_path = config.get('dataset_path', 'loan_data.csv')
    df = pd.read_csv(data_path)

    df.columns = df.columns.str.strip().str.lower()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    categorical_cols = config.get('categorical_features', [])
    df = preprocessor.data_encoding(df, categorical_cols)

    # Train-test split
    trainer = DataTrainer(config_path)
    X_train, X_test, y_train, y_test = trainer.split_data(df)

    model = XGBClassifier()
    trained_model = trainer.train_model(model, X_train, y_train)

    # Evaluate model
    evaluator = ModelEvaluator()
    y_pred = model.predict(X_test)
    evaluation_results = evaluator.evaluate(y_test, y_pred)
    print("Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    main()