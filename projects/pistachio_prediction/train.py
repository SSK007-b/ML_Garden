import yaml
import pandas as pd
from common.preprocessing import DataPreprocessor
from common.training import DataTrainer
from common.evaluation import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

def main():
    # Load configuration
    config_path = 'D:\Github Sanity\ML Garden\projects\pistachio_prediction\config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load dataset
    data_path = config.get('dataset_path', 'raw.csv')
    df = pd.read_csv(data_path)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    categorical_cols = config.get('categorical_features', [])
    df = preprocessor.data_encoding(df, categorical_cols)

    df = df[config["selected_features"] + config["dependent_features"]]
    print(f"Dataframe shape after feature selection: {df}")

    # Train-test split
    trainer = DataTrainer(config_path)
    X_train, X_test, y_train, y_test = trainer.split_data(df)

    rf = RandomForestClassifier(n_estimators=50, random_state=config['random_state'])
    ada = AdaBoostClassifier(n_estimators=50, random_state=config['random_state'])
    knn = KNeighborsClassifier(n_neighbors=5)

    model = VotingClassifier(estimators=[('rf', rf), ('ada', ada), ('knn', knn)], voting='soft')
    trained_model = trainer.train_model(model, X_train, y_train)

    # Evaluate model
    evaluator = ModelEvaluator()
    y_pred = model.predict(X_test)
    evaluation_results = evaluator.evaluate(y_test, y_pred)
    print("Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    main()