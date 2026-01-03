import yaml
import joblib
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully.")

    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split the DataFrame into training and testing sets.

        Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target column.

        Returns:
        tuple: Training and testing sets (X_train, X_test, y_train, y_test).
        """
        X = df[self.config.get('independent_features', [])]
        y = df[self.config.get("dependent_features", [])]
        
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split into training and testing sets with test size {test_size}.")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the provided model using the training data.

        Parameters:
        model: A machine learning model with a fit method.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

        Returns:
        The trained model.
        """
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        model_path = self.config.get('model_path', 'trained_model.joblib')
        joblib.dump(model, model_path)
        logger.info(f"Trained model saved to {model_path}.")