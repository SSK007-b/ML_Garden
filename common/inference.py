import yaml
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInferencer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        model_path = self.config.get('model_path', 'trained_model.joblib')
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}.")

    def predict(self, X: any) -> any:
        """
        Make predictions using the loaded model.

        Parameters:
        X: Input features for prediction.

        Returns:
        Predictions made by the model.
        """
        predictions = self.model.predict(X)
        logger.info("Predictions made successfully.")
        return predictions