import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        pass
    
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Evaluate the model's performance using accuracy, precision, and recall.

        Parameters:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.

        Returns:
        dict: A dictionary containing accuracy, precision, and recall scores.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        logger.info(f"Evaluation results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }