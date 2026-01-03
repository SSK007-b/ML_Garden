import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def data_encoding(self, df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """
        Encode categorical columns using Label Encoding.

        Parameters:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list): List of column names to be encoded.

        Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
        """
        for col in categorical_cols:
            if col in df.columns:
                df[col] = self.encoder.fit_transform(df[col])
                logger.info(f"Encoded column: {col}")
            else:
                logger.warning(f"Column {col} not found in DataFrame.")
        return df
    
    def data_decoding(self):
        pass
    
    def data_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical columns using Standard Scaling.

        Parameters:
        df (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with scaled numerical columns.
        """
        data = self.scaler.fit_transform(df)
        logger.info("Data scaling completed.")
        return data