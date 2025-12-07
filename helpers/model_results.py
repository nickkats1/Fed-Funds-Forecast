from scripts.data_ingestion import DataIngestion
from helpers.config import load_config
from helpers.logger import logger

import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error,r2_score,root_mean_squared_error



class ModelResults:
    """Utility class for storing results from 'sklearn models' portion of this project.
    
    Args:
        :config (dict): config file.
        :results (dict): A dictionary the holds all metrics for each model during training.
        :evaluate (list): A list to store all of the results(dict) for display.

    Returns:
        :results (pd.DataFrame): a dataframe of the models and metrics.
    """
    
    def __init__(self,config: dict, data_ingestion: DataIngestion | None = None):
        """Initializing Model Results Class.
        
        Args:
            config (dict): config file.
            results (dict): A dictionary the holds all metrics for each model during training.
            evaluate (list): A list to store all of the results(dict) for display.
            data_ingestion (DataIngestion): A instance of DataIngestion module to attain the data.
        Returns:
            results (pd.DataFrame): a dataframe of the models and metrics.
        """
        
        self.config = config or load_config()
        self.data_ingestion = data_ingestion or DataIngestion(self.config)
        self.results = []
        self.evaluate = {}
        
        
    def evaluate_results(self,y_test, y_pred, r2, mape, rmse, cv_scores, model_name):
        """"""
        
        results = {
            "Model":model_name,
            "R2 Score":r2,
            "Mean-Absolute Percentage Error":mape,
            "Cross-Val Score":cv_scores,
            "Root Mean-Squared Error":rmse
        }
        return results
    
    





