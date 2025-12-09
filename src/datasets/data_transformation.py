from src.datasets.data_ingestion import DataIngestion
from tools.config import load_config
from tools.logger import logger
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from constants import *
import pandas as pd
from typing import Any


class DataTransformation:
    """
    A utility class to transform data using MinMaxScaler.
    """
    
    def __init__(self, config: dict, data_ingestion: DataIngestion | None = None):
        """
        Initialize DataTransformation class.
        
        Args:
            config (dict): a configuration file for loading files from data_ingestion(Optional).
            data_ingestion (DataIngestion): a instance of data_ingestion.
            
        Returns:
            train_scaled (np.array): A array of training data scaled using MinMaxScaler.
            test_scaled (np.array): A array of testing data scaled using MinMaxScaler.
        """
        self.config = config or load_config()
        self.data_ingestion = data_ingestion or DataIngestion(self.config)
        
        
    def split(self) -> pd.DataFrame:
        """
        Splits data into training and testing arrays.
        
        Returns:
            train_scaled (np.array): A np.array of training data scaled.
            test_scaled (np.array): A np.array of testing data scaled.
        """
        # data from DataIngestion module
        data = self.data_ingestion.fetch_fred_data()
        
        # split data
        training = data.iloc[:,0:1].values
        
        logger.info(f"shape of training data: {training.shape}")
        
        
        # select training and testing data length
        
        train_size = int(len(training)*0.80)
        test_size = int(len(training)) - train_size
        
        train_data = training[:train_size]
        test_data = training[:test_size]
        logger.info(f"Length of training data: {len(train_data)}")
        logger.info(f"Length of testing data: {len(test_data)}")
        
        
        return train_data,test_data
    
    
    def transform(self) -> Any:
        """Transform training and testing data using minmaxscaler"""
        scaler = MinMaxScaler()
    
    
        # training and testing data
        train_data,test_data = self.split()
        
        # scale training and testing data
        
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
  
        
        logger.info(f"Shape of training data scaled: {train_scaled.shape}")
        logger.info(f"Shape of testing data: {test_scaled.shape}")
        
        return train_scaled,test_scaled

