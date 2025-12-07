import numpy as np
import torch
import pandas as pd
from typing import Tuple


class SequenceSlider:
    """Class for creating input sequences from a DataFrame for time-series modeling."""
    
    def __init__(self, window_size: int):
        """
        Initialize the SequenceSlider with a specified sequence length.

        Args:
            window_size (int): The length of each sequence to generate.
        """
        self.window_size = window_size
        
    def slide(self, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate input sequences and their corresponding labels from the given DataFrame.
        
        Args:
            dataframe (pd.DataFrame): The input DataFrame.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Generated sequences (X) and corresponding labels (y).
        """
        X, y = [], []
        for i in range(len(dataframe) - self.window_size):
            Xi = dataframe[i:(i + self.window_size)]
            yi = dataframe[i + self.window_size]
            X.append(Xi)
            y.append(yi)
        return np.array(X), np.array(y)
    
    def transform(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform the train and test DataFrames into tensors for subsequent modeling.

        Args:
            train_data (pd.DataFrame): The training data DataFrame.
            test_data (pd.DataFrame): The test data DataFrame.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors for training and testing inputs and labels.
        """
        X_train, y_train = self.slide(train)
        X_test, y_test = self.slide(test)
        
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()
        
        return X_train, y_train, X_test, y_test


