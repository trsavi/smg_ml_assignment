#!/usr/bin/env python3
"""
Data utilities for Madrid Housing Market pipeline.

This module provides utilities for data loading, preprocessing checks, and data management.
"""

# Standard library imports
import os
from typing import Dict, Tuple

# Third-party imports
import pandas as pd

# Local imports
from utils.file_manager import FileManager


class DataManager:
    """Utility class for data management operations."""
    
    def __init__(self, file_manager: FileManager = None):
        """
        Initialize the data manager.
        
        Args:
            file_manager (FileManager, optional): FileManager instance for file operations.
                                                If None, creates a new instance.
                                                
        Returns:
            None: Initializes the DataManager instance.
            
        Example:
            >>> dm = DataManager(file_manager)
            >>> dm = DataManager()  # Creates new FileManager
        """
        self.file_manager = file_manager or FileManager()
    
    def check_preprocessed_data(self) -> bool:
        """
        Check if preprocessed data exists.
        
        Args:
            None: Checks for default preprocessed data path.
            
        Returns:
            bool: True if preprocessed data exists, False otherwise.
            
        Example:
            >>> exists = dm.check_preprocessed_data()
        """
        return os.path.exists("data/preprocessed_houses_Madrid.csv")
    
    def prepare_data_if_needed(self) -> None:
        """
        Prepare data if preprocessed data doesn't exist.
        
        This method checks if preprocessed data exists and runs data preparation
        if it doesn't.
        
        Args:
            None: Uses internal checks and data preparation script.
            
        Returns:
            None: Prepares data if needed.
            
        Example:
            >>> dm.prepare_data_if_needed()
        """
        if not self.check_preprocessed_data():
            print("Preprocessed data not found. Running data preparation...")
            from scripts.data_prep import main as run_data_prep
            run_data_prep()
            print("Data preparation completed.")
        else:
            print("Using existing preprocessed data")
    
    def load_preprocessed_data(self) -> pd.DataFrame:
        """Load preprocessed data.
        
        Returns:
            Preprocessed DataFrame
        """
        return self.file_manager.load_dataframe("data/preprocessed_houses_Madrid.csv")
    
    def prepare_data_splits(self, data: pd.DataFrame, test_size: float = 0.2, 
                           val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Prepare train/validation/test data splits.
        
        Args:
            data: Input DataFrame
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        X = data.drop('buy_price', axis=1)
        y = data['buy_price']
        
        print(f"Loaded preprocessed data: {data.shape}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data_version_info(self) -> Dict[str, str]:
        """Get data version information.
        
        Returns:
            Dictionary with data version information
        """
        data_info = {}
        
        # Check if preprocessed data exists
        if self.check_preprocessed_data():
            data_info["data_source"] = "preprocessed_houses_Madrid.csv"
            data_info["data_type"] = "preprocessed"
        else:
            data_info["data_source"] = "houses_Madrid.csv"
            data_info["data_type"] = "raw"
        
        return data_info
