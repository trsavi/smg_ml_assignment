#!/usr/bin/env python3
"""
Metrics calculation utilities for Madrid Housing Market pipeline.

This module provides utilities for calculating and managing model performance metrics.
"""

# Standard library imports
from typing import Dict

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports
from utils.file_manager import FileManager


class MetricsCalculator:
    """Utility class for calculating model performance metrics."""
    
    def __init__(self, file_manager: FileManager = None):
        """
        Initialize the metrics calculator.
        
        Args:
            file_manager (FileManager, optional): FileManager instance for logging.
                                                If None, creates a new instance.
                                                
        Returns:
            None: Initializes the MetricsCalculator instance.
            
        Example:
            >>> mc = MetricsCalculator(file_manager)
            >>> mc = MetricsCalculator()  # Creates new FileManager
        """
        self.file_manager = file_manager or FileManager()
    
    def calculate_metrics(self, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict[str, float]:
        """Calculate metrics for a given dataset.

        Args:
            X: Features
            y: Target values
            dataset_name: Name of the dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        # This method will be called from train_model.py, so we need to handle the case
        # where model is not available in this context
        # We'll need to pass the model as a parameter
        
        raise NotImplementedError("This method should be called with a model parameter")
    
    def calculate_metrics_with_model(self, model, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict[str, float]:
        """
        Calculate metrics for a given dataset with a specific model.
        
        Args:
            model: Trained model to use for predictions.
            X (pd.DataFrame): Features.
            y (pd.Series): Target values.
            dataset_name (str): Name of the dataset (for logging).
            
        Returns:
            Dict[str, float]: Dictionary of metrics (rmse, mae, r2).
            
        Raises:
            ValueError: If no model is provided.
            
        Example:
            >>> metrics = mc.calculate_metrics_with_model(model, X_test, y_test, "test")
        """
        if model is None:
            raise ValueError("No model provided for evaluation")

        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        metrics = {
            f'{dataset_name}_rmse': rmse,
            f'{dataset_name}_mae': mae,
            f'{dataset_name}_r2': r2
        }

        print(f"{dataset_name.capitalize()} RMSE: {rmse:.2f}")
        print(f"{dataset_name.capitalize()} MAE: {mae:.2f}")
        print(f"{dataset_name.capitalize()} R²: {r2:.3f}")

        return metrics
    
    def calculate_comprehensive_metrics(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                      X_val: pd.DataFrame, y_val: pd.Series,
                                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive metrics for train, validation, and test sets.
        
        Args:
            model: Trained model.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation targets.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test targets.
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing metrics for each dataset.
            
        Example:
            >>> all_metrics = mc.calculate_comprehensive_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test)
        """
        train_metrics = self.calculate_metrics_with_model(model, X_train, y_train, 'train')
        val_metrics = self.calculate_metrics_with_model(model, X_val, y_val, 'val')
        test_metrics = self.calculate_metrics_with_model(model, X_test, y_test, 'test')

        return {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
