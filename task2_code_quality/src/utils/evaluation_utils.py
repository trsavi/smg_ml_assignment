#!/usr/bin/env python3
"""
Evaluation utilities for Madrid Housing Market pipeline.

This module provides utilities for model evaluation and performance assessment.
"""

# Standard library imports
from typing import Dict

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """Utility class for model evaluation operations."""
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        if model is None:
            raise ValueError("No model to evaluate. Train model first.")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        print("=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Test RMSE: {rmse:.2f}")
        print(f"Test MAE: {mae:.2f}")
        print(f"Test R²: {r2:.3f}")
        print("=" * 60)

        return metrics
