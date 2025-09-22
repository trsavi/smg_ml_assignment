#!/usr/bin/env python3
"""
MLflow utilities for Madrid Housing Market pipeline.

This module provides utilities for logging experiments, metrics, and models to MLflow.
"""

# Standard library imports
import os
from typing import Any, Dict, Optional

# Third-party imports
import mlflow
import mlflow.lightgbm
import pandas as pd

# Local imports
from utils.file_manager import FileManager


class MLflowLogger:
    """Utility class for MLflow logging operations."""
    
    def __init__(self, file_manager: FileManager = None):
        """Initialize the MLflow logger.
        
        Args:
            file_manager: FileManager instance for file operations
        """
        self.file_manager = file_manager or FileManager()
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking URI and experiment."""
        # Set tracking URI
        mlflow.set_tracking_uri("./mlruns")
        
        # Set experiment name
        experiment_name = "Madrid Housing Market"
        mlflow.set_experiment(experiment_name)
    
    def log_training_run(self, model, metrics: Dict[str, float] = None, 
                        all_metrics: Dict[str, Dict[str, float]] = None,
                        run_name: str = None, run_type: str = 'training') -> str:
        """Log a training run to MLflow.
        
        Args:
            model: Trained model to log
            metrics: Single set of metrics to log
            all_metrics: Comprehensive metrics for train/val/test
            run_name: Name for the run
            run_type: Type of run (training, evaluation, etc.)
            
        Returns:
            Run ID of the logged run
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log data version info
            data_version_info = self._get_data_version_info()
            mlflow.log_params(data_version_info)
            
            # Log metrics
            if all_metrics:
                for dataset_name, dataset_metrics in all_metrics.items():
                    print(f"Logging {dataset_name} metrics: {list(dataset_metrics.keys())}")
                    mlflow.log_metrics(dataset_metrics)
            elif metrics:
                mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.lightgbm.log_model(
                model,
                "model",
                registered_model_name="madrid_housing_model"
            )
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance_df = pd.DataFrame({
                    'feature': model.feature_name_,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance to file
                importance_path = "models/feature_importance.csv"
                feature_importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                print("Feature importance logged to MLflow")
            
            print(f"{run_type.capitalize()} logged to MLflow. Run ID: {run.info.run_id}")
            print("To view results, run: python -m mlflow ui --backend-store-uri ./mlruns --port 5000")
            return run.info.run_id
    
    def _get_data_version_info(self) -> Dict[str, str]:
        """Get data version information for MLflow logging.
        
        Returns:
            Dictionary with data version information
        """
        data_info = {}
        
        # Check if preprocessed data exists
        preprocessed_path = "data/preprocessed_houses_Madrid.csv"
        if os.path.exists(preprocessed_path):
            data_info["data_source"] = "preprocessed_houses_Madrid.csv"
            data_info["data_type"] = "preprocessed"
        else:
            data_info["data_source"] = "houses_Madrid.csv"
            data_info["data_type"] = "raw"
        
        return data_info
