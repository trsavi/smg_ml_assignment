#!/usr/bin/env python3
"""
Model evaluation script for Madrid Housing Market pipeline.

This script evaluates existing trained models and shows performance metrics.
It loads a trained model and evaluates it on test data.
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train import MadridHousingTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trained_model(model_path: str = "models/madrid_housing_model.pkl"):
    """Load a trained model from file."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train a model first using train.py")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return model

def evaluate_model_on_test_data(model, X_test, y_test):
    """Evaluate model performance on test data."""
    logger.info("Evaluating model on test data...")
    
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
    
    logger.info(f"Test RMSE: {rmse:.2f}")
    logger.info(f"Test MAE: {mae:.2f}")
    logger.info(f"Test R²: {r2:.3f}")
    
    return metrics

def main():
    """Evaluate the trained model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Madrid Housing Market model')
    parser.add_argument('--model-path', type=str, default='models/madrid_housing_model.pkl',
                       help='Path to the trained model file (default: models/madrid_housing_model.pkl)')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom name for the evaluation run')
    
    args = parser.parse_args()
    
    print("Evaluating trained model...")
    
    try:
        # Initialize trainer to get data preparation capabilities
        trainer = MadridHousingTrainer()
        
        # Check if preprocessed data exists, if not prepare it
        if not trainer._check_preprocessed_data():
            print("Preprocessed data not found. Preparing data first...")
            trainer._prepare_data_if_needed()
        
        # Load the trained model
        model = load_trained_model(args.model_path)
        
        # Prepare test data
        print("Preparing test data...")
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()
        
        # Evaluate model
        metrics = evaluate_model_on_test_data(model, X_test, y_test)
        
        # Log evaluation to MLflow
        run_name = args.run_name or f"evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        run_id = trainer.log_to_mlflow(model, metrics=metrics, run_name=run_name, run_type='evaluation')
        
        print(f"\nEvaluation completed! Run ID: {run_id}")
        print(f"Test RMSE: {metrics['rmse']:.2f}")
        print(f"Test MAE: {metrics['mae']:.2f}")
        print(f"Test R²: {metrics['r2']:.3f}")
        print("\nTo view MLflow UI: mlflow ui --backend-store-uri ./mlruns --port 5000")
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
