"""
Training module for Madrid Housing Market price prediction.

This module provides training functionality with MLflow experiment tracking,
hyperparameter management, and model evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
import yaml
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
from datetime import datetime
import warnings

from data_loader import load_data, split_data
from preprocessing import MadridHousingPreprocessor

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MadridHousingTrainer:
    """Trainer class for Madrid Housing Market price prediction."""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """Initialize trainer with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Use default configuration if file not found
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'data': {
                'source_path': 'houses_Madrid.csv',
                'target_column': 'buy_price',
                'test_size': 0.2,
                'random_state': 42
            },
            'mlflow': {
                'experiment_name': 'madrid_housing_experiments',
                'tracking_uri': 'sqlite:///mlruns.db'
            },
            'model': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            },
            'training': {
                'early_stopping_rounds': 10,
                'eval_metric': 'rmse',
                'verbose_eval': 100
            }
        }
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare data for training."""
        logger.info("=" * 60)
        logger.info("PREPARING DATA")
        logger.info("=" * 60)
        
        # Load data
        data_config = self.config['data']
        df = load_data(data_config['source_path'])
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            df, 
            data_config['target_column'], 
            test_size=data_config['test_size'],
            random_state=data_config['random_state']
        )
        
        # Initialize preprocessor
        self.preprocessor = MadridHousingPreprocessor()
        
        # Fit and transform training data
        X_train_transformed, y_train_transformed = self.preprocessor.fit_transform(X_train, y_train)
        
        # Transform test data
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Store feature names
        self.feature_names = self.preprocessor.get_feature_names()
        
        logger.info(f"Training data shape: {X_train_transformed.shape}")
        logger.info(f"Test data shape: {X_test_transformed.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        
        return X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray = None, y_val: np.ndarray = None) -> lgb.LGBMRegressor:
        """Train LightGBM model."""
        logger.info("=" * 60)
        logger.info("TRAINING MODEL")
        logger.info("=" * 60)
        
        # Get model parameters
        model_params = self.config['model']
        training_params = self.config['training']
        
        # Initialize model
        self.model = lgb.LGBMRegressor(**model_params)
        
        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Train model
        logger.info("Training LightGBM model...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[
                lgb.early_stopping(training_params['early_stopping_rounds']),
                lgb.log_evaluation(training_params['verbose_eval'])
            ]
        )
        
        logger.info("Model training completed")
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info("=" * 60)
        logger.info("EVALUATING MODEL")
        logger.info("=" * 60)
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"Test RMSE: {rmse:.2f}")
        logger.info(f"Test MAE: {mae:.2f}")
        logger.info(f"Test R²: {r2:.3f}")
        logger.info(f"Test MAPE: {mape:.2f}%")
        
        return metrics
    
    def log_to_mlflow(self, metrics: Dict[str, float], run_name: str = None) -> str:
        """Log experiment to MLflow."""
        logger.info("=" * 60)
        logger.info("LOGGING TO MLFLOW")
        logger.info("=" * 60)
        
        # Set MLflow tracking URI
        mlflow_config = self.config['mlflow']
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        
        # Set experiment
        mlflow.set_experiment(mlflow_config['experiment_name'])
        
        # Start run
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(self.config['model'])
            mlflow.log_params(self.config['training'])
            mlflow.log_params(self.config['data'])
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.lightgbm.log_model(
                lgb_model=self.model,
                artifact_path="model",
                registered_model_name="madrid_housing_model"
            )
            
            # Log preprocessing pipeline
            if self.preprocessor is not None:
                preprocessor_path = "preprocessor.pkl"
                self.preprocessor.save_pipeline(preprocessor_path)
                mlflow.log_artifact(preprocessor_path)
                Path(preprocessor_path).unlink()  # Clean up
            
            # Log feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                feature_importance.to_csv('feature_importance.csv', index=False)
                mlflow.log_artifact('feature_importance.csv')
                Path('feature_importance.csv').unlink()  # Clean up
            
            logger.info(f"Experiment logged to MLflow. Run ID: {run.info.run_id}")
            return run.info.run_id
    
    def save_model(self, model_path: str = "models/madrid_housing_model.pkl") -> None:
        """Save trained model and preprocessor."""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Create models directory
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save preprocessor
        preprocessor_path = str(Path(model_path).parent / "preprocessor.pkl")
        self.preprocessor.save_pipeline(preprocessor_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def run_training_pipeline(self, run_name: str = None) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting Madrid Housing Market Training Pipeline")
        logger.info("=" * 80)
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data()
            
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Train model
            self.train_model(X_train_split, y_train_split, X_val_split, y_val_split)
            
            # Evaluate model
            metrics = self.evaluate_model(X_test, y_test)
            
            # Log to MLflow
            run_id = self.log_to_mlflow(metrics, run_name)
            
            # Save model
            self.save_model()
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'run_id': run_id,
                'metrics': metrics,
                'model': self.model,
                'preprocessor': self.preprocessor
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def run_multiple_experiments(self) -> Dict[str, Any]:
        """Run multiple experiments with different configurations."""
        logger.info("Starting Multiple Experiments Training Pipeline")
        logger.info("=" * 80)
        
        # Prepare data once
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        results = {}
        experiments = self.config.get('experiments', [])
        
        if not experiments:
            logger.warning("No experiments configured. Running single experiment.")
            return self.run_training_pipeline()
        
        for i, exp_config in enumerate(experiments):
            logger.info(f"Running experiment {i+1}/{len(experiments)}: {exp_config['run_name']}")
            
            # Update config for this experiment
            original_model_config = self.config.get('model', {})
            original_training_config = self.config.get('training', {})
            
            self.config['model'] = exp_config['model']
            self.config['training'] = exp_config['training']
            
            try:
                # Train model with this configuration
                self.train_model(X_train_split, y_train_split, X_val_split, y_val_split)
                
                # Evaluate model
                metrics = self.evaluate_model(X_test, y_test)
                
                # Log to MLflow
                run_id = self.log_to_mlflow(metrics, exp_config['run_name'])
                
                # Store results
                results[exp_config['run_name']] = {
                    'run_id': run_id,
                    'metrics': metrics,
                    'model': self.model,
                    'preprocessor': self.preprocessor,
                    'description': exp_config.get('description', '')
                }
                
                logger.info(f"Experiment {exp_config['run_name']} completed successfully!")
                
            except Exception as e:
                logger.error(f"Experiment {exp_config['run_name']} failed: {e}")
                results[exp_config['run_name']] = {'error': str(e)}
            
            finally:
                # Restore original config
                self.config['model'] = original_model_config
                self.config['training'] = original_training_config
        
        # Save the best model (last successful one)
        if self.model is not None:
            self.save_model()
        
        logger.info(f"All {len(experiments)} experiments completed!")
        return results


def main():
    """Main function to run training."""
    trainer = MadridHousingTrainer()
    
    # Check if multiple experiments are configured
    if 'experiments' in trainer.config and len(trainer.config['experiments']) > 1:
        # Run multiple experiments
        results = trainer.run_multiple_experiments()
        
        print(f"Multiple experiments completed!")
        for exp_name, result in results.items():
            if 'error' not in result:
                print(f"  {exp_name}: RMSE={result['metrics']['rmse']:.2f}, R²={result['metrics']['r2']:.3f}")
            else:
                print(f"  {exp_name}: FAILED - {result['error']}")
    else:
        # Run single training pipeline
        results = trainer.run_training_pipeline(run_name=f"madrid_housing_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        print(f"Training completed! Run ID: {results['run_id']}")
        print(f"Test RMSE: {results['metrics']['rmse']:.2f}")
        print(f"Test R²: {results['metrics']['r2']:.3f}")


if __name__ == "__main__":
    main()
