"""
Training module for Madrid Housing Market price prediction.

This module provides training functionality with hyperparameter management
and model evaluation.
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
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
from datetime import datetime
import warnings
import subprocess
import sys

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
                'experiment_name': 'housing_price_experiments',
                'tracking_uri': './mlruns'
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
    
    def _check_preprocessed_data(self) -> bool:
        """Check if preprocessed data file exists."""
        preprocessed_path = Path("data/preprocessed_houses_Madrid.csv")
        return preprocessed_path.exists()
    
    def _prepare_data_if_needed(self) -> None:
        """Call prepare_data script if preprocessed data doesn't exist."""
        if not self._check_preprocessed_data():
            logger.info("Preprocessed data not found. Running data preparation...")
            try:
                # Call prepare_data script with --store flag
                result = subprocess.run(
                    [sys.executable, "scripts/prepare_data.py", "--store"],
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info("Data preparation completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running data preparation: {e}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                raise RuntimeError("Failed to prepare data")
        else:
            logger.info("Using existing preprocessed data")
    
    def _load_preprocessed_data(self) -> pd.DataFrame:
        """Load preprocessed data from file."""
        preprocessed_path = Path("data/preprocessed_houses_Madrid.csv")
        if not preprocessed_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {preprocessed_path}")
        
        df = pd.read_csv(preprocessed_path)
        logger.info(f"Loaded preprocessed data: {df.shape}")
        return df
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Load and prepare data for training with train/validation/test splits."""
        logger.info("=" * 60)
        logger.info("PREPARING DATA")
        logger.info("=" * 60)
        
        # Check and prepare data if needed
        self._prepare_data_if_needed()
        
        # Load preprocessed data
        df_processed = self._load_preprocessed_data()
        
        # Get target column from config
        data_config = self.config['data']
        target_column = data_config['target_column']
        
        # Split into features and target
        if target_column not in df_processed.columns:
            raise ValueError(f"Target column '{target_column}' not found in preprocessed data")
        
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=data_config['test_size'],
            random_state=data_config['random_state']
        )
        
        # Second split: separate train and validation from remaining data
        val_size = data_config.get('val_size', 0.2)  # 20% of remaining data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            random_state=data_config['random_state']
        )
        
        # Feature names are now cleaned during preprocessing
        
        # Store feature names for later use
        self.feature_names = X_train.columns.tolist()
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame = None, y_val: pd.Series = None) -> lgb.LGBMRegressor:
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
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
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
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Test RMSE: {rmse:.2f}")
        logger.info(f"Test MAE: {mae:.2f}")
        logger.info(f"Test R²: {r2:.3f}")
        
        return metrics
    

    def _get_data_version_info(self) -> Dict[str, str]:
        """Get simple data versioning information."""
        data_info = {}
        
        # Simple data info
        preprocessed_path = Path("data/preprocessed_houses_Madrid.csv")
        if preprocessed_path.exists():
            data_info['data_version'] = 'preprocessed'
            data_info['data_rows'] = str(len(pd.read_csv(preprocessed_path)))
        else:
            data_info['data_version'] = 'unknown'
            data_info['data_rows'] = 'unknown'
        
        return data_info

    def log_to_mlflow(self, model, metrics: Dict[str, float] = None, run_name: str = None, 
                     run_type: str = 'training') -> str:
        """
        Unified MLflow logging method for training, evaluation, or any experiment.
        
        Args:
            model: The trained model to log
            metrics: Optional metrics to log (for evaluation runs)
            run_name: Optional custom run name
            run_type: Type of run ('training', 'evaluation', 'experiment')
        """
        logger.info("=" * 60)
        logger.info(f"LOGGING {run_type.upper()} TO MLFLOW")
        logger.info("=" * 60)
        
        # Set MLflow tracking URI (local file backend)
        mlflow_config = self.config['mlflow']
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        
        # Set experiment (creates mlruns/ directory automatically)
        mlflow.set_experiment(mlflow_config['experiment_name'])
        
        # Start run
        with mlflow.start_run(run_name=run_name) as run:
            # Log run type
            mlflow.log_param('run_type', run_type)
            
            # Log hyperparameters (for training runs)
            if run_type in ['training', 'experiment']:
                mlflow.log_params(self.config['model'])
                mlflow.log_params(self.config['training'])
                mlflow.log_params(self.config['data'])
                
                # Log simple data versioning info
                data_version_info = self._get_data_version_info()
                mlflow.log_params(data_version_info)
            
            # Log metrics (for evaluation runs or when provided)
            if metrics:
                mlflow.log_metrics(metrics)
            
            # Log the model artifact
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name="madrid_housing_model"
            )
            
            # Log feature importance as artifact
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance to temporary file
                feature_importance.to_csv('feature_importance.csv', index=False)
                mlflow.log_artifact('feature_importance.csv')
                
                # Clean up temporary file
                Path('feature_importance.csv').unlink()
                
                logger.info("Feature importance logged to MLflow")
            
            logger.info(f"{run_type.capitalize()} logged to MLflow. Run ID: {run.info.run_id}")
            logger.info("To view results, run: python -m mlflow ui --backend-store-uri ./mlruns --port 5000")
            return run.info.run_id
    
    def save_model(self, model_path: str = "models/madrid_housing_model.pkl") -> None:
        """Save trained model and preprocessor."""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Create models directory
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save preprocessor (if available)
        if self.preprocessor is not None:
            preprocessor_path = str(Path(model_path).parent / "preprocessor.pkl")
            self.preprocessor.save_pipeline(preprocessor_path)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
        else:
            logger.info("No preprocessor to save (preprocessing was done separately)")
        
        logger.info(f"Model saved to {model_path}")
    
    def run_training_pipeline(self, run_name: str = None) -> Dict[str, Any]:
        """Run the complete training pipeline (training only)."""
        logger.info("Starting Madrid Housing Market Training Pipeline")
        logger.info("=" * 80)
        
        try:
            # Prepare data (now includes train/val/test splits)
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
            
            # Train model with validation set
            self.train_model(X_train, y_train, X_val, y_val)
            
            # Evaluate model on test set to get metrics
            test_metrics = self.evaluate_model(X_test, y_test)
            
            # Log training to MLflow with metrics
            run_id = self.log_to_mlflow(self.model, metrics=test_metrics, run_name=run_name, run_type='training')
            
            # Save model
            self.save_model()
            
            logger.info("Training pipeline completed successfully!")
            logger.info("Note: Use evaluate_model.py script to evaluate the trained model.")
            
            return {
                'run_id': run_id,
                'model': self.model,
                'preprocessor': self.preprocessor,
                'metrics': test_metrics,
                'data_splits': {
                    'X_train': X_train,
                    'X_val': X_val,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test
                }
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def run_multiple_experiments(self) -> Dict[str, Any]:
        """Run multiple experiments with different configurations (training only)."""
        logger.info("Starting Multiple Experiments Training Pipeline")
        logger.info("=" * 80)
        
        # Prepare data once (now includes train/val/test splits)
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
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
                self.train_model(X_train, y_train, X_val, y_val)
                
                # Evaluate model on test set to get metrics
                test_metrics = self.evaluate_model(X_test, y_test)
                
                # Log training to MLflow with metrics
                run_id = self.log_to_mlflow(self.model, metrics=test_metrics, run_name=exp_config['run_name'], run_type='training')
                
                # Store results
                results[exp_config['run_name']] = {
                    'run_id': run_id,
                    'model': self.model,
                    'preprocessor': self.preprocessor,
                    'metrics': test_metrics,
                    'description': exp_config.get('description', '')
                }
                
                logger.info(f"Experiment {exp_config['run_name']} training completed successfully!")
                
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
        
        logger.info(f"All {len(experiments)} experiments training completed!")
        logger.info("Note: Use evaluate_model.py script to evaluate the trained models.")
        return results

    def run_grid_search(self) -> Dict[str, Any]:
        """Run grid search hyperparameter tuning."""
        logger.info("Starting Grid Search Hyperparameter Tuning")
        logger.info("=" * 80)
        
        # Check if grid search is enabled
        grid_config = self.config.get('grid_search', {})
        if not grid_config.get('enabled', False):
            logger.error("Grid search is not enabled in config. Set grid_search.enabled: true")
            raise ValueError("Grid search not enabled")
        
        param_grid = grid_config.get('parameters', {})
        if not param_grid:
            logger.error("No grid search parameters defined in config")
            raise ValueError("No grid search parameters defined")
        
        logger.info("Grid search parameters:")
        for param, values in param_grid.items():
            logger.info(f"  {param}: {values}")
        
        # Prepare data once
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        best_score = float('inf')
        best_params = None
        best_run_id = None
        results = []
        
        # Generate all parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        logger.info(f"Total parameter combinations: {total_combinations}")
        logger.info("Starting grid search...")
        logger.info("-" * 60)
        
        for i, combination in enumerate(product(*param_values), 1):
            params = dict(zip(param_names, combination))
            
            logger.info(f"Combination {i}/{total_combinations}: {params}")
            
            # Update trainer config with current parameters
            original_model_config = self.config['model'].copy()
            self.config['model'].update(params)
            
            try:
                # Train model with current parameters
                self.train_model(X_train, y_train, X_val, y_val)
                
                # Evaluate on validation set
                val_metrics = self.evaluate_model(X_val, y_val)
                val_rmse = val_metrics['rmse']
                
                # Log to MLflow
                run_name = f"grid_search_{i}_{'_'.join([f'{k}_{v}' for k, v in params.items()])}"
                run_id = self.log_to_mlflow(self.model, metrics=val_metrics, run_name=run_name, run_type='grid_search')
                
                result = {
                    'params': params,
                    'run_id': run_id,
                    'val_rmse': val_rmse,
                    'val_metrics': val_metrics
                }
                results.append(result)
                
                logger.info(f"  Validation RMSE: {val_rmse:.2f}, Run ID: {run_id}")
                
                # Check if this is the best so far
                if val_rmse < best_score:
                    best_score = val_rmse
                    best_params = params
                    best_run_id = run_id
                    logger.info(f"  *** New best score! ***")
                
            except Exception as e:
                logger.error(f"  Failed: {e}")
                results.append({
                    'params': params,
                    'error': str(e)
                })
            
            finally:
                # Restore original config
                self.config['model'] = original_model_config
            
            logger.info("")
        
        # Print grid search results
        logger.info("=" * 60)
        logger.info("GRID SEARCH RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation RMSE: {best_score:.2f}")
        logger.info(f"Best run ID: {best_run_id}")
        
        # Save best model
        if best_params:
            logger.info("Training final model with best parameters...")
            self.config['model'].update(best_params)
            self.train_model(X_train, y_train, X_val, y_val)
            self.save_model()
            logger.info("Best model saved to: models/madrid_housing_model.pkl")
        
        logger.info("Grid search completed!")
        logger.info("Note: Use evaluate_model.py script to evaluate the best model.")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_run_id': best_run_id,
            'all_results': results
        }


def main():
    """Main function to run training."""
    trainer = MadridHousingTrainer()
    
    # Check if multiple experiments are configured
    if 'experiments' in trainer.config and len(trainer.config['experiments']) > 1:
        # Run multiple experiments
        results = trainer.run_multiple_experiments()
        
        print(f"Multiple experiments training completed!")
        for exp_name, result in results.items():
            if 'error' not in result:
                print(f"  {exp_name}: Training completed - Run ID: {result['run_id']}")
            else:
                print(f"  {exp_name}: FAILED - {result['error']}")
    else:
        # Run single training pipeline
        results = trainer.run_training_pipeline(run_name=f"madrid_housing_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        print(f"Training completed! Run ID: {results['run_id']}")
        print("To evaluate the model, run: python scripts/evaluate_model.py")
        print("To view MLflow UI: mlflow ui --backend-store-uri ./mlruns --port 5000")


if __name__ == "__main__":
    main()
