"""
Training module for Madrid Housing Market price prediction.

This module provides core training functionality with clean separation of concerns.
Helper methods have been moved to utility modules.
"""

# Standard library imports
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

# Third-party imports
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Local imports
from utils.file_manager import FileManager
from utils.data_utils import DataManager
from utils.metrics_utils import MetricsCalculator
from utils.model_versioning_utils import ModelVersioningManager
from utils.mlflow_utils import MLflowLogger
from utils.evaluation_utils import ModelEvaluator

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MadridHousingTrainer:
    """Trainer class for Madrid Housing Market price prediction."""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """
        Initialize trainer with configuration and utility managers.
        
        Args:
            config_path (str): Path to training configuration file.
            
        Returns:
            None: Initializes the trainer instance.
            
        Example:
            >>> trainer = MadridHousingTrainer("configs/my_training_config.yaml")
        """
        self.file_manager = FileManager()
        self.data_manager = DataManager(self.file_manager)
        self.metrics_calculator = MetricsCalculator(self.file_manager)
        self.versioning_manager = ModelVersioningManager(self.file_manager)
        self.mlflow_logger = MLflowLogger(self.file_manager)
        self.evaluator = ModelEvaluator()
        
        self.config_path = Path(config_path)
        self.config = self.file_manager.load_training_config(config_path)
        self.preprocessor = None
        self.model = None
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        This method ensures preprocessed data exists, loads it, and splits it
        into training, validation, and test sets.
        
        Args:
            None: Uses internal data manager.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: 
                (X_train, X_val, X_test, y_train, y_val, y_test)
                
        Example:
            >>> X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()
        """
        logger.info("=" * 60)
        logger.info("PREPARING DATA")
        logger.info("=" * 60)
        
        # Ensure preprocessed data exists
        self.data_manager.prepare_data_if_needed()
        
        # Load preprocessed data
        data = self.data_manager.load_preprocessed_data()
        
        # Prepare data splits
        return self.data_manager.prepare_data_splits(data)
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
        """
        Train the LightGBM model.
        
        This method creates and trains a LightGBM regressor using the provided
        training and validation data with early stopping.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation targets.
            
        Returns:
            lgb.LGBMRegressor: Trained LightGBM model.
            
        Example:
            >>> model = trainer.train_model(X_train, y_train, X_val, y_val)
        """
        logger.info("=" * 60)
        logger.info("TRAINING MODEL")
        logger.info("=" * 60)
        
        # Get model parameters from config
        model_params = self.config.get('model', {})
        training_params = self.config.get('training', {})
        
        logger.info("Training LightGBM model...")
        
        # Create and train model
        self.model = lgb.LGBMRegressor(**model_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(training_params.get('early_stopping_rounds', 10))],
            eval_metric=training_params.get('eval_metric', 'rmse')
        )
        
        logger.info("Model training completed")
        return self.model
    
    def run_training_pipeline(self, run_name: str = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline (training only).
        
        This method executes the full training pipeline including data preparation,
        model training, metrics calculation, MLflow logging, and model saving.
        
        Args:
            run_name (str, optional): Name for the training run.
            
        Returns:
            Dict[str, Any]: Dictionary with training results including run_id, model, and metrics.
            
        Raises:
            Exception: If training pipeline fails.
            
        Example:
            >>> results = trainer.run_training_pipeline("experiment_1")
            >>> print(f"Run ID: {results['run_id']}")
        """
        logger.info("Starting Madrid Housing Market Training Pipeline")
        logger.info("=" * 80)
        
        try:
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
            
            # Train model
            self.train_model(X_train, y_train, X_val, y_val)
            
            # Calculate comprehensive metrics
            logger.info("=" * 60)
            logger.info("CALCULATING COMPREHENSIVE METRICS")
            logger.info("=" * 60)
            
            all_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                self.model, X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Log to MLflow
            logger.info("=" * 60)
            logger.info("LOGGING TRAINING TO MLFLOW")
            logger.info("=" * 60)
            
            run_id = self.mlflow_logger.log_training_run(
                self.model, all_metrics=all_metrics, run_name=run_name, run_type='training'
            )
            
            # Save model with versioning
            self.versioning_manager.save_model_with_versioning(self.model)
            
            logger.info("Training pipeline completed successfully!")
            logger.info("Note: Use evaluate_model.py script to evaluate the trained model.")
            
            return {
                'run_id': run_id,
                'model': self.model,
                'preprocessor': self.preprocessor,
                'metrics': all_metrics,
                'data_splits': {
                    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                    'y_train': y_train, 'y_val': y_val, 'y_test': y_test
                }
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def run_multiple_experiments(self) -> Dict[str, Any]:
        """Run multiple experiments with different configurations (training only).
        
        Returns:
            Dictionary with experiment results
        """
        logger.info("Starting Multiple Experiments Training Pipeline")
        logger.info("=" * 80)
        
        # Prepare data once
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        results = {}
        experiments = self.config.get('experiments', [])
        
        if not experiments:
            logger.warning("No experiments configured. Running single experiment.")
            return self.run_training_pipeline()
        
        # Run each experiment
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
                
                # Calculate comprehensive metrics
                all_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                    self.model, X_train, y_train, X_val, y_val, X_test, y_test
                )
                
                # Log to MLflow
                run_id = self.mlflow_logger.log_training_run(
                    self.model, all_metrics=all_metrics, run_name=exp_config['run_name'], run_type='training'
                )
                
                # Save experiment model with versioning
                exp_model_path = self.versioning_manager.save_experiment_model(
                    self.model, exp_config['run_name'], 
                    {'val_rmse': all_metrics['val']['val_rmse'], 
                     'val_mae': all_metrics['val']['val_mae'], 
                     'val_r2': all_metrics['val']['val_r2']}
                )
                
                # Store results
                results[exp_config['run_name']] = {
                    'run_id': run_id,
                    'model': self.model,
                    'preprocessor': self.preprocessor,
                    'metrics': all_metrics,
                    'description': exp_config.get('description', ''),
                    'versioned_model_path': exp_model_path,
                    'val_rmse': all_metrics['val']['val_rmse']
                }
                
                logger.info(f"Experiment {exp_config['run_name']} training completed successfully!")
                
            except Exception as e:
                logger.error(f"Experiment {exp_config['run_name']} failed: {e}")
                results[exp_config['run_name']] = {'error': str(e)}
            
            finally:
                # Restore original config
                self.config['model'] = original_model_config
                self.config['training'] = original_training_config
        
        # Find and save the best model
        best_model, best_experiment_name, best_val_rmse = self.versioning_manager.find_best_model_from_experiments(results)
        
        if best_model is not None:
            logger.info(f"Saving best model from experiment '{best_experiment_name}' to models/ directory")
            self.model = best_model
            self.versioning_manager.save_best_model_from_experiments(
                best_model, best_experiment_name, best_val_rmse
            )
        else:
            logger.warning("No successful experiments completed. No best model to save.")
        
        logger.info(f"All {len(experiments)} experiments training completed!")
        logger.info(f"Best model from experiment '{best_experiment_name}' saved to models/madrid_housing_model.pkl")
        logger.info("Note: Use evaluate_model.py script to evaluate the trained models.")
        
        return results
    
    def run_grid_search(self) -> Dict[str, Any]:
        """Run grid search hyperparameter tuning using scikit-learn's GridSearchCV.
        
        Returns:
            Dictionary with grid search results
        """
        logger.info("Starting Grid Search Hyperparameter Tuning")
        logger.info("=" * 80)

        grid_config = self.config.get("grid_search", {})
        param_grid = grid_config.get("parameters", {})
        cv_folds = grid_config.get("cv_folds", 3)
        scoring = grid_config.get("scoring", "neg_root_mean_squared_error")

        if not param_grid:
            logger.error("No grid search parameters defined in config")
            raise ValueError("No grid search parameters defined")

        logger.info(f"Grid search parameters from config: {param_grid}")
        logger.info(f"Cross-validation folds: {cv_folds}")
        logger.info(f"Scoring metric: {scoring}")

        # Prepare train data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()

        # Initialize model with default parameters from config
        base_model = self.train_model(X_train, y_train, X_val, y_val)

        # Setup GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1,
        )

        logger.info("Running GridSearchCV...")
        grid_search.fit(X_train, y_train)

        # Extract best results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimator = grid_search.best_estimator_

        logger.info("=" * 60)
        logger.info("GRID SEARCH RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")

        # Save best model with versioning
        self.model = best_estimator
        
        # Calculate comprehensive metrics for the best model
        all_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            self.model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Log to MLflow with comprehensive metrics
        run_id = self.mlflow_logger.log_training_run(
            self.model, all_metrics=all_metrics, run_name="grid_search_best", run_type='training'
        )
        
        # Save with versioning
        self.versioning_manager.save_model_with_versioning(self.model)
        
        logger.info(f"Best model from grid search saved with versioning")
        logger.info(f"MLflow Run ID: {run_id}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": grid_search.cv_results_,
            "run_id": run_id,
            "metrics": all_metrics
        }
    
    def save_model(self, model_path: str = None) -> str:
        """Save the trained model.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if model_path is None:
            model_path = self.config.get('model_saving', {}).get('model_path', 'models/madrid_housing_model.pkl')
        
        return self.file_manager.save_model(self.model, model_path)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model to evaluate. Train a model first.")
        
        return self.evaluator.evaluate_model(self.model, X_test, y_test)
    
    def _check_preprocessed_data(self) -> bool:
        """Check if preprocessed data exists.
        
        Returns:
            True if preprocessed data exists, False otherwise
        """
        preprocessed_path = self.config.get('data', {}).get('preprocessed_path', 'data/preprocessed_houses_Madrid.csv')
        return Path(preprocessed_path).exists()
    
    def _load_preprocessed_data(self) -> pd.DataFrame:
        """Load preprocessed data.
        
        Returns:
            Preprocessed DataFrame
        """
        preprocessed_path = self.config.get('data', {}).get('preprocessed_path', 'data/preprocessed_houses_Madrid.csv')
        return pd.read_csv(preprocessed_path)
    
    def _get_data_version_info(self) -> Dict[str, Any]:
        """Get data version information.
        
        Returns:
            Dictionary with data version info
        """
        return self.file_manager.get_file_info('data/houses_Madrid.csv')


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
        # Run single training
        results = trainer.run_training_pipeline()
        print(f"Training completed! Run ID: {results['run_id']}")


if __name__ == '__main__':
    main()
