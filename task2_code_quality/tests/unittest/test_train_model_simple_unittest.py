"""
Simple unit tests for train_model.py using unittest framework.

This module provides basic unit tests for the refactored MadridHousingTrainer class
using the standard unittest framework.
"""

import unittest
import unittest.mock as mock
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from train_model import MadridHousingTrainer


class TestMadridHousingTrainerSimple(unittest.TestCase):
    """Simple test cases for MadridHousingTrainer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.trainer = MadridHousingTrainer("nonexistent.yaml")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'sq_mt_built': np.random.uniform(50, 200, n_samples),
            'n_rooms': np.random.randint(1, 6, n_samples),
            'n_bathrooms': np.random.randint(1, 4, n_samples),
            'is_new_development': np.random.choice([True, False], n_samples),
            'has_ac': np.random.choice([True, False], n_samples),
            'has_fitted_wardrobes': np.random.choice([True, False], n_samples),
            'has_lift': np.random.choice([1.0, 0.0], n_samples),
            'is_exterior': np.random.choice([1.0, 0.0], n_samples),
            'has_pool': np.random.choice([True, False], n_samples),
            'has_terrace': np.random.choice([True, False], n_samples),
            'has_balcony': np.random.choice([True, False], n_samples),
            'has_storage_room': np.random.choice([True, False], n_samples),
            'is_accessible': np.random.choice([True, False], n_samples),
            'has_green_zones': np.random.choice([True, False], n_samples),
            'has_parking': np.random.choice([True, False], n_samples),
        }
        
        # Add one-hot encoded features
        for house_type in ['HouseType_1_Piso', 'HouseType_2_Casa_o_chalet']:
            data[f'house_type_id_{house_type}'] = np.random.choice([True, False], n_samples)
        
        for district in ['District_1_Arganzuela', 'District_2_Barajas']:
            data[f'district_id_{district}'] = np.random.choice([True, False], n_samples)
        
        data['buy_price'] = data['sq_mt_built'] * 1000 + np.random.normal(0, 10000, n_samples)
        
        self.sample_data = pd.DataFrame(data)
        
        # Create train/val/test splits
        X = self.sample_data.drop('buy_price', axis=1)
        y = self.sample_data['buy_price']
        
        # Simple split for testing
        split_idx = int(len(X) * 0.6)
        val_idx = int(len(X) * 0.8)
        
        self.X_train = X[:split_idx]
        self.X_val = X[split_idx:val_idx]
        self.X_test = X[val_idx:]
        self.y_train = y[:split_idx]
        self.y_val = y[split_idx:val_idx]
        self.y_test = y[val_idx:]
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_trainer_initialization(self):
        """Test that trainer initializes correctly."""
        self.assertIsNotNone(self.trainer)
        self.assertIsNotNone(self.trainer.config)
        self.assertIsNotNone(self.trainer.data_manager)
        self.assertIsNotNone(self.trainer.metrics_calculator)
        self.assertIsNotNone(self.trainer.mlflow_logger)
        self.assertIsNotNone(self.trainer.evaluator)
        self.assertIsNotNone(self.trainer.versioning_manager)
    
    def test_prepare_data_method(self):
        """Test the prepare_data method."""
        # Mock the data manager methods
        with patch.object(self.trainer.data_manager, 'load_preprocessed_data', return_value=self.sample_data), \
             patch.object(self.trainer.data_manager, 'prepare_data_splits', return_value=(
                 self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
             )):
            
            result = self.trainer.prepare_data()
            
            # Should return 6 values
            self.assertEqual(len(result), 6)
            X_train, X_val, X_test, y_train, y_val, y_test = result
            
            # Verify data types
            self.assertIsInstance(X_train, pd.DataFrame)
            self.assertIsInstance(X_val, pd.DataFrame)
            self.assertIsInstance(X_test, pd.DataFrame)
            self.assertIsInstance(y_train, pd.Series)
            self.assertIsInstance(y_val, pd.Series)
            self.assertIsInstance(y_test, pd.Series)
    
    def test_train_model_method(self):
        """Test the train_model method."""
        # Mock LightGBM
        with patch('train_model.lgb.LGBMRegressor') as mock_lgb:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_lgb.return_value = mock_model
            
            result = self.trainer.train_model(self.X_train, self.y_train, self.X_val, self.y_val)
            
            # Should return the model
            self.assertEqual(result, mock_model)
            self.assertEqual(self.trainer.model, mock_model)
            
            # Verify model was fitted
            mock_model.fit.assert_called_once()
    
    def test_run_training_pipeline_basic(self):
        """Test basic training pipeline functionality."""
        # Mock all the utility methods
        with patch.object(self.trainer, 'prepare_data', return_value=(
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        )), \
        patch.object(self.trainer, 'train_model', return_value=Mock()), \
        patch.object(self.trainer.metrics_calculator, 'calculate_metrics_with_model', return_value={
            'train_rmse': 0.5, 'train_mae': 0.3, 'train_r2': 0.8
        }), \
        patch.object(self.trainer.mlflow_logger, 'log_training_run', return_value="test_run_id"), \
        patch.object(self.trainer.versioning_manager, 'save_model_with_versioning'):
            
            result = self.trainer.run_training_pipeline(run_name="test_run")
            
            # Should return a dictionary with results
            self.assertIsInstance(result, dict)
            self.assertIn('run_id', result)
            self.assertIn('model', result)
            self.assertIn('metrics', result)
    
    def test_config_loading(self):
        """Test that config is loaded correctly."""
        # Test with non-existent config (should use defaults)
        self.assertIsNotNone(self.trainer.config)
        self.assertIsInstance(self.trainer.config, dict)
    
    def test_utility_managers_initialization(self):
        """Test that all utility managers are properly initialized."""
        # Check that all utility managers are initialized
        self.assertIsNotNone(self.trainer.data_manager)
        self.assertIsNotNone(self.trainer.metrics_calculator)
        self.assertIsNotNone(self.trainer.mlflow_logger)
        self.assertIsNotNone(self.trainer.evaluator)
        self.assertIsNotNone(self.trainer.versioning_manager)
        
        # Check that they have the expected types
        from utils.data_utils import DataManager
        from utils.metrics_utils import MetricsCalculator
        from utils.mlflow_utils import MLflowLogger
        from utils.evaluation_utils import ModelEvaluator
        from utils.model_versioning_utils import ModelVersioningManager
        
        self.assertIsInstance(self.trainer.data_manager, DataManager)
        self.assertIsInstance(self.trainer.metrics_calculator, MetricsCalculator)
        self.assertIsInstance(self.trainer.mlflow_logger, MLflowLogger)
        self.assertIsInstance(self.trainer.evaluator, ModelEvaluator)
        self.assertIsInstance(self.trainer.versioning_manager, ModelVersioningManager)
    
    def test_run_multiple_experiments_basic(self):
        """Test multiple experiments functionality."""
        # Mock all the utility methods
        with patch.object(self.trainer, 'prepare_data', return_value=(
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        )), \
        patch.object(self.trainer, 'train_model', return_value=Mock()), \
        patch.object(self.trainer.metrics_calculator, 'calculate_metrics_with_model', return_value={
            'val_rmse': 0.5, 'val_mae': 0.3, 'val_r2': 0.8
        }), \
        patch.object(self.trainer.mlflow_logger, 'log_training_run', return_value="test_run_id"), \
        patch.object(self.trainer.versioning_manager, 'save_experiment_model', return_value=("timestamp", "path")), \
        patch.object(self.trainer.versioning_manager, 'find_best_model_from_experiments', return_value=(Mock(), Mock(), {}, "best_exp", 0.4)), \
        patch.object(self.trainer.versioning_manager, 'save_model_with_versioning'):
            
            # Mock config with experiments
            mock_config = {
                'experiments': [
                    {
                        'run_name': 'test_exp_1',
                        'model': {'n_estimators': 10, 'random_state': 42},
                        'training': {'early_stopping_rounds': 5}
                    },
                    {
                        'run_name': 'test_exp_2',
                        'model': {'n_estimators': 20, 'random_state': 42},
                        'training': {'early_stopping_rounds': 5}
                    }
                ]
            }
            
            with patch.object(self.trainer, 'config', mock_config):
                result = self.trainer.run_multiple_experiments()
                
                # Should return a dictionary with results
                self.assertIsInstance(result, dict)
                self.assertIn('test_exp_1', result)
                self.assertIn('test_exp_2', result)
    
    def test_run_grid_search_basic(self):
        """Test grid search functionality."""
        # Mock all the utility methods
        with patch.object(self.trainer, 'prepare_data', return_value=(
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        )), \
        patch('train_model.GridSearchCV') as mock_grid_search, \
        patch.object(self.trainer.metrics_calculator, 'calculate_metrics_with_model', return_value={
            'train_rmse': 0.5, 'train_mae': 0.3, 'train_r2': 0.8,
            'val_rmse': 0.6, 'val_mae': 0.4, 'val_r2': 0.7,
            'test_rmse': 0.7, 'test_mae': 0.5, 'test_r2': 0.6
        }), \
        patch.object(self.trainer.mlflow_logger, 'log_training_run', return_value="test_run_id"), \
        patch.object(self.trainer.versioning_manager, 'save_model_with_versioning'):
            
            # Setup mock grid search
            mock_grid_search_instance = Mock()
            mock_grid_search_instance.fit.return_value = mock_grid_search_instance
            mock_grid_search_instance.best_estimator_ = Mock()
            mock_grid_search_instance.best_params_ = {'n_estimators': 100}
            mock_grid_search_instance.best_score_ = 0.85
            mock_grid_search.return_value = mock_grid_search_instance
            
            result = self.trainer.run_grid_search()
            
            # Should return a dictionary with results
            self.assertIsInstance(result, dict)
            self.assertIn('best_params', result)
            self.assertIn('best_score', result)
            self.assertIn('model', result)
    
    def test_train_model_with_validation_data(self):
        """Test model training with validation data."""
        # Mock LightGBM
        with patch('train_model.lgb.LGBMRegressor') as mock_lgb:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_lgb.return_value = mock_model
            
            result = self.trainer.train_model(self.X_train, self.y_train, self.X_val, self.y_val)
            
            # Should return the model
            self.assertEqual(result, mock_model)
            self.assertEqual(self.trainer.model, mock_model)
            
            # Verify model was fitted with validation data
            mock_model.fit.assert_called_once()
            call_args = mock_model.fit.call_args
            self.assertIn('eval_set', call_args.kwargs)
    
    def test_train_model_without_validation_data(self):
        """Test model training without validation data."""
        # Mock LightGBM
        with patch('train_model.lgb.LGBMRegressor') as mock_lgb:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_lgb.return_value = mock_model
            
            # Test with None validation data
            result = self.trainer.train_model(self.X_train, self.y_train, None, None)
            
            # Should return the model
            self.assertEqual(result, mock_model)
            self.assertEqual(self.trainer.model, mock_model)
            
            # Verify model was fitted without validation data
            mock_model.fit.assert_called_once()
            call_args = mock_model.fit.call_args
            self.assertNotIn('eval_set', call_args.kwargs)
    
    def test_config_loading_with_invalid_path(self):
        """Test config loading with invalid path."""
        # Test with non-existent config (should use defaults)
        trainer = MadridHousingTrainer("nonexistent_config.yaml")
        self.assertIsNotNone(trainer.config)
        self.assertIsInstance(trainer.config, dict)
    
    def test_prepare_data_with_missing_data(self):
        """Test prepare_data when data is missing."""
        # Mock data manager to raise an exception
        with patch.object(self.trainer.data_manager, 'prepare_data_if_needed', side_effect=FileNotFoundError("Data not found")):
            with self.assertRaises(FileNotFoundError):
                self.trainer.prepare_data()
    
    def test_train_model_with_invalid_data(self):
        """Test train_model with invalid data."""
        # Test with empty data
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        
        with self.assertRaises((ValueError, IndexError)):
            self.trainer.train_model(empty_df, empty_series, empty_df, empty_series)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestMadridHousingTrainerSimple))
    
    # Run the tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Unit Tests Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with error code if there were failures or errors
    sys.exit(len(result.failures) + len(result.errors))
