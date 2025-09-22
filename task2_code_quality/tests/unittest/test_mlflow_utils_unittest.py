"""
Unit tests for mlflow_utils module.
"""

import unittest
import unittest.mock as mock
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.mlflow_utils import MLflowLogger


class TestMLflowLoggerUnittest(unittest.TestCase):
    """Unit tests for MLflowLogger class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mlflow_logger = MLflowLogger()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'sq_mt_built': [100, 150, 200, 250, 300],
            'n_rooms': [2, 3, 4, 5, 6],
            'n_bathrooms': [1, 2, 2, 3, 3],
            'buy_price': [200000, 300000, 400000, 500000, 600000]
        })
        
        # Create sample metrics
        self.sample_metrics = {
            'train_rmse': 0.5,
            'train_mae': 0.3,
            'train_r2': 0.8,
            'val_rmse': 0.6,
            'val_mae': 0.4,
            'val_r2': 0.7,
            'test_rmse': 0.7,
            'test_mae': 0.5,
            'test_r2': 0.6
        }
        
        # Create sample model
        self.sample_model = Mock()
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_initialization(self):
        """Test MLflowLogger initialization."""
        self.assertIsNotNone(self.mlflow_logger)
        self.assertIsNotNone(self.mlflow_logger.file_manager)
    
    def test_log_training_run_basic(self):
        """Test basic training run logging."""
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.lightgbm.log_model') as mock_log_model, \
             patch('mlflow.end_run') as mock_end_run:
            
            mock_run = Mock()
            mock_run.info.run_id = 'test_run_id'
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            result = self.mlflow_logger.log_training_run(
                self.sample_model, self.sample_metrics, "test_run"
            )
            
            # Should return run ID
            self.assertEqual(result, 'test_run_id')
            
            # Should log metrics
            mock_log_metrics.assert_called_once_with(self.sample_metrics)
            mock_log_model.assert_called_once()
    
    def test_log_training_run_with_preprocessor(self):
        """Test training run logging with preprocessor."""
        mock_preprocessor = Mock()
        
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.lightgbm.log_model') as mock_log_model, \
             patch('mlflow.end_run') as mock_end_run:
            
            mock_run = Mock()
            mock_run.info.run_id = 'test_run_id'
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            result = self.mlflow_logger.log_training_run(
                self.sample_model, self.sample_metrics, "test_run", preprocessor=mock_preprocessor
            )
            
            # Should return run ID
            self.assertEqual(result, 'test_run_id')
            
            # Should log model with preprocessor
            mock_log_model.assert_called_once()
    
    def test_log_training_run_with_error(self):
        """Test training run logging with error."""
        with patch('mlflow.start_run', side_effect=Exception("MLflow error")):
            result = self.mlflow_logger.log_training_run(
                self.sample_model, self.sample_metrics, "test_run"
            )
            
            # Should return None on error
            self.assertIsNone(result)
    
    def test_log_training_run_without_metrics(self):
        """Test training run logging without metrics."""
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.lightgbm.log_model') as mock_log_model, \
             patch('mlflow.end_run') as mock_end_run:
            
            mock_run = Mock()
            mock_run.info.run_id = 'test_run_id'
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            result = self.mlflow_logger.log_training_run(
                self.sample_model, None, "test_run"
            )
            
            # Should return run ID
            self.assertEqual(result, 'test_run_id')
            
            # Should not log metrics when None
            mock_log_metrics.assert_not_called()
            mock_log_model.assert_called_once()
    
    def test_log_training_run_with_empty_metrics(self):
        """Test training run logging with empty metrics."""
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.lightgbm.log_model') as mock_log_model, \
             patch('mlflow.end_run') as mock_end_run:
            
            mock_run = Mock()
            mock_run.info.run_id = 'test_run_id'
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            result = self.mlflow_logger.log_training_run(
                self.sample_model, {}, "test_run"
            )
            
            # Should return run ID
            self.assertEqual(result, 'test_run_id')
            
            # Should log empty metrics
            mock_log_metrics.assert_called_once_with({})
            mock_log_model.assert_called_once()
    
    def test_log_training_run_with_custom_run_name(self):
        """Test training run logging with custom run name."""
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.lightgbm.log_model') as mock_log_model, \
             patch('mlflow.end_run') as mock_end_run:
            
            mock_run = Mock()
            mock_run.info.run_id = 'custom_run_id'
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            result = self.mlflow_logger.log_training_run(
                self.sample_model, self.sample_metrics, "custom_run_name"
            )
            
            # Should return run ID
            self.assertEqual(result, 'custom_run_id')
            
            # Should start run with custom name
            mock_start_run.assert_called_once_with(run_name="custom_run_name")
    
    def test_log_training_run_with_mlflow_setup_error(self):
        """Test training run logging when MLflow setup fails."""
        with patch.object(self.mlflow_logger, '_setup_mlflow', side_effect=Exception("Setup error")):
            result = self.mlflow_logger.log_training_run(
                self.sample_model, self.sample_metrics, "test_run"
            )
            
            # Should return None on setup error
            self.assertIsNone(result)
    
    def test_log_training_run_with_model_logging_error(self):
        """Test training run logging when model logging fails."""
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.lightgbm.log_model', side_effect=Exception("Model logging error")), \
             patch('mlflow.end_run') as mock_end_run:
            
            mock_run = Mock()
            mock_run.info.run_id = 'test_run_id'
            mock_start_run.return_value.__enter__.return_value = mock_run
            
            result = self.mlflow_logger.log_training_run(
                self.sample_model, self.sample_metrics, "test_run"
            )
            
            # Should return None on model logging error
            self.assertIsNone(result)
    
    def test_get_data_version_info(self):
        """Test getting data version info."""
        with patch.object(self.mlflow_logger.file_manager, 'get_file_info') as mock_get_file_info:
            mock_get_file_info.return_value = {
                'file_path': 'data/houses_Madrid.csv',
                'file_size': 1024,
                'last_modified': '2024-01-01 12:00:00'
            }
            
            result = self.mlflow_logger._get_data_version_info()
            
            # Should return data version info
            self.assertIsInstance(result, dict)
            self.assertIn('data_file_path', result)
            self.assertIn('data_file_size', result)
            self.assertIn('data_last_modified', result)
    
    def test_get_data_version_info_error(self):
        """Test getting data version info with error."""
        with patch.object(self.mlflow_logger.file_manager, 'get_file_info', side_effect=Exception("File error")):
            result = self.mlflow_logger._get_data_version_info()
            
            # Should return empty dict on error
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 0)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestMLflowLoggerUnittest))
    
    # Run the tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)