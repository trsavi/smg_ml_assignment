"""
Simple integration tests for train_model.py using unittest framework.

This module provides basic integration tests for the refactored MadridHousingTrainer
testing the complete training pipeline.
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


class TestMadridHousingTrainerSimpleIntegration(unittest.TestCase):
    """Simple integration tests for MadridHousingTrainer."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.trainer = MadridHousingTrainer("nonexistent.yaml")
        
        # Create realistic test data
        np.random.seed(42)
        n_samples = 100
        
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
        
        self.integration_data = pd.DataFrame(data)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_full_training_pipeline_integration(self):
        """Test the complete training pipeline integration."""
        # Mock file operations and data loading
        with patch('train_model.pd.read_csv', return_value=self.integration_data), \
             patch('train_model.Path.exists', return_value=True), \
             patch('train_model.Path.mkdir'), \
             patch('utils.file_manager.joblib.dump'), \
             patch('utils.file_manager.json.dump'), \
             patch('utils.file_manager.open', mock.mock_open()):
            
            # Test data preparation
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data()
            
            # Verify data splits
            self.assertGreater(len(X_train), 0)
            self.assertGreater(len(X_val), 0)
            self.assertGreater(len(X_test), 0)
            
            # Verify no target column in features
            self.assertNotIn('buy_price', X_train.columns)
            self.assertNotIn('buy_price', X_val.columns)
            self.assertNotIn('buy_price', X_test.columns)
            
            # Verify target variables are series
            self.assertIsInstance(y_train, pd.Series)
            self.assertIsInstance(y_val, pd.Series)
            self.assertIsInstance(y_test, pd.Series)
    
    def test_training_pipeline_with_mocking(self):
        """Test training pipeline with comprehensive mocking."""
        # Mock all external dependencies
        with patch('train_model.pd.read_csv', return_value=self.integration_data), \
             patch('train_model.Path.exists', return_value=True), \
             patch('train_model.Path.mkdir'), \
             patch('utils.file_manager.joblib.dump'), \
             patch('utils.file_manager.json.dump'), \
             patch('utils.file_manager.open', mock.mock_open()), \
             patch('train_model.lgb.LGBMRegressor') as mock_lgb_class:
            
            # Setup mock model
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_lgb_class.return_value = mock_model
            
            # Mock utility methods
            with patch.object(self.trainer.metrics_calculator, 'calculate_metrics_with_model', return_value={
                'train_rmse': 0.5, 'train_mae': 0.3, 'train_r2': 0.8,
                'val_rmse': 0.6, 'val_mae': 0.4, 'val_r2': 0.7,
                'test_rmse': 0.7, 'test_mae': 0.5, 'test_r2': 0.6
            }), \
            patch.object(self.trainer.mlflow_logger, 'log_training_run', return_value="test_run_id"), \
            patch.object(self.trainer.versioning_manager, 'save_model_with_versioning'):
                
                # Test training pipeline
                result = self.trainer.run_training_pipeline(run_name="integration_test")
                
                # Verify result structure
                self.assertIsInstance(result, dict)
                self.assertIn('run_id', result)
                self.assertIn('model', result)
                self.assertIn('metrics', result)
                
                # Verify model was trained
                self.assertIsNotNone(self.trainer.model)
                
                # Verify metrics were calculated
                self.assertIsInstance(result['metrics'], dict)
    
    def test_data_preparation_integration(self):
        """Test data preparation integration."""
        # Mock data loading
        with patch('train_model.pd.read_csv', return_value=self.integration_data), \
             patch('train_model.Path.exists', return_value=True):
            
            # Test data preparation
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data()
            
            # Verify data structure
            self.assertIsInstance(X_train, pd.DataFrame)
            self.assertIsInstance(X_val, pd.DataFrame)
            self.assertIsInstance(X_test, pd.DataFrame)
            self.assertIsInstance(y_train, pd.Series)
            self.assertIsInstance(y_val, pd.Series)
            self.assertIsInstance(y_test, pd.Series)
            
            # Verify no data leakage
            total_samples = len(X_train) + len(X_val) + len(X_test)
            self.assertEqual(total_samples, len(self.integration_data))
            
            # Verify feature consistency
            self.assertEqual(len(X_train.columns), len(X_val.columns))
            self.assertEqual(len(X_train.columns), len(X_test.columns))


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestMadridHousingTrainerSimpleIntegration))
    
    # Run the tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Integration Tests Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with error code if there were failures or errors
    sys.exit(len(result.failures) + len(result.errors))
