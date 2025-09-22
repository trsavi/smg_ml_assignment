"""
Integration tests for train_model.py using unittest framework.

This module provides integration tests for the MadridHousingTrainer
testing the complete training pipeline and model versioning.
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


class TestMadridHousingTrainerIntegration(unittest.TestCase):
    """Integration tests for MadridHousingTrainer."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.trainer = MadridHousingTrainer("nonexistent.yaml")
    
    def test_full_training_pipeline_integration(self):
        """Test the complete training pipeline integration."""
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
        for house_type in ['HouseType_1_Piso', 'HouseType_2_Casa_o_chalet', 'HouseType_3_Estudio']:
            data[f'house_type_id_{house_type}'] = np.random.choice([True, False], n_samples)
        
        for district in ['District_1_Arganzuela', 'District_2_Barajas', 'District_3_Carabanchel']:
            data[f'district_id_{district}'] = np.random.choice([True, False], n_samples)
        
        # Create target variable with some relationship to features
        data['buy_price'] = (
            data['sq_mt_built'] * 1000 + 
            data['n_rooms'] * 50000 + 
            data['n_bathrooms'] * 30000 + 
            np.random.normal(0, 50000, n_samples)
        )
        
        df = pd.DataFrame(data)
        
        # Mock the data loading
        with patch('train_model.pd.read_csv', return_value=df), \
             patch('train_model.Path.exists', return_value=True):
            
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
    
    def test_model_versioning_integration(self):
        """Test model versioning system integration."""
        # Create test data
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
        df = pd.DataFrame(data)
        
        # Mock file operations and data loading
        with patch('train_model.pd.read_csv', return_value=df), \
             patch('train_model.Path.exists', return_value=True), \
             patch('train_model.Path.mkdir'), \
             patch('train_model.joblib.dump'), \
             patch('train_model.json.dump'), \
             patch('train_model.open', mock.mock_open()):
            
            # Test training pipeline with versioning
            result = self.trainer.run_training_pipeline(run_name="integration_test")
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn('run_id', result)
            self.assertIn('model', result)
            self.assertIn('metrics', result)
            
            # Verify model is trained
            self.assertIsNotNone(result['model'])
            
            # Verify metrics are calculated
            self.assertIsInstance(result['metrics'], dict)
            self.assertIn('train', result['metrics'])
            self.assertIn('val', result['metrics'])
            self.assertIn('test', result['metrics'])
    
    def test_experiments_integration(self):
        """Test multiple experiments integration."""
        # Create test data
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
        df = pd.DataFrame(data)
        
        # Mock file operations and data loading
        with patch('train_model.pd.read_csv', return_value=df), \
             patch('train_model.Path.exists', return_value=True), \
             patch('train_model.Path.mkdir'), \
             patch('train_model.joblib.dump'), \
             patch('train_model.json.dump'), \
             patch('train_model.open', mock.mock_open()):
            
            # Test experiments with mock config
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
                
                # Verify result structure
                self.assertIsInstance(result, dict)
                self.assertIn('test_exp_1', result)
                self.assertIn('test_exp_2', result)
                
                # Verify each experiment has required fields
                for exp_name, exp_result in result.items():
                    if 'error' not in exp_result:
                        self.assertIn('run_id', exp_result)
                        self.assertIn('model', exp_result)
                        self.assertIn('metrics', exp_result)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestMadridHousingTrainerIntegration))
    
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
