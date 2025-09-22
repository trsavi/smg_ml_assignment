"""
Integration tests for data_loader.py using unittest framework.

This module provides integration tests for the data_loader module
testing the complete data loading and splitting pipeline.
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

from data_loader import load_data, split_data, _validate_housing_data


class TestDataLoaderIntegration(unittest.TestCase):
    """Integration tests for data_loader module."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create a larger, more realistic dataset
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
        
        # Ensure all prices are positive
        data['buy_price'] = np.abs(data['buy_price'])
        
        self.integration_data = pd.DataFrame(data)
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.integration_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_full_data_pipeline(self):
        """Test complete data loading and splitting pipeline."""
        # Load data
        data = load_data(self.temp_file.name)
        
        # Verify data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertIn('buy_price', data.columns)
        
        # Validate data
        _validate_housing_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            data, target_column='buy_price'
        )
        
        # Verify splits
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        
        # Verify no data leakage
        total_split_size = len(X_train) + len(X_test)
        self.assertEqual(total_split_size, len(data))
        
        # Verify feature consistency
        self.assertEqual(len(X_train.columns), len(X_test.columns))
        
        # Verify target consistency
        self.assertEqual(y_train.name, 'buy_price')
        self.assertEqual(y_test.name, 'buy_price')
    
    def test_data_consistency_across_splits(self):
        """Test that data is consistently split without overlap."""
        data = load_data(self.temp_file.name)
        X_train, X_test, y_train, y_test = split_data(
            data, target_column='buy_price', random_state=42
        )
        
        # Check that indices don't overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)
        
        # Check that all original indices are covered
        all_split_indices = train_indices.union(test_indices)
        original_indices = set(data.index)
        self.assertEqual(all_split_indices, original_indices)
    
    def test_feature_preservation(self):
        """Test that all features are preserved in splits."""
        data = load_data(self.temp_file.name)
        original_features = [col for col in data.columns if col != 'buy_price']
        
        X_train, X_test, y_train, y_test = split_data(
            data, target_column='buy_price'
        )
        
        # All splits should have the same features
        self.assertEqual(list(X_train.columns), original_features)
        self.assertEqual(list(X_test.columns), original_features)
    
    def test_target_distribution(self):
        """Test that target distribution is reasonable across splits."""
        data = load_data(self.temp_file.name)
        X_train, X_test, y_train, y_test = split_data(
            data, target_column='buy_price', random_state=42
        )
        
        # Check that target values are within reasonable range
        original_target = data['buy_price']
        
        self.assertGreater(y_train.min(), 0)
        self.assertGreater(y_test.min(), 0)
        
        # Check that splits maintain similar target distributions
        train_mean = y_train.mean()
        test_mean = y_test.mean()
        original_mean = original_target.mean()
        
        # Means should be within reasonable range of original
        self.assertLess(abs(train_mean - original_mean) / original_mean, 0.5)
        self.assertLess(abs(test_mean - original_mean) / original_mean, 0.5)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestDataLoaderIntegration))
    
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
