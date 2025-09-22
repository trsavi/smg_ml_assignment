"""
Unit tests for metrics_utils module.
"""

import unittest
import unittest.mock as mock
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.metrics_utils import MetricsCalculator


class TestMetricsCalculatorUnittest(unittest.TestCase):
    """Unit tests for MetricsCalculator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.metrics_calculator = MetricsCalculator()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'sq_mt_built': [100, 150, 200, 250, 300],
            'n_rooms': [2, 3, 4, 5, 6],
            'n_bathrooms': [1, 2, 2, 3, 3],
            'buy_price': [200000, 300000, 400000, 500000, 600000]
        })
        
        self.X = self.sample_data.drop('buy_price', axis=1)
        self.y = self.sample_data['buy_price']
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        # Mock model for calculate_metrics
        mock_model = Mock()
        mock_model.predict.return_value = np.array([250000, 350000, 450000, 550000, 650000])
        
        metrics = self.metrics_calculator.calculate_metrics_with_model(mock_model, self.X, self.y, "test_dataset")
        
        # Check that all expected metrics are present
        expected_metrics = ['test_dataset_rmse', 'test_dataset_mae', 'test_dataset_r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_calculate_metrics_with_model(self):
        """Test metrics calculation with a model."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([250000, 350000, 450000, 550000, 650000])
        
        metrics = self.metrics_calculator.calculate_metrics_with_model(
            mock_model, self.X, self.y, "test_dataset"
        )
        
        # Check that model was called
        mock_model.predict.assert_called_once()
        
        # Check that all expected metrics are present
        expected_metrics = ['test_dataset_rmse', 'test_dataset_mae', 'test_dataset_r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_calculate_metrics_with_model_error(self):
        """Test metrics calculation with model error."""
        # Mock model that raises an error
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")
        
        with self.assertRaises(Exception):
            self.metrics_calculator.calculate_metrics_with_model(
                mock_model, self.X, self.y, "test_dataset"
            )
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        # Mock model
        mock_model = Mock()
        # Return predictions matching the data length for each split
        def mock_predict(X):
            return np.array([250000 + i * 100000 for i in range(len(X))])
        mock_model.predict.side_effect = mock_predict
        
        # Create train/val/test splits
        X_train = self.X.iloc[:2]
        y_train = self.y.iloc[:2]
        X_val = self.X.iloc[2:3]
        y_val = self.y.iloc[2:3]
        X_test = self.X.iloc[3:4]
        y_test = self.y.iloc[3:4]
        
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            mock_model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Check that all expected metrics are present
        self.assertIn('train', metrics)
        self.assertIn('val', metrics)
        self.assertIn('test', metrics)
        
        # Check train metrics
        train_metrics = metrics['train']
        expected_train_metrics = ['train_rmse', 'train_mae', 'train_r2']
        for metric in expected_train_metrics:
            self.assertIn(metric, train_metrics)
            self.assertIsInstance(train_metrics[metric], (int, float))
        
        # Check val metrics
        val_metrics = metrics['val']
        expected_val_metrics = ['val_rmse', 'val_mae', 'val_r2']
        for metric in expected_val_metrics:
            self.assertIn(metric, val_metrics)
            self.assertIsInstance(val_metrics[metric], (int, float))
        
        # Check test metrics
        test_metrics = metrics['test']
        expected_test_metrics = ['test_rmse', 'test_mae', 'test_r2']
        for metric in expected_test_metrics:
            self.assertIn(metric, test_metrics)
            self.assertIsInstance(test_metrics[metric], (int, float))
    
    def test_calculate_comprehensive_metrics_with_error(self):
        """Test comprehensive metrics calculation with error."""
        # Mock model that raises an error
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")
        
        # Create train/val/test splits
        X_train = self.X.iloc[:2]
        y_train = self.y.iloc[:2]
        X_val = self.X.iloc[2:3]
        y_val = self.y.iloc[2:3]
        X_test = self.X.iloc[3:4]
        y_test = self.y.iloc[3:4]
        
        with self.assertRaises(Exception):
            self.metrics_calculator.calculate_comprehensive_metrics(
                mock_model, X_train, y_train, X_val, y_val, X_test, y_test
            )
    
    def test_calculate_metrics_with_empty_data(self):
        """Test metrics calculation with empty data."""
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([])  # Empty array for empty data
        with self.assertRaises((ValueError, IndexError)):
            self.metrics_calculator.calculate_metrics_with_model(mock_model, empty_df, empty_series, "test_dataset")
    
    def test_calculate_metrics_with_single_value(self):
        """Test metrics calculation with single value."""
        single_X = pd.DataFrame({'feature1': [100]})
        single_y = pd.Series([200000])
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([200000])
        metrics = self.metrics_calculator.calculate_metrics_with_model(mock_model, single_X, single_y, "test_dataset")
        
        # Should handle single value
        expected_metrics = ['test_dataset_rmse', 'test_dataset_mae', 'test_dataset_r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_calculate_metrics_with_different_datasets(self):
        """Test metrics calculation with different dataset names."""
        # Test with different dataset names
        dataset_names = ["train", "validation", "test", "custom_dataset"]
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([250000, 350000, 450000, 550000, 650000])
        
        for dataset_name in dataset_names:
            metrics = self.metrics_calculator.calculate_metrics_with_model(mock_model, self.X, self.y, dataset_name)
            
            # Should return valid metrics for each dataset
            expected_metrics = [f'{dataset_name}_rmse', f'{dataset_name}_mae', f'{dataset_name}_r2']
            for metric in expected_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
    
    def test_initialization(self):
        """Test MetricsCalculator initialization."""
        # Test with default file manager
        calculator1 = MetricsCalculator()
        self.assertIsNotNone(calculator1.file_manager)
        
        # Test with custom file manager
        mock_file_manager = Mock()
        calculator2 = MetricsCalculator(mock_file_manager)
        self.assertEqual(calculator2.file_manager, mock_file_manager)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestMetricsCalculatorUnittest))
    
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