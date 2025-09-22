"""
Unit tests for model_versioning_utils module.
"""

import unittest
import unittest.mock as mock
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_versioning_utils import ModelVersioningManager


class TestModelVersioningManagerUnittest(unittest.TestCase):
    """Unit tests for ModelVersioningManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize the versioning manager
        self.versioning_manager = ModelVersioningManager()
        
        # Create sample model and preprocessor
        self.sample_model = Mock()
        self.sample_preprocessor = Mock()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'sq_mt_built': [100, 150, 200, 250, 300],
            'n_rooms': [2, 3, 4, 5, 6],
            'n_bathrooms': [1, 2, 2, 3, 3],
            'buy_price': [200000, 300000, 400000, 500000, 600000]
        })
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ModelVersioningManager initialization."""
        self.assertIsNotNone(self.versioning_manager)
        self.assertIsNotNone(self.versioning_manager.file_manager)
    
    def test_save_model_with_versioning_basic(self):
        """Test basic model saving with versioning."""
        with patch('utils.file_manager.joblib.dump') as mock_dump, \
             patch('utils.file_manager.json.dump') as mock_json_dump, \
             patch('builtins.open', mock_open()), \
             patch.object(self.versioning_manager, '_save_version_info') as mock_save_info:
            
            result = self.versioning_manager.save_model_with_versioning(
                self.sample_model, "test_model.pkl"
            )
            
            # Should return None (void method)
            self.assertIsNone(result)
            
            # Should save model
            mock_dump.assert_called()
            mock_save_info.assert_called_once()
    
    def test_save_model_with_versioning_with_preprocessor(self):
        """Test model saving with preprocessor."""
        with patch('utils.file_manager.joblib.dump') as mock_dump, \
             patch('utils.file_manager.json.dump') as mock_json_dump, \
             patch('builtins.open', mock_open()), \
             patch.object(self.versioning_manager, '_save_version_info') as mock_save_info:
            
            result = self.versioning_manager.save_model_with_versioning(
                self.sample_model, "test_model.pkl", self.sample_preprocessor
            )
            
            # Should return None (void method)
            self.assertIsNone(result)
            
            # Should save model and preprocessor
            self.assertEqual(mock_dump.call_count, 2)
            mock_save_info.assert_called_once()
    
    def test_save_experiment_model(self):
        """Test saving experiment model."""
        with patch('utils.file_manager.joblib.dump') as mock_dump, \
             patch('utils.file_manager.json.dump') as mock_json_dump, \
             patch('builtins.open', mock_open()), \
             patch.object(self.versioning_manager, '_save_version_info') as mock_save_info:
            
            result = self.versioning_manager.save_experiment_model(
                self.sample_model, "experiment_1", self.sample_preprocessor
            )
            
            # Should return None (void method)
            self.assertIsNone(result)
            
            # Should save model and preprocessor
            self.assertEqual(mock_dump.call_count, 2)
            mock_save_info.assert_called_once()
    
    def test_find_best_model_from_experiments(self):
        """Test finding best model from experiments."""
        # Mock experiment results
        experiment_results = {
            'exp1': {'val_rmse': 0.5, 'val_mae': 0.3, 'val_r2': 0.8},
            'exp2': {'val_rmse': 0.4, 'val_mae': 0.2, 'val_r2': 0.9},
            'exp3': {'val_rmse': 0.6, 'val_mae': 0.4, 'val_r2': 0.7}
        }
        
        # Mock model loading
        with patch.object(self.versioning_manager.file_manager, 'load_model') as mock_load_model:
            mock_load_model.return_value = self.sample_model
            
            result = self.versioning_manager.find_best_model_from_experiments(experiment_results)
            
            # Should return best model info
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 5)
            self.assertEqual(result[3], 'exp2')  # Best experiment name
            self.assertEqual(result[4], 0.4)  # Best validation RMSE
    
    def test_find_best_model_from_experiments_empty(self):
        """Test finding best model from empty experiments."""
        experiment_results = {}
        
        with self.assertRaises(ValueError):
            self.versioning_manager.find_best_model_from_experiments(experiment_results)
    
    def test_list_trained_models(self):
        """Test listing trained models."""
        # Mock file operations
        with patch('os.listdir') as mock_listdir, \
             patch('builtins.open', mock_open(read_data=json.dumps({
                 'timestamp': '20240101_120000',
                 'model_path': 'model_1.pkl',
                 'preprocessor_path': 'preprocessor_1.pkl',
                 'model_type': 'LightGBM',
                 'creation_date': '2024-01-01'
             }))):
            
            mock_listdir.return_value = ['version_info_20240101_120000.json']
            
            models = self.versioning_manager.list_trained_models()
            
            # Should return list of models
            self.assertIsInstance(models, list)
            self.assertEqual(len(models), 1)
    
    def test_get_latest_model_info(self):
        """Test getting latest model info."""
        latest_info = {
            'timestamp': '20240102_130000',
            'model_path': 'model_2.pkl',
            'preprocessor_path': 'preprocessor_2.pkl',
            'model_type': 'LightGBM',
            'creation_date': '2024-01-02'
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(latest_info))):
            result = self.versioning_manager.get_latest_model_info()
            
            # Should return latest model info
            self.assertIsInstance(result, dict)
            self.assertEqual(result['timestamp'], '20240102_130000')
    
    def test_get_latest_model_info_not_found(self):
        """Test getting latest model info when file doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = self.versioning_manager.get_latest_model_info()
            
            # Should return None when file doesn't exist
            self.assertIsNone(result)
    
    def test_cleanup_old_models(self):
        """Test cleaning up old models."""
        # Mock file operations
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.getmtime') as mock_getmtime, \
             patch('os.remove') as mock_remove:
            
            # Mock files with different modification times
            mock_listdir.return_value = ['model1.pkl', 'model2.pkl', 'model3.pkl']
            mock_getmtime.side_effect = [1000, 2000, 3000]  # Different modification times
            
            result = self.versioning_manager.cleanup_old_models(keep_last=2)
            
            # Should remove old models
            self.assertIsInstance(result, int)
            self.assertEqual(result, 1)  # Should remove 1 old model
    
    def test_cleanup_old_models_no_files(self):
        """Test cleaning up when no files exist."""
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = []
            
            result = self.versioning_manager.cleanup_old_models(keep_last=2)
            
            # Should return 0 when no files
            self.assertEqual(result, 0)
    
    def test_save_version_info(self):
        """Test saving version info."""
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('utils.file_manager.json.dump') as mock_json_dump:
            
            self.versioning_manager._save_version_info(
                '20240101_120000', 'model.pkl', 'preprocessor.pkl', 'LightGBM'
            )
            
            # Should write to file
            mock_file.assert_called()
            mock_json_dump.assert_called()
    
    def test_save_version_info_error(self):
        """Test saving version info with error."""
        with patch('builtins.open', side_effect=IOError("Write error")):
            # Should not raise exception, just log error
            self.versioning_manager._save_version_info(
                '20240101_120000', 'model.pkl', 'preprocessor.pkl', 'LightGBM'
            )
    
    def test_get_model_info_from_path(self):
        """Test getting model info from path."""
        model_path = os.path.join(self.temp_dir, 'model_20240101_120000.pkl')
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps({
                 'timestamp': '20240101_120000',
                 'model_path': 'model_20240101_120000.pkl',
                 'preprocessor_path': 'preprocessor_20240101_120000.pkl',
                 'model_type': 'LightGBM',
                 'creation_date': '2024-01-01'
             }))):
            
            result = self.versioning_manager.get_model_info_from_path(model_path)
            
            # Should return model info
            self.assertIsInstance(result, dict)
            self.assertEqual(result['timestamp'], '20240101_120000')
    
    def test_get_model_info_from_path_not_found(self):
        """Test getting model info from non-existent path."""
        model_path = os.path.join(self.temp_dir, 'nonexistent.pkl')
        
        with patch('os.path.exists', return_value=False):
            result = self.versioning_manager.get_model_info_from_path(model_path)
            
            # Should return None when file doesn't exist
            self.assertIsNone(result)
    
    def test_compare_models(self):
        """Test comparing models."""
        model1_info = {
            'timestamp': '20240101_120000',
            'val_rmse': 0.5,
            'val_mae': 0.3,
            'val_r2': 0.8
        }
        
        model2_info = {
            'timestamp': '20240102_130000',
            'val_rmse': 0.4,
            'val_mae': 0.2,
            'val_r2': 0.9
        }
        
        result = self.versioning_manager.compare_models(model1_info, model2_info)
        
        # Should return comparison result
        self.assertIsInstance(result, dict)
        self.assertIn('better_model', result)
        self.assertIn('metrics_comparison', result)
    
    def test_compare_models_identical(self):
        """Test comparing identical models."""
        model_info = {
            'timestamp': '20240101_120000',
            'val_rmse': 0.5,
            'val_mae': 0.3,
            'val_r2': 0.8
        }
        
        result = self.versioning_manager.compare_models(model_info, model_info)
        
        # Should return comparison result
        self.assertIsInstance(result, dict)
        self.assertIn('better_model', result)
        self.assertIn('metrics_comparison', result)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestModelVersioningManagerUnittest))
    
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