"""
Unit tests for model_manager.py using unittest framework.

This module provides comprehensive unit tests for the ModelManager class
to increase test coverage for the API utilities.
"""

import unittest
import unittest.mock as mock
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.api.model_manager import ModelManager
from utils.file_manager import FileManager


class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.file_manager = Mock(spec=FileManager)
        self.model_manager = ModelManager(self.file_manager)
        
        # Mock model with various attributes
        self.mock_model = Mock()
        self.mock_model.feature_name_ = ['sq_mt_built', 'n_rooms', 'n_bathrooms', 'is_new_development']
        self.mock_model.n_estimators = 100
        self.mock_model.learning_rate = 0.1
        self.mock_model.max_depth = 6
        self.mock_model.num_leaves = 31
        self.mock_model.objective = 'regression'
        self.mock_model.random_state = 42
        self.mock_model.predict.return_value = [250000.0, 300000.0]
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_initialization(self):
        """Test ModelManager initialization."""
        self.assertEqual(self.model_manager.file_manager, self.file_manager)
        self.assertIsNone(self.model_manager.model)
        self.assertEqual(self.model_manager.model_info, {})
        self.assertFalse(self.model_manager.is_loaded)
    
    def test_load_model_success(self):
        """Test successful model loading."""
        model_path = "test_model.pkl"
        
        with patch.object(self.file_manager, 'load_model', return_value=self.mock_model):
            self.model_manager.load_model(model_path)
            
            # Verify model is loaded
            self.assertEqual(self.model_manager.model, self.mock_model)
            self.assertTrue(self.model_manager.is_loaded)
            
            # Verify model info extraction
            model_info = self.model_manager.model_info
            self.assertEqual(model_info["model_name"], "Madrid Housing Price Prediction")
            self.assertEqual(model_info["version"], "1.0.0")
            self.assertEqual(model_info["model_type"], "Mock")
            self.assertEqual(model_info["algorithm"], "LightGBM")
            self.assertEqual(model_info["n_features"], 4)
            self.assertEqual(model_info["n_estimators"], 100)
            self.assertEqual(model_info["learning_rate"], 0.1)
            self.assertEqual(model_info["max_depth"], 6)
            self.assertEqual(model_info["num_leaves"], 31)
            self.assertEqual(model_info["objective"], "regression")
            self.assertEqual(model_info["random_state"], 42)
            self.assertEqual(model_info["feature_names"], ['sq_mt_built', 'n_rooms', 'n_bathrooms', 'is_new_development'])
            self.assertEqual(model_info["model_file"], model_path)
            self.assertIn("loaded_at", model_info)
    
    def test_load_model_with_minimal_attributes(self):
        """Test model loading with model that has minimal attributes."""
        minimal_model = Mock()
        minimal_model.predict.return_value = [250000.0]
        # Remove all attributes to simulate minimal model
        del minimal_model.feature_name_
        del minimal_model.n_estimators
        del minimal_model.learning_rate
        del minimal_model.max_depth
        del minimal_model.num_leaves
        del minimal_model.objective
        del minimal_model.random_state
        
        model_path = "minimal_model.pkl"
        
        with patch.object(self.file_manager, 'load_model', return_value=minimal_model):
            self.model_manager.load_model(model_path)
            
            # Verify model is loaded
            self.assertEqual(self.model_manager.model, minimal_model)
            self.assertTrue(self.model_manager.is_loaded)
            
            # Verify model info with default values
            model_info = self.model_manager.model_info
            self.assertEqual(model_info["n_features"], 0)
            self.assertIsNone(model_info["n_estimators"])
            self.assertIsNone(model_info["learning_rate"])
            self.assertIsNone(model_info["max_depth"])
            self.assertIsNone(model_info["num_leaves"])
            self.assertIsNone(model_info["objective"])
            self.assertIsNone(model_info["random_state"])
            self.assertEqual(model_info["feature_names"], [])
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        model_path = "nonexistent_model.pkl"
        
        with patch.object(self.file_manager, 'load_model', side_effect=Exception("File not found")):
            with self.assertRaises(Exception):
                self.model_manager.load_model(model_path)
            
            # Verify state after failure
            self.assertIsNone(self.model_manager.model)
            self.assertFalse(self.model_manager.is_loaded)
            self.assertEqual(self.model_manager.model_info, {})
    
    def test_get_model_when_loaded(self):
        """Test getting model when it's loaded."""
        self.model_manager.model = self.mock_model
        self.model_manager.is_loaded = True
        
        result = self.model_manager.get_model()
        self.assertEqual(result, self.mock_model)
    
    def test_get_model_when_not_loaded(self):
        """Test getting model when it's not loaded."""
        result = self.model_manager.get_model()
        self.assertIsNone(result)
    
    def test_get_model_info_when_loaded(self):
        """Test getting model info when model is loaded."""
        self.model_manager.model_info = {"model_type": "Mock", "n_features": 4}
        
        result = self.model_manager.get_model_info()
        self.assertEqual(result, {"model_type": "Mock", "n_features": 4})
    
    def test_get_model_info_when_not_loaded(self):
        """Test getting model info when model is not loaded."""
        result = self.model_manager.get_model_info()
        self.assertEqual(result, {})
    
    def test_is_model_loaded_true(self):
        """Test is_model_loaded when model is loaded."""
        self.model_manager.is_loaded = True
        result = self.model_manager.is_model_loaded()
        self.assertTrue(result)
    
    def test_is_model_loaded_false(self):
        """Test is_model_loaded when model is not loaded."""
        result = self.model_manager.is_model_loaded()
        self.assertFalse(result)
    
    def test_predict_success(self):
        """Test successful prediction."""
        self.model_manager.model = self.mock_model
        self.model_manager.is_loaded = True
        
        test_data = [[85.5, 3, 2, 1]]
        result = self.model_manager.predict(test_data)
        
        # Verify model predict was called
        self.mock_model.predict.assert_called_once_with(test_data)
        self.assertEqual(result, [250000.0, 300000.0])
    
    def test_predict_when_model_not_loaded(self):
        """Test prediction when model is not loaded."""
        with self.assertRaises(ValueError) as context:
            self.model_manager.predict([[85.5, 3, 2, 1]])
        
        self.assertIn("Model is not loaded", str(context.exception))
    
    def test_predict_when_model_is_none(self):
        """Test prediction when model is None."""
        self.model_manager.is_loaded = True
        self.model_manager.model = None
        
        with self.assertRaises(ValueError) as context:
            self.model_manager.predict([[85.5, 3, 2, 1]])
        
        self.assertIn("Model is not loaded", str(context.exception))
    
    def test_load_model_with_partial_attributes(self):
        """Test model loading with model that has some attributes missing."""
        partial_model = Mock()
        partial_model.feature_name_ = ['sq_mt_built', 'n_rooms']
        partial_model.n_estimators = 50
        # Remove missing attributes
        del partial_model.learning_rate
        del partial_model.max_depth
        del partial_model.num_leaves
        del partial_model.objective
        del partial_model.random_state
        
        model_path = "partial_model.pkl"
        
        with patch.object(self.file_manager, 'load_model', return_value=partial_model):
            self.model_manager.load_model(model_path)
            
            # Verify model info with partial attributes
            model_info = self.model_manager.model_info
            self.assertEqual(model_info["n_features"], 2)
            self.assertEqual(model_info["n_estimators"], 50)
            self.assertIsNone(model_info["learning_rate"])
            self.assertIsNone(model_info["max_depth"])
            self.assertIsNone(model_info["num_leaves"])
            self.assertIsNone(model_info["objective"])
            self.assertIsNone(model_info["random_state"])
            self.assertEqual(model_info["feature_names"], ['sq_mt_built', 'n_rooms'])
    
    def test_load_model_with_different_model_type(self):
        """Test model loading with different model type."""
        different_model = Mock()
        different_model.__class__.__name__ = "RandomForestRegressor"
        different_model.feature_name_ = ['feature1', 'feature2']
        different_model.n_estimators = 200
        different_model.max_depth = 10
        different_model.random_state = 123
        # Remove LightGBM specific attributes
        del different_model.learning_rate
        del different_model.num_leaves
        del different_model.objective
        
        model_path = "rf_model.pkl"
        
        with patch.object(self.file_manager, 'load_model', return_value=different_model):
            self.model_manager.load_model(model_path)
            
            # Verify model info
            model_info = self.model_manager.model_info
            self.assertEqual(model_info["model_type"], "RandomForestRegressor")
            self.assertEqual(model_info["n_features"], 2)
            self.assertEqual(model_info["n_estimators"], 200)
            self.assertEqual(model_info["max_depth"], 10)
            self.assertEqual(model_info["random_state"], 123)
            # LightGBM specific attributes should be None
            self.assertIsNone(model_info["learning_rate"])
            self.assertIsNone(model_info["num_leaves"])
            self.assertIsNone(model_info["objective"])
    
    def test_predict_with_different_data_types(self):
        """Test prediction with different data types."""
        self.model_manager.model = self.mock_model
        self.model_manager.is_loaded = True
        
        # Test with list
        list_data = [[85.5, 3, 2, 1], [90.0, 4, 3, 0]]
        result1 = self.model_manager.predict(list_data)
        self.mock_model.predict.assert_called_with(list_data)
        
        # Reset mock
        self.mock_model.reset_mock()
        
        # Test with numpy array (mocked)
        import numpy as np
        array_data = np.array([[85.5, 3, 2, 1]])
        result2 = self.model_manager.predict(array_data)
        self.mock_model.predict.assert_called_with(array_data)


if __name__ == '__main__':
    unittest.main()
