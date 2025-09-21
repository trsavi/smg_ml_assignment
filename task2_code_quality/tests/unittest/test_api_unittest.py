"""
Unit tests for api.py using unittest framework.

This module provides comprehensive unit tests for the FastAPI application
using the standard unittest framework.
"""

import unittest
import unittest.mock as mock
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import FastAPI test client
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# Import the API module
from api import app, load_model, save_request_json, PredictionRequest, BatchPredictionRequest, file_manager


class TestAPIComponents(unittest.TestCase):
    """Test cases for API components and utility functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
        
        # Create sample prediction request data
        self.sample_request = {
            'sq_mt_built': 85.5,
            'n_rooms': 3.0,
            'n_bathrooms': 2.0,
            'is_new_development': True,
            'has_ac': True,
            'has_fitted_wardrobes': True,
            'has_lift': 1.0,
            'is_exterior': 1.0,
            'has_pool': False,
            'has_terrace': True,
            'has_balcony': False,
            'has_storage_room': True,
            'is_accessible': True,
            'has_green_zones': True,
            'has_parking': True,
            'house_type_id_HouseType_1_Pisos': True,
            'house_type_id_HouseType_2_Casa_o_chalet': False,
            'house_type_id_HouseType_4_D_plex': False,
            'house_type_id_HouseType_5_ticos': False,
            'district_id_1': True,
            'district_id_2': False,
            'district_id_3': False,
            'district_id_4': False,
            'district_id_5': False,
            'district_id_6': False,
            'district_id_7': False,
            'district_id_8': False,
            'district_id_9': False,
            'district_id_10': False,
            'district_id_11': False,
            'district_id_12': False,
            'district_id_13': False,
            'district_id_14': False,
            'district_id_15': False,
            'district_id_17': False,
            'district_id_18': False,
            'district_id_19': False,
            'district_id_20': False
        }
        
        # Create sample batch request data
        self.sample_batch_request = {
            'data': [
                self.sample_request,
                {
                    **self.sample_request,
                    'sq_mt_built': 120.0,
                    'n_rooms': 4.0,
                    'district_id_1': False,
                    'district_id_2': True
                }
            ]
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_prediction_request_model(self):
        """Test PredictionRequest model validation."""
        request = PredictionRequest(**self.sample_request)
        
        self.assertEqual(request.sq_mt_built, 85.5)
        self.assertEqual(request.n_rooms, 3.0)
        self.assertEqual(request.n_bathrooms, 2.0)
        self.assertTrue(request.is_new_development)
        self.assertTrue(request.has_ac)
        self.assertTrue(request.house_type_id_HouseType_1_Pisos)
        self.assertTrue(request.district_id_1)
        self.assertFalse(request.district_id_2)
    
    def test_prediction_request_model_validation(self):
        """Test PredictionRequest model validation with invalid data."""
        invalid_request = self.sample_request.copy()
        invalid_request['sq_mt_built'] = "invalid"  # Should be float
        
        with self.assertRaises(ValueError):
            PredictionRequest(**invalid_request)
    
    def test_batch_prediction_request_model(self):
        """Test BatchPredictionRequest model validation."""
        batch_request = BatchPredictionRequest(**self.sample_batch_request)
        
        self.assertEqual(len(batch_request.data), 2)
        self.assertIsInstance(batch_request.data[0], PredictionRequest)
        self.assertIsInstance(batch_request.data[1], PredictionRequest)
    
    def test_save_request_json(self):
        """Test saving request JSON to file."""
        test_data = {"test": "data", "value": 123}
        
        with patch.object(file_manager, 'save_json') as mock_save_json:
            mock_save_json.return_value = "json_requests/test_file.json"
            
            result = save_request_json(test_data)
            
            # Verify FileManager.save_json was called
            mock_save_json.assert_called_once()
            call_args = mock_save_json.call_args
            self.assertEqual(call_args[0][0], test_data)  # First argument should be test_data
            self.assertTrue(call_args[0][1].startswith("json_requests/"))  # Second argument should be filepath
            
            # Verify return value is a string
            self.assertIsInstance(result, str)
    
    def test_save_request_json_creates_directory(self):
        """Test that save_request_json creates directory if it doesn't exist."""
        test_data = {"test": "data"}
        
        with patch.object(file_manager, 'save_json') as mock_save_json:
            mock_save_json.return_value = "json_requests/test_file.json"
            
            save_request_json(test_data)
            
            # Verify FileManager.save_json was called (which handles directory creation internally)
            mock_save_json.assert_called_once()
    
    def test_load_model_success(self):
        """Test successful model loading."""
        # Create mock model
        mock_model = Mock()
        mock_model.feature_name_ = ['feature1', 'feature2', 'feature3']
        mock_model.n_estimators = 100
        mock_model.learning_rate = 0.1
        mock_model.num_leaves = 31
        mock_model.objective = 'regression'
        mock_model.random_state = 42
        
        with patch.object(file_manager, 'load_model') as mock_load_model:
            mock_load_model.return_value = mock_model
            
            # Load model
            load_model("test_model.pkl")
            
            # Verify model was loaded
            mock_load_model.assert_called_once_with("test_model.pkl")
            
            # Verify global model was set
            from api import model, model_info
            self.assertEqual(model, mock_model)
            self.assertIn("model_name", model_info)
            self.assertEqual(model_info["model_type"], "Mock")
            self.assertEqual(model_info["algorithm"], "LightGBM")
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        with patch.object(file_manager, 'load_model') as mock_load_model:
            mock_load_model.side_effect = FileNotFoundError("Model file not found")
            
            with self.assertRaises(FileNotFoundError):
                load_model("nonexistent_model.pkl")
            
            mock_load_model.assert_called_once_with("nonexistent_model.pkl")


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
        
        # Create sample request data
        self.sample_request = {
            'sq_mt_built': 85.5,
            'n_rooms': 3.0,
            'n_bathrooms': 2.0,
            'is_new_development': True,
            'has_ac': True,
            'has_fitted_wardrobes': True,
            'has_lift': 1.0,
            'is_exterior': 1.0,
            'has_pool': False,
            'has_terrace': True,
            'has_balcony': False,
            'has_storage_room': True,
            'is_accessible': True,
            'has_green_zones': True,
            'has_parking': True,
            'house_type_id_HouseType_1_Pisos': True,
            'house_type_id_HouseType_2_Casa_o_chalet': False,
            'house_type_id_HouseType_4_D_plex': False,
            'house_type_id_HouseType_5_ticos': False,
            'district_id_1': True,
            'district_id_2': False,
            'district_id_3': False,
            'district_id_4': False,
            'district_id_5': False,
            'district_id_6': False,
            'district_id_7': False,
            'district_id_8': False,
            'district_id_9': False,
            'district_id_10': False,
            'district_id_11': False,
            'district_id_12': False,
            'district_id_13': False,
            'district_id_14': False,
            'district_id_15': False,
            'district_id_17': False,
            'district_id_18': False,
            'district_id_19': False,
            'district_id_20': False
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    @patch('api.model', None)
    def test_health_endpoint_no_model(self):
        """Test health endpoint when model is not loaded."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertFalse(data["model_loaded"])
    
    @patch('api.model', Mock())
    def test_health_endpoint_with_model(self):
        """Test health endpoint when model is loaded."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
    
    @patch('api.model', None)
    def test_model_info_endpoint_no_model(self):
        """Test model info endpoint when model is not loaded."""
        response = self.client.get("/model/info")
        
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["detail"], "Model not loaded")
    
    @patch('api.model', Mock())
    @patch('api.model_info', {"model_name": "Test Model", "version": "1.0.0"})
    def test_model_info_endpoint_with_model(self):
        """Test model info endpoint when model is loaded."""
        response = self.client.get("/model/info")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "Test Model")
        self.assertEqual(data["version"], "1.0.0")
    
    @patch('api.model', None)
    def test_predict_endpoint_no_model(self):
        """Test predict endpoint when model is not loaded."""
        response = self.client.post("/predict", json=self.sample_request)
        
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["detail"], "Model not loaded")
    
    @patch('api.model')
    @patch('api.save_request_json')
    def test_predict_endpoint_success(self, mock_save_request, mock_model):
        """Test successful prediction endpoint."""
        # Setup mock model
        mock_model.predict.return_value = np.array([250000.0])
        
        response = self.client.post("/predict", json=self.sample_request)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertEqual(data["prediction"], 250000.0)
        
        # Verify request was saved
        mock_save_request.assert_called_once()
        
        # Verify model prediction was called
        mock_model.predict.assert_called_once()
    
    @patch('api.model')
    @patch('api.save_request_json')
    def test_predict_endpoint_model_error(self, mock_save_request, mock_model):
        """Test predict endpoint when model prediction fails."""
        # Setup mock model to raise exception
        mock_model.predict.side_effect = Exception("Model prediction failed")
        
        response = self.client.post("/predict", json=self.sample_request)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Prediction failed", data["detail"])
        
        # Verify request was saved before prediction
        mock_save_request.assert_called_once()
    
    def test_predict_endpoint_invalid_request(self):
        """Test predict endpoint with invalid request data."""
        invalid_request = self.sample_request.copy()
        invalid_request['sq_mt_built'] = "invalid"  # Should be float
        
        response = self.client.post("/predict", json=invalid_request)
        
        self.assertEqual(response.status_code, 422)  # Validation error
    
    @patch('api.model', None)
    def test_batch_predict_endpoint_no_model(self):
        """Test batch predict endpoint when model is not loaded."""
        batch_request = {"data": [self.sample_request]}
        
        response = self.client.post("/batch_predict", json=batch_request)
        
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["detail"], "Model not loaded")
    
    @patch('api.model')
    @patch('api.save_request_json')
    def test_batch_predict_endpoint_success(self, mock_save_request, mock_model):
        """Test successful batch prediction endpoint."""
        # Setup mock model
        mock_model.predict.return_value = np.array([250000.0, 350000.0])
        
        batch_request = {
            "data": [self.sample_request, self.sample_request]
        }
        
        response = self.client.post("/batch_predict", json=batch_request)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertIn("count", data)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["predictions"], [250000.0, 350000.0])
        
        # Verify request was saved
        mock_save_request.assert_called_once()
        
        # Verify model prediction was called
        mock_model.predict.assert_called_once()
    
    @patch('api.model')
    @patch('api.save_request_json')
    def test_batch_predict_endpoint_model_error(self, mock_save_request, mock_model):
        """Test batch predict endpoint when model prediction fails."""
        # Setup mock model to raise exception
        mock_model.predict.side_effect = Exception("Batch prediction failed")
        
        batch_request = {"data": [self.sample_request]}
        
        response = self.client.post("/batch_predict", json=batch_request)
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Batch prediction failed", data["detail"])
        
        # Verify request was saved before prediction
        mock_save_request.assert_called_once()
    
    def test_batch_predict_endpoint_invalid_request(self):
        """Test batch predict endpoint with invalid request data."""
        batch_request = {"data": "invalid"}  # Should be list
        
        response = self.client.post("/batch_predict", json=batch_request)
        
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_batch_predict_endpoint_empty_batch(self):
        """Test batch predict endpoint with empty batch."""
        batch_request = {"data": []}
        
        with patch('api.model') as mock_model:
            mock_model.predict.return_value = np.array([])
            
            response = self.client.post("/batch_predict", json=batch_request)
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(len(data["predictions"]), 0)
            self.assertEqual(data["count"], 0)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for the API."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
        
        # Create sample request data
        self.sample_request = {
            'sq_mt_built': 85.5,
            'n_rooms': 3.0,
            'n_bathrooms': 2.0,
            'is_new_development': True,
            'has_ac': True,
            'has_fitted_wardrobes': True,
            'has_lift': 1.0,
            'is_exterior': 1.0,
            'has_pool': False,
            'has_terrace': True,
            'has_balcony': False,
            'has_storage_room': True,
            'is_accessible': True,
            'has_green_zones': True,
            'has_parking': True,
            'house_type_id_HouseType_1_Pisos': True,
            'house_type_id_HouseType_2_Casa_o_chalet': False,
            'house_type_id_HouseType_4_D_plex': False,
            'house_type_id_HouseType_5_ticos': False,
            'district_id_1': True,
            'district_id_2': False,
            'district_id_3': False,
            'district_id_4': False,
            'district_id_5': False,
            'district_id_6': False,
            'district_id_7': False,
            'district_id_8': False,
            'district_id_9': False,
            'district_id_10': False,
            'district_id_11': False,
            'district_id_12': False,
            'district_id_13': False,
            'district_id_14': False,
            'district_id_15': False,
            'district_id_17': False,
            'district_id_18': False,
            'district_id_19': False,
            'district_id_20': False
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    @patch('api.model')
    @patch('api.model_info')
    def test_full_api_workflow(self, mock_model_info, mock_model):
        """Test complete API workflow from health check to prediction."""
        # Setup mock model
        mock_model.predict.return_value = np.array([250000.0])
        mock_model_info = {
            "model_name": "Test Model",
            "version": "1.0.0",
            "model_type": "MockModel"
        }
        
        # Test health endpoint
        health_response = self.client.get("/health")
        self.assertEqual(health_response.status_code, 200)
        
        # Test model info endpoint
        info_response = self.client.get("/model/info")
        self.assertEqual(info_response.status_code, 200)
        
        # Test prediction endpoint
        predict_response = self.client.post("/predict", json=self.sample_request)
        self.assertEqual(predict_response.status_code, 200)
        
        # Test batch prediction endpoint
        batch_request = {"data": [self.sample_request, self.sample_request]}
        batch_response = self.client.post("/batch_predict", json=batch_request)
        self.assertEqual(batch_response.status_code, 200)
    
    def test_api_error_handling(self):
        """Test API error handling for various scenarios."""
        # Test with no model loaded
        with patch('api.model', None):
            # Health should work
            health_response = self.client.get("/health")
            self.assertEqual(health_response.status_code, 200)
            
            # Model info should fail
            info_response = self.client.get("/model/info")
            self.assertEqual(info_response.status_code, 503)
            
            # Predict should fail
            predict_response = self.client.post("/predict", json=self.sample_request)
            self.assertEqual(predict_response.status_code, 503)
    
    def test_api_validation(self):
        """Test API request validation."""
        # Test missing required fields
        incomplete_request = {
            'sq_mt_built': 85.5,
            'n_rooms': 3.0
            # Missing other required fields
        }
        
        response = self.client.post("/predict", json=incomplete_request)
        self.assertEqual(response.status_code, 422)  # Validation error
        
        # Test wrong data types
        invalid_request = self.sample_request.copy()
        invalid_request['sq_mt_built'] = "not_a_number"
        
        response = self.client.post("/predict", json=invalid_request)
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_api_content_types(self):
        """Test API handles different content types correctly."""
        # Test with valid JSON
        response = self.client.post("/predict", json=self.sample_request)
        # Should not be 415 (unsupported media type)
        self.assertNotEqual(response.status_code, 415)
        
        # Test with invalid JSON (should be handled by FastAPI)
        response = self.client.post(
            "/predict", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestAPIComponents))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpoints))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))
    
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
