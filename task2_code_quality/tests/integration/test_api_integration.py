"""
Integration tests for api.py using unittest framework.

This module provides integration tests for the FastAPI application
testing the complete workflow and interactions between components.
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
from api import app
from utils.api import (
    APIConfigLoader,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    JSONHandler,
    ModelManager,
    PredictionRequest,
    SinglePredictionResponse,
    PredictionService
)


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
    test_suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))
    
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
