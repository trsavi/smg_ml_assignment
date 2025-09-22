"""
Simple unit tests for api.py using unittest framework.

This module provides basic unit tests for the refactored FastAPI application
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
from fastapi import FastAPI
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


class TestAPISimple(unittest.TestCase):
    """Simple test cases for API components."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_prediction_request_validation(self):
        """Test PredictionRequest model validation."""
        # Test valid request
        valid_request = PredictionRequest(
            sq_mt_built=85.5,
            n_rooms=3.0,
            n_bathrooms=2.0,
            is_new_development=True,
            has_ac=True,
            has_fitted_wardrobes=True,
            has_lift=1.0,
            is_exterior=1.0,
            has_pool=False,
            has_terrace=True,
            has_balcony=False,
            has_storage_room=True,
            is_accessible=True,
            has_green_zones=True,
            has_parking=True,
            house_type_id_HouseType_1_Pisos=True,
            house_type_id_HouseType_2_Casa_o_chalet=False,
            house_type_id_HouseType_4_D_plex=False,
            house_type_id_HouseType_5_ticos=False,
            district_id_1=True,
            district_id_2=False,
            district_id_3=False,
            district_id_4=False,
            district_id_5=False,
            district_id_6=False,
            district_id_7=False,
            district_id_8=False,
            district_id_9=False,
            district_id_10=False,
            district_id_11=False,
            district_id_12=False,
            district_id_13=False,
            district_id_14=False,
            district_id_15=False,
            district_id_17=False,
            district_id_18=False,
            district_id_19=False,
            district_id_20=False
        )
        
        # Should not raise any validation error
        self.assertIsInstance(valid_request, PredictionRequest)
        self.assertEqual(valid_request.sq_mt_built, 85.5)
        self.assertEqual(valid_request.n_rooms, 3.0)
        self.assertTrue(valid_request.is_new_development)
    
    def test_batch_prediction_request_validation(self):
        """Test BatchPredictionRequest model validation."""
        # Test valid batch request
        sample_data = {
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
        
        valid_batch_request = BatchPredictionRequest(data=[sample_data, sample_data])
        
        # Should not raise any validation error
        self.assertIsInstance(valid_batch_request, BatchPredictionRequest)
        self.assertEqual(len(valid_batch_request.data), 2)
        self.assertIsInstance(valid_batch_request.data[0], PredictionRequest)
    
    def test_api_app_initialization(self):
        """Test that FastAPI app is properly initialized."""
        # Check that app is available
        self.assertIsNotNone(app)
        self.assertIsInstance(app, FastAPI)
        
        # Check that app has expected routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/model/info", "/predict", "/batch_predict"]
        
        for expected_route in expected_routes:
            self.assertIn(expected_route, routes)
    
    def test_api_app_metadata(self):
        """Test that FastAPI app has correct metadata."""
        # Check app title
        self.assertIsNotNone(app.title)
        self.assertIn("Madrid Housing", app.title)
        
        # Check app version
        self.assertIsNotNone(app.version)
        
        # Check app description
        self.assertIsNotNone(app.description)
        self.assertIn("housing", app.description.lower())


class TestAPIEndpointsSimple(unittest.TestCase):
    """Simple test cases for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)
        
        # Sample request data for testing
        self.sample_request = {
            "sq_mt_built": 100.0,
            "n_rooms": 3.0,
            "n_bathrooms": 2.0,
            "is_new_development": True,
            "has_ac": False,
            "has_fitted_wardrobes": True,
            "has_lift": 1.0,
            "is_exterior": 0.0,
            "has_pool": False,
            "has_terrace": True,
            "has_balcony": False,
            "has_storage_room": True,
            "is_accessible": False,
            "has_green_zones": True,
            "has_parking": False,
            "house_type_id_HouseType_1_Pisos": True,
            "house_type_id_HouseType_2_Casa_o_chalet": False,
            "house_type_id_HouseType_4_D_plex": False,
            "house_type_id_HouseType_5_ticos": False,
            "district_id_1": True,
            "district_id_2": False,
            "district_id_3": False,
            "district_id_4": False,
            "district_id_5": False,
            "district_id_6": False,
            "district_id_7": False,
            "district_id_8": False,
            "district_id_9": False,
            "district_id_10": False,
            "district_id_11": False,
            "district_id_12": False,
            "district_id_13": False,
            "district_id_14": False,
            "district_id_15": False,
            "district_id_17": False,
            "district_id_18": False,
            "district_id_19": False,
            "district_id_20": False
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        # Should return 200 status
        self.assertEqual(response.status_code, 200)
        
        # Should return JSON response
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = self.client.get("/model/info")
        
        # Should return either 200 or 503 (depending on model state)
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("model_name", data)
            self.assertIn("version", data)
    
    def test_predict_endpoint_structure(self):
        """Test prediction endpoint structure (without actual prediction)."""
        # Test with invalid data to check endpoint exists
        invalid_data = {"invalid": "data"}
        response = self.client.post("/predict", json=invalid_data)
        
        # Should return validation error (422) or service unavailable (503)
        self.assertIn(response.status_code, [422, 503])
    
    def test_batch_predict_endpoint_structure(self):
        """Test batch prediction endpoint structure (without actual prediction)."""
        # Test with invalid data to check endpoint exists
        invalid_data = {"invalid": "data"}
        response = self.client.post("/batch_predict", json=invalid_data)
        
        # Should return validation error (422) or service unavailable (503)
        self.assertIn(response.status_code, [422, 503])
    
    def test_predict_endpoint_with_model_loaded(self):
        """Test prediction endpoint when model is loaded."""
        # Mock model and model info
        with patch('api.model_manager.is_model_loaded', return_value=True), \
             patch('api.model_manager.get_model', return_value=Mock()), \
             patch('api.prediction_service.make_single_prediction', return_value={"prediction": 250000.0}):
            
            response = self.client.post("/predict", json=self.sample_request)
            
            # Should return 200 status
            self.assertEqual(response.status_code, 200)
            
            # Should return prediction data
            data = response.json()
            self.assertIn("prediction", data)
    
    def test_batch_predict_endpoint_with_model_loaded(self):
        """Test batch prediction endpoint when model is loaded."""
        # Mock model and model info
        with patch('api.model_manager.is_model_loaded', return_value=True), \
             patch('api.model_manager.get_model', return_value=Mock()), \
             patch('api.prediction_service.make_batch_prediction', return_value={"predictions": [250000.0, 300000.0], "count": 2}):
            
            batch_request = {"data": [self.sample_request, self.sample_request]}
            response = self.client.post("/batch_predict", json=batch_request)
            
            # Should return 200 status
            self.assertEqual(response.status_code, 200)
            
            # Should return prediction data
            data = response.json()
            self.assertIn("predictions", data)
    
    def test_model_info_endpoint_with_model_loaded(self):
        """Test model info endpoint when model is loaded."""
        # Mock model info
        with patch('api.model_manager.is_model_loaded', return_value=True), \
             patch('api.model_manager.get_model_info', return_value={
                 "model_name": "Test Model",
                 "version": "1.0.0",
                 "model_type": "LightGBM"
             }):
            
            response = self.client.get("/model/info")
            
            # Should return 200 status
            self.assertEqual(response.status_code, 200)
            
            # Should return model info
            data = response.json()
            self.assertIn("model_name", data)
            self.assertIn("version", data)
    
    def test_startup_event(self):
        """Test startup event handler."""
        # Mock model loading
        with patch('api.model_manager.load_model') as mock_load_model:
            # Import the startup function
            from api import startup_event
            
            # Test startup event
            import asyncio
            asyncio.run(startup_event())
            
            # Verify model was loaded
            mock_load_model.assert_called_once()


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestAPISimple))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpointsSimple))
    
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
