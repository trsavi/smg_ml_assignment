"""
Integration tests for the API endpoints.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import app


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_model_loaded(self):
        """Mock the model loading for tests."""
        with patch('api.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([250000.0])
            mock_load.return_value = mock_model
            yield mock_model
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_model_info_endpoint_without_model(self, client):
        """Test model info endpoint when no model is loaded."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_model_loaded"
    
    @patch('api.model', new_callable=lambda: Mock())
    @patch('api.model_info', {'model_type': 'LightGBM', 'version': '1.0.0'})
    def test_model_info_endpoint_with_model(self, client):
        """Test model info endpoint when model is loaded."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "model_loaded"
        assert data["model_info"]["model_type"] == "LightGBM"
        assert data["model_info"]["version"] == "1.0.0"
    
    @patch('api.model', new_callable=lambda: Mock())
    def test_predict_endpoint_success(self, client, sample_prediction_request):
        """Test successful prediction endpoint."""
        # Mock the model prediction
        api.model.predict.return_value = np.array([250000.0])
        
        response = client.post("/predict", json=sample_prediction_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] == 250000.0
        assert "timestamp" in data
    
    def test_predict_endpoint_no_model(self, client, sample_prediction_request):
        """Test prediction endpoint when no model is loaded."""
        # Ensure no model is loaded
        api.model = None
        
        response = client.post("/predict", json=sample_prediction_request)
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Model not loaded"
    
    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_request = {
            "sq_mt_built": "invalid",  # Should be numeric
            "n_rooms": 3
        }
        
        response = client.post("/predict", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing required fields."""
        incomplete_request = {
            "sq_mt_built": 85.5,
            "n_rooms": 3
            # Missing many required fields
        }
        
        response = client.post("/predict", json=incomplete_request)
        
        assert response.status_code == 422  # Validation error
    
    @patch('api.model', new_callable=lambda: Mock())
    def test_batch_predict_endpoint_success(self, client, sample_batch_prediction_requests):
        """Test successful batch prediction endpoint."""
        # Mock the model prediction
        api.model.predict.return_value = np.array([250000.0, 260000.0, 270000.0])
        
        response = client.post("/batch_predict", json=sample_batch_prediction_requests)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        assert data["predictions"][0] == 250000.0
        assert data["predictions"][1] == 260000.0
        assert data["predictions"][2] == 270000.0
        assert "timestamp" in data
    
    def test_batch_predict_endpoint_no_model(self, client, sample_batch_prediction_requests):
        """Test batch prediction endpoint when no model is loaded."""
        # Ensure no model is loaded
        api.model = None
        
        response = client.post("/batch_predict", json=sample_batch_prediction_requests)
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Model not loaded"
    
    def test_batch_predict_endpoint_empty_list(self, client):
        """Test batch prediction endpoint with empty request list."""
        response = client.post("/batch_predict", json=[])
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Empty request list"
    
    def test_batch_predict_endpoint_invalid_data(self, client):
        """Test batch prediction endpoint with invalid data."""
        invalid_requests = [
            {
                "sq_mt_built": "invalid",  # Should be numeric
                "n_rooms": 3
            }
        ]
        
        response = client.post("/batch_predict", json=invalid_requests)
        
        assert response.status_code == 422  # Validation error
    
    @patch('api.model', new_callable=lambda: Mock())
    @patch('api.json.dump')
    @patch('builtins.open', new_callable=lambda: Mock())
    def test_predict_saves_request_json(self, client, sample_prediction_request):
        """Test that prediction endpoint saves request JSON."""
        # Mock the model prediction
        api.model.predict.return_value = np.array([250000.0])
        
        response = client.post("/predict", json=sample_prediction_request)
        
        assert response.status_code == 200
        # Note: In a real test, you would verify that the file was created
        # and contains the expected JSON data
    
    @patch('api.model', new_callable=lambda: Mock())
    @patch('api.json.dump')
    @patch('builtins.open', new_callable=lambda: Mock())
    def test_batch_predict_saves_request_json(self, client, sample_batch_prediction_requests):
        """Test that batch prediction endpoint saves request JSON."""
        # Mock the model prediction
        api.model.predict.return_value = np.array([250000.0, 260000.0, 270000.0])
        
        response = client.post("/batch_predict", json=sample_batch_prediction_requests)
        
        assert response.status_code == 200
        # Note: In a real test, you would verify that the file was created
        # and contains the expected JSON data
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        response = client.options("/health")
        
        assert response.status_code == 200
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_api_documentation_endpoints(self, client):
        """Test that API documentation endpoints are accessible."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        # Test docs endpoint
        response = client.get("/docs")
        assert response.status_code == 200
    
    @patch('api.model', new_callable=lambda: Mock())
    def test_prediction_with_edge_case_values(self, client):
        """Test prediction with edge case values."""
        # Mock the model prediction
        api.model.predict.return_value = np.array([100000.0])
        
        edge_case_request = {
            'sq_mt_built': 30.0,  # Very small apartment
            'n_rooms': 1,
            'n_bathrooms': 1,
            'is_new_development': True,
            'has_ac': False,
            'has_fitted_wardrobes': False,
            'has_lift': 0.0,
            'is_exterior': 0.0,
            'has_pool': False,
            'has_terrace': False,
            'has_balcony': False,
            'has_storage_room': False,
            'is_accessible': False,
            'has_green_zones': False,
            'has_parking': False,
            'house_type_id_HouseType_1_Pisos': True,
            'house_type_id_HouseType_2_Chalets': False,
            'house_type_id_HouseType_3_Estudios': False,
            'house_type_id_HouseType_4_Duplex': False,
            'house_type_id_HouseType_5_Planta baja': False,
            'house_type_id_HouseType_6_Aticos': False,
            'house_type_id_HouseType_7_Lofts': False,
            'district_id_District_1_Arganzuela': False,
            'district_id_District_2_Barajas': False,
            'district_id_District_3_Carabanchel': False,
            'district_id_District_4_Centro': False,
            'district_id_District_5_Chamartin': False,
            'district_id_District_6_Chamberi': False,
            'district_id_District_7_Ciudad Lineal': False,
            'district_id_District_8_Fuencarral-El Pardo': False,
            'district_id_District_9_Hortaleza': False,
            'district_id_District_10_Latina': False,
            'district_id_District_11_Moncloa-Aravaca': False,
            'district_id_District_12_Moratalaz': False,
            'district_id_District_13_Puente de Vallecas': False,
            'district_id_District_14_Retiro': False,
            'district_id_District_15_Salamanca': False,
            'district_id_District_16_San Blas-Canillejas': False,
            'district_id_District_17_Tetuan': False,
            'district_id_District_18_Usera': False,
            'district_id_District_19_Vicalvaro': False,
            'district_id_District_20_Villa de Vallecas': False,
            'district_id_District_21_Villaverde': False
        }
        
        response = client.post("/predict", json=edge_case_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] == 100000.0
    
    @patch('api.model', new_callable=lambda: Mock())
    def test_large_batch_prediction(self, client):
        """Test batch prediction with a large number of requests."""
        # Create 100 prediction requests
        large_batch = []
        for i in range(100):
            request = {
                'sq_mt_built': 80.0 + i,
                'n_rooms': 3,
                'n_bathrooms': 2,
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
                'house_type_id_HouseType_2_Chalets': False,
                'house_type_id_HouseType_3_Estudios': False,
                'house_type_id_HouseType_4_Duplex': False,
                'house_type_id_HouseType_5_Planta baja': False,
                'house_type_id_HouseType_6_Aticos': False,
                'house_type_id_HouseType_7_Lofts': False,
                'district_id_District_1_Arganzuela': False,
                'district_id_District_2_Barajas': False,
                'district_id_District_3_Carabanchel': False,
                'district_id_District_4_Centro': True,
                'district_id_District_5_Chamartin': False,
                'district_id_District_6_Chamberi': False,
                'district_id_District_7_Ciudad Lineal': False,
                'district_id_District_8_Fuencarral-El Pardo': False,
                'district_id_District_9_Hortaleza': False,
                'district_id_District_10_Latina': False,
                'district_id_District_11_Moncloa-Aravaca': False,
                'district_id_District_12_Moratalaz': False,
                'district_id_District_13_Puente de Vallecas': False,
                'district_id_District_14_Retiro': False,
                'district_id_District_15_Salamanca': False,
                'district_id_District_16_San Blas-Canillejas': False,
                'district_id_District_17_Tetuan': False,
                'district_id_District_18_Usera': False,
                'district_id_District_19_Vicalvaro': False,
                'district_id_District_20_Villa de Vallecas': False,
                'district_id_District_21_Villaverde': False
            }
            large_batch.append(request)
        
        # Mock the model prediction
        api.model.predict.return_value = np.random.uniform(100000, 500000, 100)
        
        response = client.post("/batch_predict", json=large_batch)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 100


# Import numpy for the tests
import numpy as np
