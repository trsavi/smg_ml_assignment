"""
Unit tests for json_handler.py using unittest framework.

This module provides comprehensive unit tests for the JSONHandler class
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

from utils.api.json_handler import JSONHandler
from utils.file_manager import FileManager


class TestJSONHandler(unittest.TestCase):
    """Test cases for JSONHandler class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.file_manager = Mock(spec=FileManager)
        self.config = {
            "request_handling": {
                "request_dir": "test_requests",
                "save_requests": True
            }
        }
        self.json_handler = JSONHandler(self.file_manager, self.config)
        
        # Sample request data
        self.sample_request = {
            "sq_mt_built": 85.5,
            "n_rooms": 3.0,
            "n_bathrooms": 2.0,
            "is_new_development": True,
            "has_ac": True
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_initialization_with_save_requests_enabled(self):
        """Test JSONHandler initialization with save_requests enabled."""
        self.assertEqual(self.json_handler.file_manager, self.file_manager)
        self.assertEqual(self.json_handler.config, self.config)
        self.assertEqual(self.json_handler.request_dir, Path("test_requests"))
        self.assertTrue(self.json_handler.save_requests)
    
    def test_initialization_with_save_requests_disabled(self):
        """Test JSONHandler initialization with save_requests disabled."""
        config_disabled = {
            "request_handling": {
                "request_dir": "test_requests",
                "save_requests": False
            }
        }
        handler = JSONHandler(self.file_manager, config_disabled)
        self.assertFalse(handler.save_requests)
    
    def test_initialization_with_default_config(self):
        """Test JSONHandler initialization with minimal config."""
        minimal_config = {}
        handler = JSONHandler(self.file_manager, minimal_config)
        self.assertEqual(handler.request_dir, Path("json_requests"))
        self.assertTrue(handler.save_requests)
    
    def test_save_request_success(self):
        """Test successful request saving."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch.object(self.file_manager, 'save_json') as mock_save_json:
                result = self.json_handler.save_request(self.sample_request, "single")
                
                # Verify directory creation
                mock_mkdir.assert_called_once_with(exist_ok=True)
                
                # Verify file saving
                mock_save_json.assert_called_once()
                call_args = mock_save_json.call_args
                self.assertEqual(call_args[0][0], self.sample_request)
                file_path = str(call_args[0][1])
                self.assertTrue("request_single_" in file_path)
                self.assertTrue(file_path.endswith(".json"))
                
                # Verify return value
                self.assertTrue("request_single_" in result)
                self.assertTrue(result.endswith(".json"))
    
    def test_save_request_with_save_disabled(self):
        """Test request saving when save_requests is disabled."""
        self.json_handler.save_requests = False
        result = self.json_handler.save_request(self.sample_request, "single")
        self.assertEqual(result, "")
    
    def test_save_request_exception_handling(self):
        """Test request saving when an exception occurs."""
        with patch('pathlib.Path.mkdir', side_effect=Exception("Directory error")):
            result = self.json_handler.save_request(self.sample_request, "single")
            self.assertEqual(result, "")
    
    def test_save_request_batch_type(self):
        """Test saving batch request."""
        with patch('pathlib.Path.mkdir'):
            with patch.object(self.file_manager, 'save_json') as mock_save_json:
                result = self.json_handler.save_request(self.sample_request, "batch")
                
                # Verify file saving with batch type
                mock_save_json.assert_called_once()
                call_args = mock_save_json.call_args
                file_path = str(call_args[0][1])
                self.assertTrue("request_batch_" in file_path)
                self.assertTrue(file_path.endswith(".json"))
    
    def test_format_single_response_without_request_id(self):
        """Test formatting single response without request ID."""
        prediction = 250000.0
        response = self.json_handler.format_single_response(prediction)
        
        expected = {"prediction": prediction}
        self.assertEqual(response, expected)
    
    def test_format_single_response_with_request_id(self):
        """Test formatting single response with request ID."""
        prediction = 250000.0
        request_id = "req_123"
        response = self.json_handler.format_single_response(prediction, request_id)
        
        expected = {
            "prediction": prediction,
            "request_id": request_id
        }
        self.assertEqual(response, expected)
    
    def test_format_batch_response_without_request_id(self):
        """Test formatting batch response without request ID."""
        predictions = [250000.0, 300000.0, 350000.0]
        count = 3
        response = self.json_handler.format_batch_response(predictions, count)
        
        expected = {
            "predictions": predictions,
            "count": count
        }
        self.assertEqual(response, expected)
    
    def test_format_batch_response_with_request_id(self):
        """Test formatting batch response with request ID."""
        predictions = [250000.0, 300000.0, 350000.0]
        count = 3
        request_id = "batch_req_123"
        response = self.json_handler.format_batch_response(predictions, count, request_id)
        
        expected = {
            "predictions": predictions,
            "count": count,
            "request_id": request_id
        }
        self.assertEqual(response, expected)
    
    def test_format_error_response_default_code(self):
        """Test formatting error response with default error code."""
        error_message = "Model prediction failed"
        response = self.json_handler.format_error_response(error_message)
        
        self.assertTrue(response["error"])
        self.assertEqual(response["error_code"], "PREDICTION_ERROR")
        self.assertEqual(response["message"], error_message)
        self.assertIn("timestamp", response)
        self.assertIsInstance(response["timestamp"], str)
    
    def test_format_error_response_custom_code(self):
        """Test formatting error response with custom error code."""
        error_message = "Invalid input data"
        error_code = "VALIDATION_ERROR"
        response = self.json_handler.format_error_response(error_message, error_code)
        
        self.assertTrue(response["error"])
        self.assertEqual(response["error_code"], error_code)
        self.assertEqual(response["message"], error_message)
        self.assertIn("timestamp", response)
    
    def test_validate_response_schema_single_valid(self):
        """Test validating single response schema with valid data."""
        response = {"prediction": 250000.0}
        result = self.json_handler.validate_response_schema(response, "single")
        self.assertTrue(result)
    
    def test_validate_response_schema_single_invalid_missing_prediction(self):
        """Test validating single response schema with missing prediction."""
        response = {"error": "Invalid request"}
        result = self.json_handler.validate_response_schema(response, "single")
        self.assertFalse(result)
    
    def test_validate_response_schema_single_invalid_wrong_type(self):
        """Test validating single response schema with wrong prediction type."""
        response = {"prediction": "not_a_number"}
        result = self.json_handler.validate_response_schema(response, "single")
        self.assertFalse(result)
    
    def test_validate_response_schema_batch_valid(self):
        """Test validating batch response schema with valid data."""
        response = {
            "predictions": [250000.0, 300000.0],
            "count": 2
        }
        result = self.json_handler.validate_response_schema(response, "batch")
        self.assertTrue(result)
    
    def test_validate_response_schema_batch_invalid_missing_predictions(self):
        """Test validating batch response schema with missing predictions."""
        response = {"count": 2}
        result = self.json_handler.validate_response_schema(response, "batch")
        self.assertFalse(result)
    
    def test_validate_response_schema_batch_invalid_missing_count(self):
        """Test validating batch response schema with missing count."""
        response = {"predictions": [250000.0, 300000.0]}
        result = self.json_handler.validate_response_schema(response, "batch")
        self.assertFalse(result)
    
    def test_validate_response_schema_batch_invalid_wrong_predictions_type(self):
        """Test validating batch response schema with wrong predictions type."""
        response = {
            "predictions": "not_a_list",
            "count": 2
        }
        result = self.json_handler.validate_response_schema(response, "batch")
        self.assertFalse(result)
    
    def test_validate_response_schema_batch_invalid_wrong_count_type(self):
        """Test validating batch response schema with wrong count type."""
        response = {
            "predictions": [250000.0, 300000.0],
            "count": "not_a_number"
        }
        result = self.json_handler.validate_response_schema(response, "batch")
        self.assertFalse(result)
    
    def test_validate_response_schema_unknown_type(self):
        """Test validating response schema with unknown response type."""
        response = {"some_field": "some_value"}
        result = self.json_handler.validate_response_schema(response, "unknown")
        self.assertFalse(result)
    
    def test_validate_response_schema_batch_with_integer_predictions(self):
        """Test validating batch response schema with integer predictions."""
        response = {
            "predictions": [250000, 300000],
            "count": 2
        }
        result = self.json_handler.validate_response_schema(response, "batch")
        self.assertTrue(result)
    
    def test_validate_response_schema_single_with_integer_prediction(self):
        """Test validating single response schema with integer prediction."""
        response = {"prediction": 250000}
        result = self.json_handler.validate_response_schema(response, "single")
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
