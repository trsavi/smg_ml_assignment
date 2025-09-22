"""
JSON handling module for Madrid Housing Market API.

This module handles request/response JSON operations, including saving requests
for debugging and testing purposes.
"""

# Standard library imports
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Local imports
from ..file_manager import FileManager

# Setup logging
logger = logging.getLogger(__name__)


class JSONHandler:
    """Handles JSON operations for request/response processing."""
    
    def __init__(self, file_manager: FileManager, config: Dict[str, Any]) -> None:
        """
        Initialize the JSON handler.

        Args:
            file_manager (FileManager): File manager instance for saving files.
            config (Dict[str, Any]): API configuration dictionary.
        """
        self.file_manager = file_manager
        self.config = config
        self.request_dir = Path(config.get("request_handling", {}).get("request_dir", "json_requests"))
        self.save_requests = config.get("request_handling", {}).get("save_requests", True)

    def save_request(self, request_data: Dict[str, Any], request_type: str = "single") -> str:
        """
        Save raw JSON request for debugging/testing.

        Args:
            request_data (Dict[str, Any]): Request data to save as JSON.
            request_type (str): Type of request (single, batch).

        Returns:
            str: Path to the saved JSON file.
        """
        if not self.save_requests:
            return ""
        
        try:
            # Create request directory if it doesn't exist
            self.request_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"request_{request_type}_{timestamp}.json"
            filepath = self.request_dir / filename
            
            # Save request data
            self.file_manager.save_json(request_data, str(filepath))
            logger.info(f"Request saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save request: {e}")
            return ""

    def format_single_response(self, prediction: float, request_id: str = "") -> Dict[str, Any]:
        """
        Format single prediction response.

        Args:
            prediction (float): Prediction value.
            request_id (str): Optional request ID for tracking.

        Returns:
            Dict[str, Any]: Formatted response.
        """
        response = {"prediction": prediction}
        
        if request_id:
            response["request_id"] = request_id
            
        return response

    def format_batch_response(self, predictions: list, count: int, request_id: str = "") -> Dict[str, Any]:
        """
        Format batch prediction response.

        Args:
            predictions (list): List of prediction values.
            count (int): Number of predictions.
            request_id (str): Optional request ID for tracking.

        Returns:
            Dict[str, Any]: Formatted response.
        """
        response = {
            "predictions": predictions,
            "count": count
        }
        
        if request_id:
            response["request_id"] = request_id
            
        return response

    def format_error_response(self, error_message: str, error_code: str = "PREDICTION_ERROR") -> Dict[str, Any]:
        """
        Format error response.

        Args:
            error_message (str): Error message.
            error_code (str): Error code for categorization.

        Returns:
            Dict[str, Any]: Formatted error response.
        """
        return {
            "error": True,
            "error_code": error_code,
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }

    def validate_response_schema(self, response: Dict[str, Any], response_type: str = "single") -> bool:
        """
        Validate response against expected schema.

        Args:
            response (Dict[str, Any]): Response to validate.
            response_type (str): Type of response (single, batch).

        Returns:
            bool: True if valid, False otherwise.
        """
        if response_type == "single":
            return "prediction" in response and isinstance(response["prediction"], (int, float))
        elif response_type == "batch":
            return ("predictions" in response and 
                   "count" in response and 
                   isinstance(response["predictions"], list) and
                   isinstance(response["count"], int))
        
        return False
