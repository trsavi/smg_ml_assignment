"""
Prediction service module for Madrid Housing Market API.

This module handles prediction logic, input validation, and response formatting
for the API service.
"""

# Standard library imports
import logging
from typing import Any, Dict, List

# Third-party imports
import pandas as pd

# Local imports
from .model_manager import ModelManager

# Setup logging
logger = logging.getLogger(__name__)


class PredictionService:
    """Handles prediction logic and data processing for the API service."""
    
    def __init__(self, model_manager: ModelManager) -> None:
        """
        Initialize the prediction service.

        Args:
            model_manager (ModelManager): Model manager instance for making predictions.
        """
        self.model_manager = model_manager

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data against expected schema.

        Args:
            data (Dict[str, Any]): Input data to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Basic validation - check if all required fields are present
        required_fields = [
            "sq_mt_built", "n_rooms", "n_bathrooms", "is_new_development",
            "has_ac", "has_fitted_wardrobes", "has_lift", "is_exterior",
            "has_pool", "has_terrace", "has_balcony", "has_storage_room",
            "is_accessible", "has_green_zones", "has_parking",
            "house_type_id_HouseType_1_Pisos", "house_type_id_HouseType_2_Casa_o_chalet",
            "house_type_id_HouseType_4_D_plex", "house_type_id_HouseType_5_ticos",
            "district_id_1", "district_id_2", "district_id_3", "district_id_4",
            "district_id_5", "district_id_6", "district_id_7", "district_id_8",
            "district_id_9", "district_id_10", "district_id_11", "district_id_12",
            "district_id_13", "district_id_14", "district_id_15", "district_id_17",
            "district_id_18", "district_id_19", "district_id_20"
        ]
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True

    def prepare_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input data for model prediction.

        Args:
            data (Dict[str, Any]): Input data dictionary.

        Returns:
            pd.DataFrame: Prepared data for model prediction.
        """
        return pd.DataFrame([data])

    def make_single_prediction(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Make a single prediction.

        Args:
            data (Dict[str, Any]): Input data for prediction.

        Returns:
            Dict[str, float]: Prediction result.

        Raises:
            ValueError: If model is not loaded or input is invalid.
            Exception: If prediction fails.
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("Model is not loaded")
        
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        try:
            # Prepare data for prediction
            X = self.prepare_data(data)
            
            # Make prediction
            prediction = self.model_manager.predict(X)[0]
            
            return {"prediction": float(prediction)}
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            raise

    def make_batch_prediction(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make batch predictions.

        Args:
            data_list (List[Dict[str, Any]]): List of input data for predictions.

        Returns:
            Dict[str, Any]: Batch prediction results.

        Raises:
            ValueError: If model is not loaded or input is invalid.
            Exception: If prediction fails.
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("Model is not loaded")
        
        if not data_list:
            raise ValueError("Empty data list provided")
        
        # Validate all inputs
        for i, data in enumerate(data_list):
            if not self.validate_input(data):
                raise ValueError(f"Invalid input data at index {i}")
        
        try:
            # Prepare data for prediction
            X = pd.DataFrame(data_list)
            
            # Make predictions
            predictions = self.model_manager.predict(X)
            
            return {
                "predictions": [float(p) for p in predictions],
                "count": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
