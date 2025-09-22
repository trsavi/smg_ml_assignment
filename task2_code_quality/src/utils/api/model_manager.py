"""
Model management module for Madrid Housing Market API.

This module handles model loading, metadata extraction, and model state management
for the API service.
"""

# Standard library imports
import logging
from datetime import datetime
from typing import Any, Dict, Optional

# Third-party imports
import joblib

# Local imports
from ..file_manager import FileManager

# Setup logging
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and metadata for the API service."""
    
    def __init__(self, file_manager: FileManager) -> None:
        """
        Initialize the model manager.

        Args:
            file_manager (FileManager): File manager instance for loading models.
        """
        self.file_manager = file_manager
        self.model: Optional[Any] = None
        self.model_info: Dict[str, Any] = {}
        self.is_loaded: bool = False

    def load_model(self, model_path: str) -> None:
        """
        Load trained model from file and extract metadata.

        Args:
            model_path (str): Path to the trained model file.

        Raises:
            Exception: If model loading fails.
        """
        try:
            self.model = self.file_manager.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Extract model metadata
            self.model_info = {
                "model_name": "Madrid Housing Price Prediction",
                "version": "1.0.0",
                "model_type": type(self.model).__name__,
                "algorithm": "LightGBM",
                "n_features": len(self.model.feature_name_) if hasattr(self.model, 'feature_name_') else 0,
                "n_estimators": self.model.n_estimators if hasattr(self.model, 'n_estimators') else None,
                "learning_rate": self.model.learning_rate if hasattr(self.model, 'learning_rate') else None,
                "max_depth": self.model.max_depth if hasattr(self.model, 'max_depth') else None,
                "num_leaves": self.model.num_leaves if hasattr(self.model, 'num_leaves') else None,
                "objective": self.model.objective if hasattr(self.model, 'objective') else None,
                "random_state": self.model.random_state if hasattr(self.model, 'random_state') else None,
                "feature_names": self.model.feature_name_ if hasattr(self.model, 'feature_name_') else [],
                "loaded_at": datetime.now().isoformat(),
                "model_file": model_path
            }
            
            self.is_loaded = True
            logger.info(f"Model metadata extracted: {self.model_info['model_type']} with {self.model_info['n_features']} features")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            raise

    def get_model(self) -> Optional[Any]:
        """
        Get the loaded model.

        Returns:
            Optional[Any]: The loaded model or None if not loaded.
        """
        return self.model

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dict[str, Any]: Model metadata dictionary.
        """
        return self.model_info

    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            bool: True if model is loaded, False otherwise.
        """
        return self.is_loaded

    def predict(self, data: Any) -> Any:
        """
        Make predictions using the loaded model.

        Args:
            data: Input data for prediction.

        Returns:
            Any: Model predictions.

        Raises:
            ValueError: If model is not loaded.
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded")
        
        return self.model.predict(data)
