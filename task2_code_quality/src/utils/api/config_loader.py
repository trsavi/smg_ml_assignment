"""
Configuration loader module for Madrid Housing Market API.

This module handles loading and managing API configuration from JSON files.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Any, Dict

# Local imports
from ..file_manager import FileManager

# Setup logging
logger = logging.getLogger(__name__)


class APIConfigLoader:
    """Handles API configuration loading and management."""
    
    def __init__(self, file_manager: FileManager) -> None:
        """
        Initialize the API config loader.
        
        Args:
            file_manager (FileManager): File manager instance for loading configs.
            
        Returns:
            None: Initializes the APIConfigLoader instance.
            
        Example:
            >>> loader = APIConfigLoader(file_manager)
        """
        self.file_manager = file_manager
        self.config: Dict[str, Any] = {}

    def load_config(self, config_path: str = "configs/api_config.json") -> Dict[str, Any]:
        """
        Load API configuration from JSON file.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            Dict[str, Any]: Loaded configuration dictionary.
            
        Raises:
            Exception: If configuration loading fails.
            
        Example:
            >>> config = loader.load_config("my_api_config.json")
        """
        try:
            self.config = self.file_manager.load_json(config_path)
            logger.info(f"API configuration loaded from {config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load API configuration: {e}")
            # Return default configuration if loading fails
            return self._get_default_config()

    def get_api_config(self) -> Dict[str, Any]:
        """
        Get API configuration.
        
        Args:
            None: Uses loaded configuration.
            
        Returns:
            Dict[str, Any]: API configuration dictionary.
            
        Example:
            >>> api_config = loader.get_api_config()
        """
        return self.config.get("api", {})

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dict[str, Any]: Model configuration dictionary.
        """
        return self.config.get("model", {})

    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get input schema configuration.

        Returns:
            Dict[str, Any]: Input schema dictionary.
        """
        return self.config.get("input_schema", {})

    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get output schema configuration.

        Returns:
            Dict[str, Any]: Output schema dictionary.
        """
        return self.config.get("output_schema", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.

        Returns:
            Dict[str, Any]: Logging configuration dictionary.
        """
        return self.config.get("logging", {})

    def get_request_handling_config(self) -> Dict[str, Any]:
        """
        Get request handling configuration.

        Returns:
            Dict[str, Any]: Request handling configuration dictionary.
        """
        return self.config.get("request_handling", {})

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if loading fails.

        Returns:
            Dict[str, Any]: Default configuration dictionary.
        """
        return {
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "title": "Madrid Housing Price Prediction API",
                "version": "1.0.0"
            },
            "model": {
                "path": "../models/madrid_housing_model.pkl"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            },
            "request_handling": {
                "save_requests": True,
                "request_dir": "json_requests",
                "timeout": 30
            }
        }
