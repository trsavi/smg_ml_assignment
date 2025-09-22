"""
API utilities package for Madrid Housing Market ML Pipeline.

This package contains all API-related utility modules including configuration,
model management, prediction services, and request/response handling.
"""

from .config_loader import APIConfigLoader
from .json_handler import JSONHandler
from .model_manager import ModelManager
from .models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    SinglePredictionResponse,
)
from .prediction_service import PredictionService

__all__ = [
    "APIConfigLoader",
    "JSONHandler",
    "ModelManager",
    "PredictionService",
    "PredictionRequest",
    "BatchPredictionRequest",
    "SinglePredictionResponse",
    "BatchPredictionResponse",
    "HealthResponse",
    "ErrorResponse",
]
