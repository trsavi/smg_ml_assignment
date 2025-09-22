"""
Pydantic models for Madrid Housing Market API.

This module defines the request and response models for the API endpoints.
"""

# Third-party imports
from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    """Request model for single prediction endpoint."""
    sq_mt_built: float
    n_rooms: float
    n_bathrooms: float
    is_new_development: bool
    has_ac: bool
    has_fitted_wardrobes: bool
    has_lift: float
    is_exterior: float
    has_pool: bool
    has_terrace: bool
    has_balcony: bool
    has_storage_room: bool
    is_accessible: bool
    has_green_zones: bool
    has_parking: bool
    house_type_id_HouseType_1_Pisos: bool
    house_type_id_HouseType_2_Casa_o_chalet: bool
    house_type_id_HouseType_4_D_plex: bool
    house_type_id_HouseType_5_ticos: bool
    district_id_1: bool
    district_id_2: bool
    district_id_3: bool
    district_id_4: bool
    district_id_5: bool
    district_id_6: bool
    district_id_7: bool
    district_id_8: bool
    district_id_9: bool
    district_id_10: bool
    district_id_11: bool
    district_id_12: bool
    district_id_13: bool
    district_id_14: bool
    district_id_15: bool
    district_id_17: bool
    district_id_18: bool
    district_id_19: bool
    district_id_20: bool


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    data: List[PredictionRequest]


class SinglePredictionResponse(BaseModel):
    """Response model for single prediction endpoint."""
    prediction: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    predictions: List[float]
    count: int


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    error: bool
    error_code: str
    message: str
    timestamp: str
