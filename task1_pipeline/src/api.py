"""
FastAPI application for Madrid Housing Market price prediction model serving.

This module provides REST API endpoints for model inference, health checks,
and batch predictions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
import uvicorn

from preprocessing import MadridHousingPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Madrid Housing Market Price Prediction API",
    description="API for predicting housing prices in Madrid using LightGBM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_info = {
    "model_name": "Madrid Housing Price Prediction",
    "version": "1.0.0",
    "algorithm": "LightGBM",
    "target": "buy_price",
    "features": []
}


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    sq_mt_built: float = Field(..., description="Built surface area in square meters", gt=0)
    n_rooms: int = Field(..., description="Number of rooms", ge=1)
    n_bathrooms: int = Field(..., description="Number of bathrooms", ge=1)
    house_type_id: str = Field(..., description="House type identifier")
    neighborhood_id: str = Field(..., description="Neighborhood identifier")
    has_ac: Optional[bool] = Field(False, description="Has air conditioning")
    has_fitted_wardrobes: Optional[bool] = Field(False, description="Has fitted wardrobes")
    has_pool: Optional[bool] = Field(False, description="Has pool")
    has_terrace: Optional[bool] = Field(False, description="Has terrace")
    has_balcony: Optional[bool] = Field(False, description="Has balcony")
    has_storage_room: Optional[bool] = Field(False, description="Has storage room")
    is_accessible: Optional[bool] = Field(False, description="Is accessible")
    has_green_zones: Optional[bool] = Field(False, description="Has green zones")
    is_new_development: Optional[bool] = Field(False, description="Is new development")
    has_lift: Optional[float] = Field(None, description="Has lift (0 or 1)")
    is_exterior: Optional[float] = Field(None, description="Is exterior (0 or 1)")
    built_year: Optional[int] = Field(None, description="Year built")
    
    @validator('sq_mt_built')
    def validate_sq_mt_built(cls, v):
        if v <= 0:
            raise ValueError('sq_mt_built must be positive')
        if v > 10000:
            raise ValueError('sq_mt_built seems unreasonably large (>10000 sqm)')
        return v
    
    @validator('n_rooms')
    def validate_n_rooms(cls, v):
        if v < 1:
            raise ValueError('n_rooms must be at least 1')
        if v > 20:
            raise ValueError('n_rooms seems unreasonably high (>20)')
        return v
    
    @validator('n_bathrooms')
    def validate_n_bathrooms(cls, v):
        if v < 1:
            raise ValueError('n_bathrooms must be at least 1')
        if v > 10:
            raise ValueError('n_bathrooms seems unreasonably high (>10)')
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    prediction: float = Field(..., description="Predicted price in euros")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    data: List[Dict[str, Any]] = Field(..., description="List of property data for prediction")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError('data cannot be empty')
        if len(v) > 1000:
            raise ValueError('batch size cannot exceed 1000 predictions')
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[float] = Field(..., description="List of predicted prices")
    count: int = Field(..., description="Number of predictions made")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


def load_model(model_path: str = "models/madrid_housing_model.pkl"):
    """Load the trained model and preprocessor."""
    global model, preprocessor, model_info
    
    try:
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load preprocessor
        preprocessor_path = str(Path(model_path).parent / "preprocessor.pkl")
        preprocessor = MadridHousingPreprocessor()
        preprocessor.load_pipeline(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        
        # Update model info
        model_info["features"] = preprocessor.get_feature_names()
        model_info["loaded_at"] = datetime.now().isoformat()
        
        logger.info("Model and preprocessor loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version=model_info["version"]
    )


@app.get("/model/info")
async def model_info_endpoint():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        **model_info,
        "feature_count": len(model_info["features"]),
        "model_type": type(model).__name__
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single price prediction."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        data = request.dict()
        df = pd.DataFrame([data])
        
        # Preprocess data
        X_transformed = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(X_transformed)[0]
        
        # Calculate confidence (simple approach using prediction variance)
        # In a real scenario, you might use prediction intervals or model uncertainty
        confidence = min(0.95, max(0.5, 1.0 - abs(prediction) / 1000000))
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make batch price predictions."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        required_columns = ['sq_mt_built', 'n_rooms', 'n_bathrooms', 'house_type_id', 'neighborhood_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Preprocess data
        X_transformed = preprocessor.transform(df)
        
        # Make predictions
        predictions = model.predict(X_transformed)
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            count=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Madrid Housing Market Price Prediction API",
        "version": model_info["version"],
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info"
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
