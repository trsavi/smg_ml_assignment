"""
Refactored FastAPI service for Madrid Housing Market price prediction.

This module contains only API route definitions and request/response handling.
All business logic is separated into dedicated utility modules for better maintainability.

Endpoints:
- GET /health -> service status
- GET /model/info -> model metadata
- POST /predict -> make single prediction
- POST /batch_predict -> make batch predictions
"""

# Standard library imports
import logging
from typing import Any, Dict

# Third-party imports
import uvicorn
from fastapi import FastAPI, HTTPException

# Local imports
from utils.api import (
    APIConfigLoader,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    JSONHandler,
    ModelManager,
    PredictionRequest,
    PredictionService,
    SinglePredictionResponse,
)
from utils.file_manager import FileManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
file_manager = FileManager()
config_loader = APIConfigLoader(file_manager)
config = config_loader.load_config()

# Initialize services
model_manager = ModelManager(file_manager)
prediction_service = PredictionService(model_manager)
json_handler = JSONHandler(file_manager, config)

# Create FastAPI app
api_config = config_loader.get_api_config()
app = FastAPI(
    title=api_config.get("title", "Madrid Housing Price Prediction API"),
    version=api_config.get("version", "1.0.0"),
    description=api_config.get("description", "API for predicting Madrid housing market prices")
)


@app.on_event("startup")
async def startup_event() -> None:
    """
    Load model when app starts.

    Returns:
        None: Model is loaded into the model manager.
    """
    model_config = config_loader.get_model_config()
    model_path = model_config.get("path", "../models/madrid_housing_model.pkl")
    
    try:
        model_manager.load_model(model_path)
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse: Status information including model loaded state.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_model_loaded()
    )


@app.get("/model/info")
async def model_info_endpoint() -> Dict[str, Any]:
    """
    Get model information endpoint.

    Returns:
        Dict[str, Any]: Model metadata and information.

    Raises:
        HTTPException: If model is not loaded.
    """
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_manager.get_model_info()


@app.post("/predict", response_model=SinglePredictionResponse)
async def predict(request: PredictionRequest) -> SinglePredictionResponse:
    """
    Make single prediction endpoint.

    Args:
        request (PredictionRequest): Single prediction request data.

    Returns:
        SinglePredictionResponse: Prediction result.

    Raises:
        HTTPException: If model is not loaded or prediction fails.
    """
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic model to dict
        request_data = request.dict()
        
        # Save request for debugging
        json_handler.save_request(request_data, "single")
        
        # Make prediction
        result = prediction_service.make_single_prediction(request_data)
        
        return SinglePredictionResponse(prediction=result["prediction"])
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Make batch predictions endpoint.

    Args:
        request (BatchPredictionRequest): Batch prediction request data.

    Returns:
        BatchPredictionResponse: Batch prediction results.

    Raises:
        HTTPException: If model is not loaded or batch prediction fails.
    """
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic models to list of dicts
        batch_data = [item.dict() for item in request.data]
        
        # Save batch request for debugging
        json_handler.save_request({"batch_data": batch_data, "count": len(batch_data)}, "batch")
        
        # Make batch predictions
        result = prediction_service.make_batch_prediction(batch_data)
        
        return BatchPredictionResponse(
            predictions=result["predictions"],
            count=result["count"]
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    # Get API configuration
    api_config = config_loader.get_api_config()
    host = api_config.get("host", "127.0.0.1")
    port = api_config.get("port", 8000)
    
    uvicorn.run("api:app", host=host, port=port, reload=True)
