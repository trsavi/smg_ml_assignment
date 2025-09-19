"""
Minimal FastAPI service for Madrid Housing Market price prediction.

Endpoints:
- GET /health -> service status
- GET /model/info -> model metadata
- POST /predict -> make single prediction and save request JSON
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Madrid Housing Price Prediction API", version="1.0.0")

# Global model
model = None
model_info = {}


class PredictionRequest(BaseModel):
    """Request model matches model input features exactly."""
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


def load_model(model_path: str = "models/madrid_housing_model.pkl"):
    """Load trained model from file and extract metadata."""
    global model, model_info
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Extract model metadata
        model_info = {
            "model_name": "Madrid Housing Price Prediction",
            "version": "1.0.0",
            "model_type": type(model).__name__,
            "algorithm": "LightGBM",
            "n_features": len(model.feature_name_) if hasattr(model, 'feature_name_') else 0,
            "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None,
            "learning_rate": model.learning_rate if hasattr(model, 'learning_rate') else None,
            "max_depth": model.max_depth if hasattr(model, 'max_depth') else None,
            "num_leaves": model.num_leaves if hasattr(model, 'num_leaves') else None,
            "objective": model.objective if hasattr(model, 'objective') else None,
            "random_state": model.random_state if hasattr(model, 'random_state') else None,
            "feature_names": model.feature_name_ if hasattr(model, 'feature_name_') else [],
            "loaded_at": datetime.now().isoformat(),
            "model_file": model_path
        }
        
        logger.info(f"Model metadata extracted: {model_info['model_type']} with {model_info['n_features']} features")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def save_request_json(request_data: Dict[str, Any]) -> str:
    """Save raw JSON request for debugging/testing."""
    requests_dir = Path("json_requests")
    requests_dir.mkdir(exist_ok=True)
    filename = f"request_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = requests_dir / filename
    with open(filepath, "w") as f:
        json.dump(request_data, f, indent=2)
    logger.info(f"Request saved: {filepath}")
    return str(filepath)


@app.on_event("startup")
async def startup_event():
    """Load model when app starts."""
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model/info")
async def model_info_endpoint():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info


@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_data = request.dict()
    save_request_json(request_data)

    try:
        X = pd.DataFrame([request_data])
        prediction = model.predict(X)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
