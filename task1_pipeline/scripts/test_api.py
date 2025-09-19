#!/usr/bin/env python3
"""
API testing script for Madrid Housing Market pipeline.

This script tests the API endpoints to ensure they're working correctly.
"""

import requests
import json
import time
import sys

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_model_info_endpoint():
    """Test the model info endpoint."""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get("http://localhost:8000/model/info")
        if response.status_code == 200:
            print("✅ Model info endpoint passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")

def test_predict_endpoint():
    """Test the prediction endpoint."""
    print("\nTesting prediction endpoint...")
    try:
        payload = {
            "sq_mt_built": 100,
            "n_rooms": 3,
            "n_bathrooms": 2,
            "house_type_id": "HouseType 1: Piso",
            "neighborhood_id": "Neighborhood 1",
            "has_ac": True,
            "has_terrace": False
        }
        
        response = requests.post(
            "http://localhost:8000/predict",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            print("✅ Prediction endpoint passed")
            result = response.json()
            print(f"   Predicted price: €{result['prediction']:,.2f}")
            print(f"   Confidence: {result['confidence']:.2f}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")

def test_batch_predict_endpoint():
    """Test the batch prediction endpoint."""
    print("\nTesting batch prediction endpoint...")
    try:
        payload = {
            "data": [
                {
                    "sq_mt_built": 100,
                    "n_rooms": 3,
                    "n_bathrooms": 2,
                    "house_type_id": "HouseType 1: Piso",
                    "neighborhood_id": "Neighborhood 1"
                },
                {
                    "sq_mt_built": 150,
                    "n_rooms": 4,
                    "n_bathrooms": 3,
                    "house_type_id": "HouseType 2: Casa o chalet",
                    "neighborhood_id": "Neighborhood 2"
                }
            ]
        }
        
        response = requests.post(
            "http://localhost:8000/batch_predict",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            print("✅ Batch prediction endpoint passed")
            result = response.json()
            print(f"   Predictions: {len(result['predictions'])}")
            for i, pred in enumerate(result['predictions']):
                print(f"   Property {i+1}: €{pred:,.2f}")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")

def main():
    """Run all API tests."""
    print("🧪 Testing Madrid Housing Market API")
    print("=" * 50)
    
    # Wait a moment for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    # Run tests
    test_health_endpoint()
    test_model_info_endpoint()
    test_predict_endpoint()
    test_batch_predict_endpoint()
    
    print("\n" + "=" * 50)
    print("API testing completed!")

if __name__ == "__main__":
    main()
