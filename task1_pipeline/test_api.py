#!/usr/bin/env python3
"""
Simple test script for the Madrid Housing Price Prediction API.
"""

import requests
import json
import time

def test_api():
    """Test all API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing Madrid Housing Price Prediction API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test model info endpoint
    print("\n2. Testing /model/info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test predict endpoint
    print("\n3. Testing /predict endpoint...")
    test_data = {
        "sq_mt_built": 100.0,
        "n_rooms": 3,
        "n_bathrooms": 2,
        "is_new_development": False,
        "has_ac": True,
        "has_fitted_wardrobes": True,
        "has_pool": False,
        "has_terrace": True,
        "has_balcony": False,
        "has_storage_room": False,
        "is_accessible": False,
        "has_green_zones": True,
        "has_parking": False,
        "has_lift": 1.0,
        "is_exterior": 1.0,
        "house_type_id": "HouseType_1_Pisos",
        "district_id": 1
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    # Wait a moment for API to start
    print("Waiting for API to start...")
    time.sleep(3)
    test_api()
