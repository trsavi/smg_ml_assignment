#!/usr/bin/env python3
"""
Test the simplified API with the existing test data.
"""

import requests
import json
import time
from pathlib import Path

def test_api():
    """Test the simplified API."""
    base_url = "http://localhost:8000"
    
    print("Testing Simplified Madrid Housing Price Prediction API")
    print("=" * 60)
    
    # Wait for API to start
    print("Waiting for API to start...")
    time.sleep(5)
    
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
    
    # Test predict endpoint with existing test data
    print("\n3. Testing /predict endpoint...")
    
    # Use one of the existing test cases
    test_case_path = Path("test_cases/test_case_1.json")
    if test_case_path.exists():
        with open(test_case_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"Using test data from {test_case_path}")
        print(f"Test data keys: {list(test_data.keys())}")
        
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
    else:
        print("No test case file found, creating simple test data...")
        
        # Create simple test data
        simple_test_data = {
            "sq_mt_built": 100.0,
            "n_rooms": 3.0,
            "n_bathrooms": 2.0,
            "is_new_development": False,
            "has_ac": True,
            "has_fitted_wardrobes": True,
            "has_lift": 1.0,
            "is_exterior": 1.0,
            "has_pool": False,
            "has_terrace": True,
            "has_balcony": False,
            "has_storage_room": False,
            "is_accessible": False,
            "has_green_zones": True,
            "has_parking": False,
            "house_type_id_HouseType_1_Pisos": True,
            "house_type_id_HouseType_2_Casa_o_chalet": False,
            "house_type_id_HouseType_4_D_plex": False,
            "house_type_id_HouseType_5_ticos": False,
            "district_id_1": True,
            "district_id_2": False,
            "district_id_3": False,
            "district_id_4": False,
            "district_id_5": False,
            "district_id_6": False,
            "district_id_7": False,
            "district_id_8": False,
            "district_id_9": False,
            "district_id_10": False,
            "district_id_11": False,
            "district_id_12": False,
            "district_id_13": False,
            "district_id_14": False,
            "district_id_15": False,
            "district_id_17": False,
            "district_id_18": False,
            "district_id_19": False,
            "district_id_20": False
        }
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=simple_test_data,
                headers={"Content-Type": "application/json"}
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("API testing completed!")

if __name__ == "__main__":
    test_api()
