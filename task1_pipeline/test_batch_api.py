#!/usr/bin/env python3
"""
Test script for batch prediction API endpoint.
"""

import requests
import json
import time
from pathlib import Path

def test_batch_api():
    """Test the batch prediction API."""
    base_url = "http://localhost:8000"
    
    print("Testing Batch Prediction API")
    print("=" * 40)
    
    # Wait for API to start
    print("Waiting for API to start...")
    time.sleep(3)
    
    # Test health endpoint first
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test batch prediction with multiple test cases
    print("\n2. Testing /batch_predict endpoint...")
    
    # Load test cases from existing files
    test_cases = []
    test_cases_dir = Path("test_cases")
    
    if test_cases_dir.exists():
        for i in range(1, 4):  # Use first 3 test cases
            test_file = test_cases_dir / f"test_case_{i}.json"
            if test_file.exists():
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                    test_cases.append(test_data)
                    print(f"Loaded test case {i}")
    
    if not test_cases:
        print("No test cases found, creating sample data...")
        # Create sample batch data
        test_cases = [
            {
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
            },
            {
                "sq_mt_built": 150.0,
                "n_rooms": 4.0,
                "n_bathrooms": 3.0,
                "is_new_development": True,
                "has_ac": False,
                "has_fitted_wardrobes": False,
                "has_lift": 0.0,
                "is_exterior": 0.0,
                "has_pool": True,
                "has_terrace": False,
                "has_balcony": True,
                "has_storage_room": True,
                "is_accessible": True,
                "has_green_zones": False,
                "has_parking": True,
                "house_type_id_HouseType_1_Pisos": False,
                "house_type_id_HouseType_2_Casa_o_chalet": True,
                "house_type_id_HouseType_4_D_plex": False,
                "house_type_id_HouseType_5_ticos": False,
                "district_id_1": False,
                "district_id_2": True,
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
        ]
    
    # Prepare batch request
    batch_request = {"data": test_cases}
    
    try:
        response = requests.post(
            f"{base_url}/batch_predict",
            json=batch_request,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if "predictions" in result:
            print(f"\nPredictions summary:")
            for i, pred in enumerate(result["predictions"]):
                print(f"  Property {i+1}: €{pred:,.2f}")
                
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 40)
    print("Batch API testing completed!")

if __name__ == "__main__":
    test_batch_api()
