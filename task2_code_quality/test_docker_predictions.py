#!/usr/bin/env python3
"""
Docker Prediction Testing Script for Madrid Housing ML Pipeline
Tests all prediction endpoints after Docker container is running
"""

# Standard library imports
import json
import subprocess
import sys
import time
from pathlib import Path

# Third-party imports
import requests

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
CONTAINER_NAME = "task1_pipeline-madrid-housing-api-1"


def test_health_endpoint():
    """Test the health endpoint"""
    print("\nTesting health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("SUCCESS: Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"ERROR: Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        if response.status_code == 200:
            print("SUCCESS: Model info retrieved")
            info = response.json()
            print(f"   Model type: {info.get('model_type', 'Unknown')}")
            print(f"   Features: {len(info.get('features', []))}")
            return True
        else:
            print(f"ERROR: Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Model info error: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\nTesting single prediction...")
    
    # Load test case
    test_file = Path("api_test_cases/test_case_1.json")
    if not test_file.exists():
        print("ERROR: Test case file not found")
        return False
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Single prediction successful")
            predicted_price = result.get('prediction', 'N/A')
            if isinstance(predicted_price, (int, float)):
                print(f"   Predicted price: €{predicted_price:,.2f}")
            else:
                print(f"   Predicted price: {predicted_price}")
            return True
        else:
            print(f"ERROR: Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Single prediction error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\nTesting batch prediction...")
    
    # Load batch test case
    test_file = Path("api_test_cases/test_case_batch_prediction.json")
    if not test_file.exists():
        print("ERROR: Batch test case file not found")
        return False
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch_predict",
            json=test_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Batch prediction successful")
            predictions = result.get('predictions', [])
            print(f"   Predictions count: {len(predictions)}")
            if predictions:
                avg_price = sum(predictions) / len(predictions)
                print(f"   Average price: €{avg_price:,.2f}")
                print(f"   Price range: €{min(predictions):,.2f} - €{max(predictions):,.2f}")
            return True
        else:
            print(f"ERROR: Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Batch prediction error: {e}")
        return False

def test_custom_prediction():
    """Test with custom prediction data"""
    print("\nTesting custom prediction...")
    
    # Create a simple test case
    custom_data = {
        "sq_mt_built": 80.0,
        "n_rooms": 3,
        "n_bathrooms": 2.0,
        "is_new_development": True,
        "has_ac": True,
        "has_fitted_wardrobes": True,
        "has_lift": 1.0,
        "is_exterior": 1.0,
        "has_pool": False,
        "has_terrace": True,
        "has_balcony": True,
        "has_storage_room": True,
        "is_accessible": True,
        "has_green_zones": True,
        "has_parking": True,
        "house_type_id_HouseType_1_Pisos": True,
        "house_type_id_HouseType_2_Casa_o_chalet": False,
        "house_type_id_HouseType_4_D_plex": False,
        "house_type_id_HouseType_5_ticos": False,
        "district_id_1": True,  # Centro
        "district_id_10": False,
        "district_id_11": False,
        "district_id_12": False,
        "district_id_13": False,
        "district_id_14": False,
        "district_id_15": False,
        "district_id_17": False,
        "district_id_18": False,
        "district_id_19": False,
        "district_id_2": False,
        "district_id_20": False,
        "district_id_3": False,
        "district_id_4": False,
        "district_id_5": False,
        "district_id_6": False,
        "district_id_7": False,
        "district_id_8": False,
        "district_id_9": False
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=custom_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Custom prediction successful")
            predicted_price = result.get('prediction', 'N/A')
            if isinstance(predicted_price, (int, float)):
                print(f"   Predicted price: €{predicted_price:,.2f}")
            else:
                print(f"   Predicted price: {predicted_price}")
            print(f"   Property: {custom_data['sq_mt_built']}m², {custom_data['n_rooms']} rooms")
            return True
        else:
            print(f"ERROR: Custom prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Custom prediction error: {e}")
        return False

def main():
    """Run all prediction tests"""
    print("Madrid Housing ML Pipeline - Docker Prediction Tests")
    print("=" * 60)
    
    # Wait a moment for API to be ready
    print("\nWaiting for API to be ready...")
    time.sleep(5)
    
    # Run all tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Custom Prediction", test_custom_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"ERROR: {test_name} crashed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Your Docker container is working perfectly!")
        print("\nAPI is available at: http://localhost:8000")
        print("API docs at: http://localhost:8000/docs")
    else:
        print("WARNING: Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
