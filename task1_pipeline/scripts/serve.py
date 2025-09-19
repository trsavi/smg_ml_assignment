#!/usr/bin/env python3
"""
API serving script for Madrid Housing Market pipeline.

This script can:
- Start the FastAPI server
- Test specific API endpoints using test case files
"""

import argparse
import subprocess
import sys
import time
import requests
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def start_api_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server."""
    print(f"Starting API server on {host}:{port}...")
    
    cmd = ["python", "api.py"]
    
    try:
        # Change to src directory for api.py
        src_dir = Path(__file__).parent.parent / "src"
        subprocess.run(cmd, cwd=src_dir, check=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def test_health_check(host="localhost", port=8000):
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"http://{host}:{port}/health")
        if response.status_code == 200:
            print("Health check passed")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False


def test_model_info(host="localhost", port=8000):
    """Test the model info endpoint."""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"http://{host}:{port}/model/info")
        if response.status_code == 200:
            print("Model info endpoint passed")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Model info error: {e}")
        return False


def test_predict(host="localhost", port=8000):
    """Test the prediction endpoint using test_case_1.json."""
    print("\nTesting prediction endpoint...")
    
    # Load test case
    test_file = Path(__file__).parent.parent / "api_test_cases" / "test_case_1.json"
    
    try:
        with open(test_file, 'r') as f:
            payload = json.load(f)
        
        response = requests.post(
            f"http://{host}:{port}/predict",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            print("Prediction endpoint passed")
            result = response.json()
            print(f"Predicted price: €{result['prediction']:,.2f}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.2f}")
            return True
        else:
            print(f"Prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Prediction error: {e}")
        return False


def test_batch_predict(host="localhost", port=8000):
    """Test the batch prediction endpoint using test_case_batch_prediction.json."""
    print("\nTesting batch prediction endpoint...")
    
    # Load test case
    test_file = Path(__file__).parent.parent / "api_test_cases" / "test_case_batch_prediction.json"
    
    try:
        with open(test_file, 'r') as f:
            payload = json.load(f)
        
        response = requests.post(
            f"http://{host}:{port}/batch_predict",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            print("Batch prediction endpoint passed")
            result = response.json()
            print(f"Predictions: {len(result['predictions'])}")
            for i, pred in enumerate(result['predictions']):
                print(f"Property {i+1}: €{pred:,.2f}")
            return True
        else:
            print(f"Batch prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return False


def main():
    """Main function to handle serve and test operations."""
    parser = argparse.ArgumentParser(
        description='Serve and test Madrid Housing Market API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start API server
  %(prog)s health_check             # Test health check endpoint
  %(prog)s model_info               # Test model info endpoint
  %(prog)s predict                  # Test prediction endpoint
  %(prog)s batch_predict            # Test batch prediction endpoint
        """
    )
    
    parser.add_argument('action', choices=['start', 'health_check', 'model_info', 'predict', 'batch_predict'],
                       help='Action to perform: start server or test specific endpoint')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to (default: 8000)')
    
    args = parser.parse_args()
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    if args.action == 'start':
        start_api_server(args.host, args.port)
    
    elif args.action == 'health_check':
        test_health_check(args.host, args.port)
    
    elif args.action == 'model_info':
        test_model_info(args.host, args.port)
    
    elif args.action == 'predict':
        test_predict(args.host, args.port)
    
    elif args.action == 'batch_predict':
        test_batch_predict(args.host, args.port)


if __name__ == '__main__':
    main()