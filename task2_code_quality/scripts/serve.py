#!/usr/bin/env python3
"""
Refactored API serving script for Madrid Housing Market pipeline.

This script can:
- Start the FastAPI server (refactored version)
- Test specific API endpoints using test case files
- Load configuration from config files
"""

# Standard library imports
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Third-party imports
import requests

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Local imports
from utils.api import APIConfigLoader
from utils.file_manager import FileManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_api_config() -> dict:
    """
    Load API configuration from config file.
    
    Args:
        None: Uses default config file path.
        
    Returns:
        dict: API configuration dictionary.
        
    Example:
        >>> config = load_api_config()
        >>> print(config['api']['host'])
    """
    file_manager = FileManager()
    config_loader = APIConfigLoader(file_manager)
    return config_loader.load_config()


def start_api_server(host=None, port=None, use_refactored=True):
    """
    Start the FastAPI server.
    
    Args:
        host (str, optional): Host to bind to. If None, uses config default.
        port (int, optional): Port to bind to. If None, uses config default.
        use_refactored (bool): Whether to use refactored API.
        
    Returns:
        None: Starts the server process.
        
    Raises:
        SystemExit: If server startup fails.
        
    Example:
        >>> start_api_server(host="0.0.0.0", port=8080)
    """
    # Load configuration to get default values
    config = load_api_config()
    api_config = config.get("api", {})
    
    # Use provided values or fall back to config defaults
    host = host or api_config.get("host", "127.0.0.1")
    port = port or api_config.get("port", 8000)
    print(f"Starting API server on {host}:{port}...")
    
    # Choose which API to run
    api_file = "api.py"
    cmd = ["python", api_file]
    
    try:
        # Change to src directory for api.py
        src_dir = Path(__file__).parent.parent / "src"
        subprocess.run(cmd, cwd=src_dir, check=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def test_health_check(host=None, port=None):
    """
    Test the health check endpoint.
    
    Args:
        host (str, optional): API host. If None, uses config default.
        port (int, optional): API port. If None, uses config default.
        
    Returns:
        bool: True if health check passes, False otherwise.
        
    Example:
        >>> success = test_health_check()
        >>> print(f"Health check: {'PASSED' if success else 'FAILED'}")
    """
    # Load configuration to get default values
    config = load_api_config()
    api_config = config.get("api", {})
    host = host or api_config.get("host", "127.0.0.1")
    port = port or api_config.get("port", 8000)
    
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


def test_model_info(host=None, port=None):
    """
    Test the model info endpoint.
    
    Args:
        host (str, optional): API host. If None, uses config default.
        port (int, optional): API port. If None, uses config default.
        
    Returns:
        bool: True if model info endpoint works, False otherwise.
        
    Example:
        >>> success = test_model_info()
        >>> print(f"Model info: {'PASSED' if success else 'FAILED'}")
    """
    # Load configuration to get default values
    config = load_api_config()
    api_config = config.get("api", {})
    host = host or api_config.get("host", "127.0.0.1")
    port = port or api_config.get("port", 8000)
    
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


def test_predict(host=None, port=None):
    """
    Test the prediction endpoint using test_case_1.json.
    
    Args:
        host (str, optional): API host. If None, uses config default.
        port (int, optional): API port. If None, uses config default.
        
    Returns:
        bool: True if prediction endpoint works, False otherwise.
        
    Example:
        >>> success = test_predict()
        >>> print(f"Prediction: {'PASSED' if success else 'FAILED'}")
    """
    # Load configuration to get default values
    config = load_api_config()
    api_config = config.get("api", {})
    host = host or api_config.get("host", "127.0.0.1")
    port = port or api_config.get("port", 8000)
    
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
            return True
        else:
            print(f"Prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Prediction error: {e}")
        return False


def test_batch_predict(host=None, port=None):
    """
    Test the batch prediction endpoint using test_case_batch_prediction.json.
    
    Args:
        host (str, optional): API host. If None, uses config default.
        port (int, optional): API port. If None, uses config default.
        
    Returns:
        bool: True if batch prediction endpoint works, False otherwise.
        
    Example:
        >>> success = test_batch_predict()
        >>> print(f"Batch prediction: {'PASSED' if success else 'FAILED'}")
    """
    # Load configuration to get default values
    config = load_api_config()
    api_config = config.get("api", {})
    host = host or api_config.get("host", "127.0.0.1")
    port = port or api_config.get("port", 8000)
    
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
    """
    Main function to handle serve and test operations.
    
    This function provides a command-line interface for starting the API server
    or testing specific endpoints.
    
    Args:
        None: Uses command line arguments for configuration.
        
    Returns:
        None: Executes the requested operation.
        
    Example:
        >>> python serve.py start --host 0.0.0.0 --port 8080
        >>> python serve.py health_check
    """
    parser = argparse.ArgumentParser(
        description='Serve and test Madrid Housing Market API (Refactored)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start refactored API server
  %(prog)s start --legacy           # Start legacy API server
  %(prog)s health_check             # Test health check endpoint
  %(prog)s model_info               # Test model info endpoint
  %(prog)s predict                  # Test prediction endpoint
  %(prog)s batch_predict            # Test batch prediction endpoint
        """
    )
    
    parser.add_argument('action', choices=['start', 'health_check', 'model_info', 'predict', 'batch_predict'],
                       help='Action to perform: start server or test specific endpoint')
    # Load configuration to get default values for argument parser
    config = load_api_config()
    api_config = config.get("api", {})
    default_host = api_config.get("host", "127.0.0.1")
    default_port = api_config.get("port", 8000)
    
    parser.add_argument('--host', default=default_host,
                       help=f'Host to bind to (default: {default_host})')
    parser.add_argument('--port', type=int, default=default_port,
                       help=f'Port to bind to (default: {default_port})')
    parser.add_argument('--legacy', action='store_true',
                       help='Use legacy API instead of refactored version')
    
    args = parser.parse_args()
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    if args.action == 'start':
        start_api_server(args.host, args.port, use_refactored=not args.legacy)
    
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
