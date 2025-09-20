#!/usr/bin/env python3
"""
Test runner script for Madrid Housing Market ML Pipeline.

This script runs all tests with coverage reporting and ensures
minimum 80% code coverage requirement is met.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """Run all tests with coverage reporting."""
    print("🧪 Running Madrid Housing Market ML Pipeline Tests")
    print("=" * 60)
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Run pytest with coverage
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-fail-under=80",
        "-v",
        "--tb=short"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
        print("📈 XML coverage report generated in coverage.xml")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False


def run_specific_test_type(test_type):
    """Run specific type of tests."""
    print(f"🧪 Running {test_type} tests")
    print("=" * 40)
    
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    cmd = [
        "python", "-m", "pytest",
        f"tests/test_{test_type}.py",
        "--cov=src",
        "--cov-report=term-missing",
        "-v"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ {test_type} tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {test_type} tests failed with exit code {e.returncode}")
        return False


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type in ["data_loader", "preprocessing", "train_model", "api_integration"]:
            success = run_specific_test_type(test_type)
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: data_loader, preprocessing, train_model, api_integration")
            sys.exit(1)
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
