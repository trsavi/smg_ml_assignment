#!/usr/bin/env python3
"""
Test runner script for unittest-based tests.

This script demonstrates how to run the unittest tests for the train_model.py module
with different configurations and output options.
"""

# Standard library imports
import os
import sys
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_all_tests():
    """Run all unittest tests."""
    print("Running all unittest tests...")
    print("=" * 60)
    
    # Discover and run all tests in the tests/unittest directory
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests', 'unittest')
    suite = loader.discover(start_dir, pattern='test_*unittest.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    return result

def run_specific_test_class():
    """Run a specific test class."""
    print("Running specific test class...")
    print("=" * 60)
    
    # Import the specific test
    from tests.unittest.test_train_model_unittest import TestMadridHousingTrainer
    
    # Create test suite for specific class
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMadridHousingTrainer)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_specific_test_method():
    """Run a specific test method."""
    print("Running specific test method...")
    print("=" * 60)
    
    # Import the specific test
    from tests.unittest.test_train_model_unittest import TestMadridHousingTrainer
    
    # Create test suite for specific method
    suite = unittest.TestSuite()
    suite.addTest(TestMadridHousingTrainer('test_init_default_config'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_with_coverage():
    """Run tests with coverage reporting (if available)."""
    try:
        import coverage
        
        print("Running tests with coverage...")
        print("=" * 60)
        
        # Start coverage
        cov = coverage.Coverage(source=['src'])
        cov.start()
        
        # Run tests
        result = run_all_tests()
        
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        print("\nCoverage Report:")
        print("-" * 40)
        cov.report()
        
        # Generate HTML report
        cov.html_report(directory='htmlcov_unittest')
        print(f"\nHTML coverage report generated in 'htmlcov_unittest' directory")
        
        return result
        
    except ImportError:
        print("Coverage module not available. Running tests without coverage...")
        return run_all_tests()

def print_test_summary(result):
    """Print a detailed test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"FAIL: {test}")
            print(f"      {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
            print(f"       {traceback.split('Exception:')[-1].strip()}")
    
    print("=" * 60)
    return result.wasSuccessful()

def main():
    """Main function to run different test configurations."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = "all"
    
    print("Unittest Test Runner for Madrid Housing Trainer")
    print("=" * 60)
    
    if test_type == "all":
        result = run_all_tests()
    elif test_type == "class":
        result = run_specific_test_class()
    elif test_type == "method":
        result = run_specific_test_method()
    elif test_type == "coverage":
        result = run_with_coverage()
    else:
        print(f"Unknown test type: {test_type}")
        print("Available options: all, class, method, coverage")
        sys.exit(1)
    
    success = print_test_summary(result)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
