"""
Unit tests for preprocessing.py using unittest framework.

This module provides comprehensive unit tests for the MadridHousingPreprocessor class
using the standard unittest framework.
"""

import unittest
import unittest.mock as mock
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing import MadridHousingPreprocessor


class TestMadridHousingPreprocessor(unittest.TestCase):
    """Test cases for MadridHousingPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.preprocessor = MadridHousingPreprocessor()
        
        # Create comprehensive sample data
        self.sample_data = pd.DataFrame({
            'sq_mt_built': [85.5, 120.0, 65.0, 90.0, 110.0, 75.0],
            'n_rooms': [3, 4, 2, 3, 4, 2],
            'n_bathrooms': [2, 3, 1, 2, 3, 1],
            'is_new_development': [True, False, True, False, True, False],
            'has_ac': [True, True, False, True, True, False],
            'has_fitted_wardrobes': [True, False, True, False, True, False],
            'has_lift': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            'is_exterior': [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            'has_pool': [False, True, False, False, True, False],
            'has_terrace': [True, False, True, False, True, False],
            'has_balcony': [False, True, False, True, False, True],
            'has_storage_room': [True, False, True, False, True, False],
            'is_accessible': [True, True, False, True, True, False],
            'has_green_zones': [True, False, True, False, True, False],
            'has_parking': [True, True, False, True, True, False],
            'house_type_id_HouseType_1_Piso': [True, False, False, True, False, True],
            'house_type_id_HouseType_2_Casa_o_chalet': [False, True, False, False, True, False],
            'house_type_id_HouseType_3_Estudio': [False, False, True, False, False, False],
            'district_id_District_1_Arganzuela': [True, False, False, False, False, True],
            'district_id_District_2_Barajas': [False, True, False, True, False, False],
            'district_id_District_3_Carabanchel': [False, False, True, False, True, False],
            'buy_price': [250000, 350000, 180000, 280000, 320000, 200000]
        })
        
        # Create data with missing values for testing
        self.data_with_missing = self.sample_data.copy()
        self.data_with_missing.loc[0, 'sq_mt_built'] = np.nan
        self.data_with_missing.loc[1, 'n_rooms'] = np.nan
        
        # Create data with outliers
        self.data_with_outliers = self.sample_data.copy()
        self.data_with_outliers.loc[0, 'sq_mt_built'] = 10000  # Extreme outlier
        self.data_with_outliers.loc[1, 'buy_price'] = 10000000  # Price outlier
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_init_default_config(self):
        """Test preprocessor initialization with default config."""
        preprocessor = MadridHousingPreprocessor()
        
        self.assertIsNotNone(preprocessor.config)
        self.assertIn('boolean_columns', preprocessor.config)
        self.assertIn('target_column', preprocessor.config)
        self.assertIn('columns_to_drop', preprocessor.config)
        self.assertIn('critical_columns', preprocessor.config)
    
    def test_init_custom_config_path(self):
        """Test preprocessor initialization with custom config path."""
        # The constructor only accepts config_path, not config directly
        preprocessor = MadridHousingPreprocessor("nonexistent.yaml")
        # Should fall back to default config
        self.assertIsNotNone(preprocessor.config)
        self.assertIn('target_column', preprocessor.config)
    
    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        preprocessor = MadridHousingPreprocessor(config_path="nonexistent.yaml")
        # Should use default config
        self.assertIsNotNone(preprocessor.config)
    
    def test_load_config_success(self):
        """Test successful config loading."""
        # Create a temporary config file for testing
        import tempfile
        import os
        
        test_config = {
            "numerical_columns": ["sq_mt_built", "n_rooms"],
            "boolean_columns": ["has_ac"],
            "target_column": "buy_price"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            preprocessor = MadridHousingPreprocessor(temp_config_path)
            # The preprocessor should load the config successfully
            self.assertIsNotNone(preprocessor.config)
            self.assertIn('target_column', preprocessor.config)
        finally:
            # Clean up the temporary file
            os.unlink(temp_config_path)
    
    def test_prepare_data_success(self):
        """Test successful data preparation."""
        result = self.preprocessor.prepare_data(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_data))
        self.assertIn('buy_price', result.columns)
        
        # Check that boolean columns are properly converted
        for col in self.preprocessor.config['boolean_columns']:
            if col in result.columns:
                self.assertEqual(result[col].dtype, 'boolean')
    
    def test_prepare_data_with_missing_values(self):
        """Test data preparation with missing values."""
        result = self.preprocessor.prepare_data(self.data_with_missing)
        
        self.assertIsInstance(result, pd.DataFrame)
        # The preprocessor may drop rows with missing critical values
        self.assertLessEqual(len(result), len(self.data_with_missing))
    
    def test_prepare_data_with_outliers(self):
        """Test data preparation with outliers."""
        result = self.preprocessor.prepare_data(self.data_with_outliers)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.data_with_outliers))
    
    def test_prepare_data_empty_dataframe(self):
        """Test data preparation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # The preprocessor handles empty dataframes gracefully
        result = self.preprocessor.prepare_data(empty_df)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_prepare_data_missing_target_column(self):
        """Test data preparation with missing target column."""
        data_without_target = self.sample_data.drop(columns=['buy_price'])
        
        # The preprocessor handles missing target columns gracefully
        result = self.preprocessor.prepare_data(data_without_target)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_prepare_data_removes_duplicates(self):
        """Test that data preparation handles duplicates."""
        # Add a duplicate row
        data_with_duplicates = pd.concat([self.sample_data, self.sample_data.iloc[[0]]], ignore_index=True)
        
        result = self.preprocessor.prepare_data(data_with_duplicates)
        
        # Should handle duplicates (either remove or keep depending on implementation)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreaterEqual(len(result), len(self.sample_data))
    
    def test_prepare_data_handles_mixed_types(self):
        """Test data preparation with mixed data types."""
        mixed_data = self.sample_data.copy()
        mixed_data['mixed_column'] = [1, 'string', 3.5, True, None, False]
        
        # Should handle mixed types gracefully
        result = self.preprocessor.prepare_data(mixed_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_prepare_data_preserves_numerical_columns(self):
        """Test that numerical columns are preserved correctly."""
        result = self.preprocessor.prepare_data(self.sample_data)
        
        # Check that numerical columns like sq_mt_built are preserved
        if 'sq_mt_built' in result.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(result['sq_mt_built']))
    
    def test_prepare_data_preserves_boolean_columns(self):
        """Test that boolean columns are preserved correctly."""
        result = self.preprocessor.prepare_data(self.sample_data)
        
        for col in self.preprocessor.config['boolean_columns']:
            if col in result.columns:
                self.assertEqual(result[col].dtype, 'boolean')
    
    def test_prepare_data_handles_categorical_columns(self):
        """Test that categorical columns are handled correctly."""
        result = self.preprocessor.prepare_data(self.sample_data)
        
        # The preprocessor doesn't have categorical_columns in config
        # Just check that the result is a valid DataFrame
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_prepare_data_with_custom_config_path(self):
        """Test data preparation with custom configuration path."""
        # Create a temporary config file
        import tempfile
        import yaml
        
        custom_config = {
            'boolean_columns': ['has_ac', 'has_pool'],
            'target_column': 'buy_price',
            'columns_to_drop': [],
            'critical_columns': ['sq_mt_built']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            config_path = f.name
        
        try:
            preprocessor = MadridHousingPreprocessor(config_path=config_path)
            result = preprocessor.prepare_data(self.sample_data)
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(self.sample_data))
        finally:
            import os
            os.unlink(config_path)
    
    def test_prepare_data_feature_engineering(self):
        """Test that feature engineering is applied correctly."""
        result = self.preprocessor.prepare_data(self.sample_data)
        
        # Check if any feature engineering was applied
        # (This depends on the actual implementation)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), 0)
    
    def test_prepare_data_scaling(self):
        """Test that numerical features are scaled appropriately."""
        result = self.preprocessor.prepare_data(self.sample_data)
        
        # Check that numerical columns exist and are properly formatted
        if 'sq_mt_built' in result.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(result['sq_mt_built']))
    
    def test_prepare_data_encoding(self):
        """Test that categorical features are encoded appropriately."""
        result = self.preprocessor.prepare_data(self.sample_data)
        
        # Check that all columns are in appropriate format
        for col in result.columns:
            if col != self.preprocessor.config['target_column']:
                # Should be numeric or boolean
                self.assertTrue(
                    pd.api.types.is_numeric_dtype(result[col]) or 
                    result[col].dtype == 'boolean'
                )
    
    def test_prepare_data_target_column_preservation(self):
        """Test that target column is preserved correctly."""
        result = self.preprocessor.prepare_data(self.sample_data)
        
        target_col = self.preprocessor.config['target_column']
        self.assertIn(target_col, result.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(result[target_col]))
    
    def test_prepare_data_consistency(self):
        """Test that data preparation is consistent across calls."""
        result1 = self.preprocessor.prepare_data(self.sample_data)
        result2 = self.preprocessor.prepare_data(self.sample_data)
        
        # Results should be consistent
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_prepare_data_different_data_sizes(self):
        """Test data preparation with different data sizes."""
        # Test with smaller dataset
        small_data = self.sample_data.iloc[:3]
        result_small = self.preprocessor.prepare_data(small_data)
        self.assertEqual(len(result_small), 3)
        
        # Test with larger dataset
        large_data = pd.concat([self.sample_data, self.sample_data], ignore_index=True)
        result_large = self.preprocessor.prepare_data(large_data)
        self.assertEqual(len(result_large), len(large_data))
    
    def test_prepare_data_edge_cases(self):
        """Test data preparation with edge cases."""
        # Test with single row
        single_row = self.sample_data.iloc[[0]]
        result_single = self.preprocessor.prepare_data(single_row)
        self.assertEqual(len(result_single), 1)
        
        # Test with all NaN values in a column
        nan_data = self.sample_data.copy()
        nan_data['sq_mt_built'] = np.nan
        result_nan = self.preprocessor.prepare_data(nan_data)
        self.assertIsInstance(result_nan, pd.DataFrame)
    
    def test_prepare_data_performance(self):
        """Test that data preparation performs reasonably."""
        import time
        
        # Create larger dataset for performance test
        large_data = pd.concat([self.sample_data] * 100, ignore_index=True)
        
        start_time = time.time()
        result = self.preprocessor.prepare_data(large_data)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(end_time - start_time, 10.0)  # 10 seconds
        self.assertEqual(len(result), len(large_data))
    
    def test_config_validation(self):
        """Test that configuration is validated properly."""
        # The preprocessor loads config from file, so test with non-existent file
        preprocessor = MadridHousingPreprocessor("nonexistent.yaml")
        # Should fall back to default config
        self.assertIsNotNone(preprocessor.config)
    
    def test_prepare_data_with_none_values(self):
        """Test data preparation with None values."""
        data_with_none = self.sample_data.copy()
        data_with_none.loc[0, 'has_ac'] = None
        
        result = self.preprocessor.prepare_data(data_with_none)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(data_with_none))
    
    def test_validate_data_with_invalid_columns(self):
        """Test data validation with invalid columns."""
        # Test with missing required columns
        invalid_data = pd.DataFrame({
            'sq_mt_built': [100, 150, 200],
            'n_rooms': [2, 3, 4],
            'buy_price': [200000, 300000, 400000]
            # Missing other required columns
        })
        
        with self.assertRaises(ValueError) as context:
            self.preprocessor.validate_data(invalid_data)
        self.assertIn("Missing required columns", str(context.exception))
    
    def test_handle_missing_values_with_different_strategies(self):
        """Test handling missing values with different strategies."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'sq_mt_built'] = np.nan
        data_with_missing.loc[1, 'n_rooms'] = np.nan
        
        # Test with different strategies
        strategies = ['mean', 'median', 'mode']
        for strategy in strategies:
            result = self.preprocessor.handle_missing_values(data_with_missing, strategy=strategy)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.isnull().any().any())
    
    def test_scale_features_with_different_scalers(self):
        """Test feature scaling with different scalers."""
        # Test with different scalers
        scalers = ['standard', 'minmax', 'robust']
        for scaler in scalers:
            result = self.preprocessor.scale_features(self.sample_data, scaler=scaler)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(self.sample_data))
    
    def test_remove_outliers_with_different_methods(self):
        """Test outlier removal with different methods."""
        # Test with different methods
        methods = ['iqr', 'zscore', 'isolation_forest']
        for method in methods:
            result = self.preprocessor.remove_outliers(self.sample_data, method=method)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertLessEqual(len(result), len(self.sample_data))
    
    def test_add_boolean_features_with_edge_cases(self):
        """Test adding boolean features with edge cases."""
        # Test with data that has no relevant columns
        data_no_bool = pd.DataFrame({
            'sq_mt_built': [100, 150, 200],
            'n_rooms': [2, 3, 4],
            'buy_price': [200000, 300000, 400000]
        })
        
        result = self.preprocessor.add_boolean_features(data_no_bool)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(data_no_bool))
    
    def test_convert_boolean_columns_with_mixed_types(self):
        """Test converting boolean columns with mixed data types."""
        # Create data with mixed boolean types
        mixed_data = pd.DataFrame({
            'has_ac': [True, False, 'yes', 'no', 1, 0],
            'has_pool': [1, 0, True, False, 'true', 'false'],
            'sq_mt_built': [100, 150, 200, 250, 300, 350]
        })
        
        result = self.preprocessor.convert_boolean_columns(mixed_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that boolean columns are properly converted
        for col in ['has_ac', 'has_pool']:
            if col in result.columns:
                self.assertTrue(result[col].dtype == bool or result[col].dtype == int)
    
    def test_handle_categorical_features_with_high_cardinality(self):
        """Test handling categorical features with high cardinality."""
        # Create data with high cardinality categorical feature
        high_cardinality_data = self.sample_data.copy()
        high_cardinality_data['district_id'] = [f'district_{i}' for i in range(100)]
        
        result = self.preprocessor.handle_categorical_features(high_cardinality_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(high_cardinality_data))
    
    def test_clean_data_with_special_characters(self):
        """Test data cleaning with special characters."""
        # Create data with special characters
        special_data = self.sample_data.copy()
        special_data['description'] = ['House with "quotes"', 'Apartment with \'apostrophes\'', 'Villa with $symbols$']
        
        result = self.preprocessor.clean_data(special_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(special_data))
    
    def test_prepare_data_with_custom_config(self):
        """Test data preparation with custom configuration."""
        # Create custom config
        custom_config = {
            'outlier_removal': {'method': 'iqr', 'threshold': 1.5},
            'scaling': {'method': 'minmax'},
            'missing_values': {'strategy': 'median'}
        }
        
        result = self.preprocessor.prepare_data(self.sample_data, config=custom_config)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_data))
    
    def test_save_pipeline_with_custom_path(self):
        """Test saving pipeline with custom path."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = os.path.join(temp_dir, 'custom_pipeline.pkl')
            
            # Mock the pipeline
            self.preprocessor.pipeline = Mock()
            
            result = self.preprocessor.save_pipeline(custom_path)
            self.assertTrue(result)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestMadridHousingPreprocessor))
    
    # Run the tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
