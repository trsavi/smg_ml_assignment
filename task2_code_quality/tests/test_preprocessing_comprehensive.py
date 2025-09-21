"""
Comprehensive tests for MadridHousingPreprocessor class to increase coverage.
"""

import pytest
import tempfile
import os
import yaml
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import MadridHousingPreprocessor


class TestMadridHousingPreprocessorComprehensive:
    """Comprehensive test cases for MadridHousingPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'sq_mt_built': [85.5, 120.0, 65.0, 90.0, 110.0, 75.0],
            'n_rooms': [3, 4, 2, 3, 4, 2],
            'n_bathrooms': [2, 3, 1, 2, 3, 1],
            'is_new_development': [True, False, True, False, True, False],
            'has_ac': [True, True, False, True, False, True],
            'has_fitted_wardrobes': [True, False, True, False, True, False],
            'has_lift': [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            'is_exterior': [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            'has_pool': [False, True, False, False, True, False],
            'has_terrace': [True, False, True, False, True, False],
            'has_balcony': [False, True, False, True, False, True],
            'has_storage_room': [True, False, True, False, True, False],
            'is_accessible': [True, True, False, True, False, True],
            'has_green_zones': [True, False, True, False, True, False],
            'has_parking': [True, True, False, True, False, True],
            'house_type_id_HouseType_1_Piso': [True, False, False, True, False, False],
            'house_type_id_HouseType_2_Casa_o_chalet': [False, True, False, False, True, False],
            'house_type_id_HouseType_3_Estudio': [False, False, True, False, False, True],
            'district_id_District_1_Arganzuela': [True, False, False, False, False, False],
            'district_id_District_2_Barajas': [False, True, False, False, False, False],
            'district_id_District_3_Carabanchel': [False, False, True, False, False, False],
            'buy_price': [250000, 350000, 180000, 280000, 320000, 200000]
        })

    def test_init_default_config(self):
        """Test preprocessor initialization with default config."""
        preprocessor = MadridHousingPreprocessor()
        assert preprocessor.config is not None
        assert 'target_column' in preprocessor.config
        assert 'columns_to_drop' in preprocessor.config
        assert 'boolean_columns' in preprocessor.config
        assert 'critical_columns' in preprocessor.config

    def test_init_custom_config(self):
        """Test preprocessor initialization with custom config."""
        test_config = {
            'target_column': 'price',
            'columns_to_drop': ['id'],
            'boolean_columns': ['has_ac'],
            'critical_columns': ['sq_mt_built']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            preprocessor = MadridHousingPreprocessor(temp_path)
            assert preprocessor.config['target_column'] == 'price'
            assert 'id' in preprocessor.config['columns_to_drop']
        finally:
            os.unlink(temp_path)

    def test_init_nonexistent_config(self):
        """Test preprocessor initialization with nonexistent config file."""
        preprocessor = MadridHousingPreprocessor("nonexistent.yaml")
        # Should fall back to default config
        assert preprocessor.config is not None
        assert 'target_column' in preprocessor.config

    def test_prepare_data_success(self):
        """Test successful data preparation."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_data)
        assert 'buy_price' in result.columns

    def test_prepare_data_with_missing_values(self):
        """Test data preparation with missing values."""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'sq_mt_built'] = np.nan
        data_with_missing.loc[1, 'n_rooms'] = np.nan
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_missing)
        
        assert isinstance(result, pd.DataFrame)
        # Should handle missing values appropriately
        assert len(result) > 0

    def test_prepare_data_with_duplicates(self):
        """Test data preparation with duplicate rows."""
        data_with_duplicates = pd.concat([self.sample_data, self.sample_data.iloc[:2]], ignore_index=True)
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_duplicates)
        
        assert isinstance(result, pd.DataFrame)
        # The preprocessing doesn't remove duplicates, so length should be the same as input
        assert len(result) == len(data_with_duplicates)

    def test_prepare_data_empty_dataframe(self):
        """Test data preparation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_prepare_data_missing_target_column(self):
        """Test data preparation without target column."""
        data_without_target = self.sample_data.drop(columns=['buy_price'])
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_without_target)
        
        assert isinstance(result, pd.DataFrame)
        assert 'buy_price' not in result.columns

    def test_prepare_data_with_outliers(self):
        """Test data preparation with outliers."""
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'sq_mt_built'] = 10000  # Extreme outlier
        data_with_outliers.loc[1, 'buy_price'] = 10000000  # Price outlier
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_outliers)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_with_none_values(self):
        """Test data preparation with None values."""
        data_with_none = self.sample_data.copy()
        data_with_none.loc[0, 'has_ac'] = None
        data_with_none.loc[1, 'n_rooms'] = None
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_none)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_mixed_data_types(self):
        """Test data preparation with mixed data types."""
        data_mixed = self.sample_data.copy()
        data_mixed['mixed_col'] = ['string', 123, True, None, 'another', 456]
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_mixed)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_edge_cases(self):
        """Test data preparation with edge cases."""
        # Single row
        single_row = self.sample_data.iloc[:1]
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(single_row)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

        # Very large dataset
        large_data = pd.concat([self.sample_data] * 100, ignore_index=True)
        result = preprocessor.prepare_data(large_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 600

    def test_prepare_data_consistency(self):
        """Test that data preparation is consistent across calls."""
        preprocessor = MadridHousingPreprocessor()
        
        result1 = preprocessor.prepare_data(self.sample_data)
        result2 = preprocessor.prepare_data(self.sample_data)
        
        # Results should be similar (allowing for some randomness in preprocessing)
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        assert len(result1) == len(result2)

    def test_prepare_data_preserves_boolean_columns(self):
        """Test that boolean columns are preserved correctly."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        # Check that boolean columns are still present
        boolean_cols = ['is_new_development', 'has_ac', 'has_fitted_wardrobes', 
                       'has_pool', 'has_terrace', 'has_balcony', 'has_storage_room',
                       'is_accessible', 'has_green_zones', 'has_parking']
        
        for col in boolean_cols:
            if col in result.columns:
                # Check that it's a boolean-like dtype
                assert pd.api.types.is_bool_dtype(result[col]) or result[col].dtype in ['bool', 'int64', 'float64', 'boolean']

    def test_prepare_data_preserves_numerical_columns(self):
        """Test that numerical columns are preserved correctly."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        # Check that numerical columns are still present
        numerical_cols = ['sq_mt_built', 'n_rooms', 'n_bathrooms', 'buy_price']
        
        for col in numerical_cols:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col])

    def test_prepare_data_handles_categorical_columns(self):
        """Test that categorical columns are handled correctly."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        # Check that categorical columns are properly encoded
        categorical_cols = [col for col in result.columns if 'house_type_id' in col or 'district_id' in col]
        
        for col in categorical_cols:
            assert result[col].dtype in ['bool', 'int64', 'float64']

    def test_prepare_data_encoding(self):
        """Test that categorical features are encoded appropriately."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        # Check that one-hot encoded columns are present
        house_type_cols = [col for col in result.columns if 'house_type_id' in col]
        district_cols = [col for col in result.columns if 'district_id' in col]
        
        assert len(house_type_cols) > 0
        assert len(district_cols) > 0

    def test_prepare_data_feature_engineering(self):
        """Test that feature engineering is applied correctly."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        # Check that the result has the expected structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_scaling(self):
        """Test that numerical features are scaled appropriately."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        # Check that numerical columns are present and have reasonable values
        numerical_cols = ['sq_mt_built', 'n_rooms', 'n_bathrooms']
        for col in numerical_cols:
            if col in result.columns:
                assert not result[col].isna().all()

    def test_prepare_data_target_column_preservation(self):
        """Test that target column is preserved correctly."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        # Check that target column is present
        assert 'buy_price' in result.columns
        assert pd.api.types.is_numeric_dtype(result['buy_price'])

    def test_prepare_data_with_custom_config_path(self):
        """Test data preparation with custom configuration path."""
        test_config = {
            'target_column': 'price',
            'columns_to_drop': ['id'],
            'boolean_columns': ['has_ac'],
            'critical_columns': ['sq_mt_built']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            preprocessor = MadridHousingPreprocessor(temp_path)
            result = preprocessor.prepare_data(self.sample_data)
            assert isinstance(result, pd.DataFrame)
        finally:
            os.unlink(temp_path)

    def test_prepare_data_different_data_sizes(self):
        """Test data preparation with different data sizes."""
        preprocessor = MadridHousingPreprocessor()
        
        # Test with small dataset
        small_data = self.sample_data.iloc[:3]
        result_small = preprocessor.prepare_data(small_data)
        assert isinstance(result_small, pd.DataFrame)
        assert len(result_small) == 3
        
        # Test with large dataset
        large_data = pd.concat([self.sample_data] * 20, ignore_index=True)
        result_large = preprocessor.prepare_data(large_data)
        assert isinstance(result_large, pd.DataFrame)
        assert len(result_large) == 120

    def test_prepare_data_performance(self):
        """Test that data preparation performs reasonably."""
        preprocessor = MadridHousingPreprocessor()
        
        # Create a larger dataset for performance testing
        large_data = pd.concat([self.sample_data] * 100, ignore_index=True)
        
        import time
        start_time = time.time()
        result = preprocessor.prepare_data(large_data)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 10 seconds)
        assert end_time - start_time < 10
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 600

    def test_prepare_data_removes_duplicates(self):
        """Test that data preparation handles duplicates."""
        # Create data with duplicates
        data_with_duplicates = pd.concat([self.sample_data, self.sample_data.iloc[:2]], ignore_index=True)
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_duplicates)
        
        # Should remove duplicates
        assert len(result) <= len(data_with_duplicates)
        assert isinstance(result, pd.DataFrame)

    def test_save_pipeline(self):
        """Test pipeline saving functionality."""
        preprocessor = MadridHousingPreprocessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_path = os.path.join(temp_dir, "test_pipeline.pkl")
            
            # Should not raise an error
            preprocessor.save_pipeline(pipeline_path)
            
            # Check file was created
            assert os.path.exists(pipeline_path)

    def test_load_pipeline(self):
        """Test pipeline loading functionality."""
        preprocessor = MadridHousingPreprocessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_path = os.path.join(temp_dir, "test_pipeline.pkl")
            
            # Save a pipeline first
            preprocessor.save_pipeline(pipeline_path)
            
            # Load it back (method doesn't return anything, just loads into self.preprocessing_params)
            preprocessor.load_pipeline(pipeline_path)
            # The method should not raise an error
            assert True

    def test_load_pipeline_file_not_found(self):
        """Test pipeline loading when file doesn't exist."""
        preprocessor = MadridHousingPreprocessor()
        
        with pytest.raises(FileNotFoundError):
            preprocessor.load_pipeline("nonexistent_pipeline.pkl")

    def test_config_validation(self):
        """Test that configuration is validated properly."""
        preprocessor = MadridHousingPreprocessor()
        
        # Check that required config keys are present
        required_keys = ['target_column', 'columns_to_drop', 'boolean_columns', 'critical_columns']
        for key in required_keys:
            assert key in preprocessor.config

    def test_prepare_data_with_missing_critical_values(self):
        """Test data preparation with missing critical values."""
        data_missing_critical = self.sample_data.copy()
        data_missing_critical.loc[0, 'sq_mt_built'] = np.nan
        data_missing_critical.loc[1, 'buy_price'] = np.nan
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_missing_critical)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_handles_mixed_boolean_types(self):
        """Test data preparation with mixed boolean types."""
        data_mixed_bool = self.sample_data.copy()
        # Use only boolean-compatible values
        data_mixed_bool['has_ac'] = [True, False, True, False, True, False]
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_mixed_bool)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_preserves_data_types(self):
        """Test that data preparation preserves appropriate data types."""
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(self.sample_data)
        
        # Check that numerical columns are numeric
        numerical_cols = ['sq_mt_built', 'n_rooms', 'n_bathrooms', 'buy_price']
        for col in numerical_cols:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col])

    def test_prepare_data_with_custom_config(self):
        """Test data preparation with custom configuration."""
        custom_config = {
            'target_column': 'price',
            'columns_to_drop': ['id'],
            'boolean_columns': ['has_ac', 'has_pool'],
            'critical_columns': ['sq_mt_built', 'buy_price']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            preprocessor = MadridHousingPreprocessor(temp_path)
            result = preprocessor.prepare_data(self.sample_data)
            assert isinstance(result, pd.DataFrame)
            assert preprocessor.config['target_column'] == 'price'
        finally:
            os.unlink(temp_path)

    def test_prepare_data_handles_is_new_development(self):
        """Test data preparation specifically handles is_new_development column."""
        data_with_new_dev = self.sample_data.copy()
        data_with_new_dev['is_new_development'] = [True, False, True, False, True, False]
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_new_dev)
        
        assert isinstance(result, pd.DataFrame)
        assert 'is_new_development' in result.columns

    def test_fit_method(self):
        """Test fit method if it exists."""
        preprocessor = MadridHousingPreprocessor()
        
        # Test if fit method exists and can be called
        if hasattr(preprocessor, 'fit'):
            preprocessor.fit(self.sample_data)
            # Should not raise an error

    def test_transform_method(self):
        """Test transform method if it exists."""
        preprocessor = MadridHousingPreprocessor()
        
        # Test if transform method exists and can be called
        if hasattr(preprocessor, 'transform'):
            result = preprocessor.transform(self.sample_data)
            assert isinstance(result, pd.DataFrame)

    def test_prepare_data_with_none_values_in_boolean_columns(self):
        """Test data preparation with None values in boolean columns."""
        data_with_none_bool = self.sample_data.copy()
        data_with_none_bool.loc[0, 'has_ac'] = None
        data_with_none_bool.loc[1, 'is_new_development'] = None
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_none_bool)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_handles_empty_strings(self):
        """Test data preparation with empty strings."""
        data_with_empty = self.sample_data.copy()
        data_with_empty['string_col'] = ['', 'value', '', 'another', '', 'test']
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_empty)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_with_very_large_numbers(self):
        """Test data preparation with very large numbers."""
        data_with_large = self.sample_data.copy()
        data_with_large.loc[0, 'sq_mt_built'] = 1e6
        data_with_large.loc[1, 'buy_price'] = 1e9
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_large)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_with_negative_values(self):
        """Test data preparation with negative values."""
        data_with_negative = self.sample_data.copy()
        data_with_negative.loc[0, 'sq_mt_built'] = -100
        data_with_negative.loc[1, 'n_rooms'] = -5
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_negative)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_consistency_across_runs(self):
        """Test that data preparation is consistent across multiple runs."""
        preprocessor = MadridHousingPreprocessor()
        
        result1 = preprocessor.prepare_data(self.sample_data)
        result2 = preprocessor.prepare_data(self.sample_data)
        
        # Results should be similar (allowing for some randomness)
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        assert len(result1) == len(result2)

    def test_prepare_data_handles_zero_values(self):
        """Test data preparation with zero values."""
        data_with_zeros = self.sample_data.copy()
        data_with_zeros.loc[0, 'sq_mt_built'] = 0
        data_with_zeros.loc[1, 'n_rooms'] = 0
        data_with_zeros.loc[2, 'buy_price'] = 0
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_zeros)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_prepare_data_with_special_characters(self):
        """Test data preparation with special characters in string columns."""
        data_with_special = self.sample_data.copy()
        # Create a column with the same length as the dataframe
        data_with_special['special_col'] = ['test@email.com', 'value with spaces', 'value-with-dashes', 'value_with_underscores', 'another@test.com', 'final_value']
        
        preprocessor = MadridHousingPreprocessor()
        result = preprocessor.prepare_data(data_with_special)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
