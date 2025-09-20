"""
Unit tests for the data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_data, split_data, _validate_housing_data


class TestLoadData:
    """Test cases for the load_data function."""
    
    def test_load_data_success(self, temp_csv_file):
        """Test successful data loading."""
        df = load_data(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'sq_mt_built' in df.columns
        assert 'n_rooms' in df.columns
        assert 'n_bathrooms' in df.columns
        assert 'buy_price' in df.columns
    
    def test_load_data_file_not_found(self):
        """Test load_data with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")
    
    def test_load_data_invalid_extension(self, temp_csv_file):
        """Test load_data with invalid file extension."""
        # Create a file with wrong extension
        wrong_file = temp_csv_file.replace('.csv', '.txt')
        Path(temp_csv_file).rename(wrong_file)
        
        with pytest.raises(ValueError, match="File must be a CSV"):
            load_data(wrong_file)
        
        # Cleanup
        Path(wrong_file).unlink()
    
    def test_load_data_empty_file(self):
        """Test load_data with empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(pd.errors.EmptyDataError):
                load_data(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_data_with_missing_columns(self, sample_housing_data):
        """Test load_data with missing critical columns."""
        # Create data without critical columns
        incomplete_data = sample_housing_data.drop(columns=['sq_mt_built', 'buy_price'])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing critical columns"):
                load_data(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_data_with_invalid_data_types(self):
        """Test load_data with invalid data types."""
        # Create data with string values in numeric columns
        invalid_data = pd.DataFrame({
            'sq_mt_built': ['invalid', 'data', 'here'],
            'n_rooms': [1, 2, 3],
            'buy_price': [100000, 200000, 300000]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid data types"):
                load_data(temp_path)
        finally:
            Path(temp_path).unlink()


class TestSplitData:
    """Test cases for the split_data function."""
    
    def test_split_data_success(self, sample_housing_data):
        """Test successful data splitting."""
        X_train, X_test, y_train, y_test = split_data(
            sample_housing_data, 
            target_column='buy_price',
            test_size=0.2,
            random_state=42
        )
        
        # Check shapes
        assert X_train.shape[0] == 80  # 80% of 100
        assert X_test.shape[0] == 20   # 20% of 100
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
        
        # Check that target column is not in features
        assert 'buy_price' not in X_train.columns
        assert 'buy_price' not in X_test.columns
        
        # Check data types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    
    def test_split_data_with_custom_test_size(self, sample_housing_data):
        """Test data splitting with custom test size."""
        X_train, X_test, y_train, y_test = split_data(
            sample_housing_data, 
            target_column='buy_price',
            test_size=0.3,
            random_state=42
        )
        
        assert X_train.shape[0] == 70  # 70% of 100
        assert X_test.shape[0] == 30   # 30% of 100
    
    def test_split_data_missing_target_column(self, sample_housing_data):
        """Test split_data with missing target column."""
        with pytest.raises(ValueError, match="Target column not found"):
            split_data(sample_housing_data, target_column='nonexistent_column')
    
    def test_split_data_empty_dataframe(self):
        """Test split_data with empty dataframe."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            split_data(empty_df, target_column='buy_price')
    
    def test_split_data_reproducible_with_random_state(self, sample_housing_data):
        """Test that split_data is reproducible with same random state."""
        # First split
        X_train1, X_test1, y_train1, y_test1 = split_data(
            sample_housing_data, 
            target_column='buy_price',
            test_size=0.2,
            random_state=42
        )
        
        # Second split with same random state
        X_train2, X_test2, y_train2, y_test2 = split_data(
            sample_housing_data, 
            target_column='buy_price',
            test_size=0.2,
            random_state=42
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)


class TestValidateHousingData:
    """Test cases for the _validate_housing_data function."""
    
    def test_validate_housing_data_success(self, sample_housing_data):
        """Test successful data validation."""
        # Should not raise any exception
        _validate_housing_data(sample_housing_data)
    
    def test_validate_housing_data_empty_dataframe(self):
        """Test validation with empty dataframe."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            _validate_housing_data(empty_df)
    
    def test_validate_housing_data_missing_critical_columns(self, sample_housing_data):
        """Test validation with missing critical columns."""
        incomplete_data = sample_housing_data.drop(columns=['sq_mt_built'])
        
        with pytest.raises(ValueError, match="Missing critical columns"):
            _validate_housing_data(incomplete_data)
    
    def test_validate_housing_data_with_nulls_in_critical_columns(self, sample_housing_data):
        """Test validation with nulls in critical columns."""
        data_with_nulls = sample_housing_data.copy()
        data_with_nulls.loc[0, 'sq_mt_built'] = np.nan
        
        with pytest.raises(ValueError, match="Null values found in critical columns"):
            _validate_housing_data(data_with_nulls)
    
    def test_validate_housing_data_with_negative_values(self, sample_housing_data):
        """Test validation with negative values in critical columns."""
        data_with_negatives = sample_housing_data.copy()
        data_with_negatives.loc[0, 'sq_mt_built'] = -10
        
        with pytest.raises(ValueError, match="Negative values found in critical columns"):
            _validate_housing_data(data_with_negatives)
    
    def test_validate_housing_data_with_zero_values(self, sample_housing_data):
        """Test validation with zero values in critical columns."""
        data_with_zeros = sample_housing_data.copy()
        data_with_zeros.loc[0, 'sq_mt_built'] = 0
        
        with pytest.raises(ValueError, match="Zero values found in critical columns"):
            _validate_housing_data(data_with_zeros)


# Import tempfile for tests that need it
import tempfile
