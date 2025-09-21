"""
Comprehensive tests for FileManager class to increase coverage.
"""

import pytest
import tempfile
import os
import yaml
import json
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.file_manager import FileManager


class TestFileManager:
    """Test cases for FileManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.file_manager = FileManager()

    def test_init(self):
        """Test FileManager initialization."""
        fm = FileManager()
        assert fm is not None

    def test_get_absolute_path(self):
        """Test path resolution."""
        # Test relative path
        rel_path = "test_file.txt"
        abs_path = self.file_manager.get_absolute_path(rel_path)
        assert isinstance(abs_path, Path)
        assert abs_path.is_absolute()

        # Test absolute path (on Windows, this will be converted to Windows format)
        abs_input = "/absolute/path/file.txt"
        result = self.file_manager.get_absolute_path(abs_input)
        # On Windows, the path will be converted to Windows format
        assert result.is_absolute()

    def test_file_exists(self):
        """Test file existence checking."""
        # Test with non-existent file
        assert not self.file_manager.file_exists("nonexistent_file.txt")

        # Test with existing file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        try:
            assert self.file_manager.file_exists(temp_file)
        finally:
            os.unlink(temp_file)

    def test_ensure_directory_exists(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new", "nested", "directory")
            file_path = os.path.join(new_dir, "test.txt")
            
            # Directory shouldn't exist initially
            assert not os.path.exists(new_dir)
            
            # Create directory
            self.file_manager.ensure_directory_exists(file_path)
            
            # Directory should exist now
            assert os.path.exists(new_dir)

    def test_load_config_success(self):
        """Test successful config loading."""
        test_config = {
            'data': {'test_size': 0.2},
            'model': {'learning_rate': 0.1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            config = self.file_manager.load_config(temp_path)
            assert config == test_config
        finally:
            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.file_manager.load_config("nonexistent_config.yaml")

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                self.file_manager.load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_training_config_success(self):
        """Test successful training config loading."""
        test_config = {
            'data': {'test_size': 0.2},
            'model': {'learning_rate': 0.1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            config = self.file_manager.load_training_config(temp_path)
            assert config == test_config
        finally:
            os.unlink(temp_path)

    def test_load_training_config_default(self):
        """Test training config loading with default fallback."""
        config = self.file_manager.load_training_config("nonexistent.yaml")
        assert 'data' in config
        assert 'model' in config
        assert 'mlflow' in config
        assert 'training' in config

    def test_get_default_training_config(self):
        """Test default training config generation."""
        # Test by loading a non-existent config file which should return default
        config = self.file_manager.load_training_config("nonexistent.yaml")
        assert 'data' in config
        assert 'model' in config
        assert 'mlflow' in config
        assert 'training' in config
        assert config['data']['test_size'] == 0.2
        assert config['data']['train_size'] == 0.6

    def test_load_preprocessing_config_success(self):
        """Test successful preprocessing config loading."""
        test_config = {
            'target_column': 'price',
            'columns_to_drop': ['id']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            config = self.file_manager.load_preprocessing_config(temp_path)
            assert config == test_config
        finally:
            os.unlink(temp_path)

    def test_load_preprocessing_config_default(self):
        """Test preprocessing config loading with default fallback."""
        config = self.file_manager.load_preprocessing_config("nonexistent.yaml")
        assert 'target_column' in config
        assert 'columns_to_drop' in config
        assert 'boolean_columns' in config
        assert 'critical_columns' in config

    def test_get_default_preprocessing_config(self):
        """Test default preprocessing config generation."""
        # Test by loading a non-existent config file which should return default
        config = self.file_manager.load_preprocessing_config("nonexistent.yaml")
        assert 'target_column' in config
        assert 'columns_to_drop' in config
        assert 'boolean_columns' in config
        assert 'critical_columns' in config

    def test_save_model(self):
        """Test model saving."""
        # Use a simple object that can be pickled
        test_model = {"test": "data", "value": 123}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            
            self.file_manager.save_model(test_model, model_path)
            
            # Check file was created
            assert os.path.exists(model_path)
            
            # Check file can be loaded
            loaded_model = joblib.load(model_path)
            assert loaded_model == test_model

    def test_save_model_with_nested_directories(self):
        """Test model saving with nested directories."""
        test_model = {"test": "data", "value": 123}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "nested", "deep", "test_model.pkl")
            
            self.file_manager.save_model(test_model, model_path)
            
            # Check file was created
            assert os.path.exists(model_path)

    def test_load_model_success(self):
        """Test successful model loading."""
        test_model = {"test": "data", "value": 123}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            joblib.dump(test_model, model_path)
            
            loaded_model = self.file_manager.load_model(model_path)
            assert loaded_model == test_model

    def test_load_model_file_not_found(self):
        """Test model loading when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.file_manager.load_model("nonexistent_model.pkl")

    def test_save_dataframe(self):
        """Test DataFrame saving."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "test_data.csv")
            
            self.file_manager.save_dataframe(df, csv_path)
            
            # Check file was created
            assert os.path.exists(csv_path)
            
            # Check file can be loaded
            loaded_df = pd.read_csv(csv_path)
            pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_dataframe_with_index(self):
        """Test DataFrame saving with index."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "test_data.csv")
            
            self.file_manager.save_dataframe(df, csv_path, index=True)
            
            # Check file was created
            assert os.path.exists(csv_path)
            
            # Check file can be loaded with index
            loaded_df = pd.read_csv(csv_path, index_col=0)
            pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_dataframe_success(self):
        """Test successful DataFrame loading."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "test_data.csv")
            df.to_csv(csv_path, index=False)
            
            loaded_df = self.file_manager.load_dataframe(csv_path)
            pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_dataframe_with_index(self):
        """Test DataFrame loading with index."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "test_data.csv")
            df.to_csv(csv_path, index=True)
            
            loaded_df = self.file_manager.load_dataframe(csv_path)
            # Note: load_dataframe doesn't handle index, so we expect the index to be a column
            assert len(loaded_df.columns) == 3  # col1, col2, and index column

    def test_load_dataframe_file_not_found(self):
        """Test DataFrame loading when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.file_manager.load_dataframe("nonexistent_data.csv")

    def test_save_json(self):
        """Test JSON saving."""
        test_data = {
            'key1': 'value1',
            'key2': [1, 2, 3],
            'key3': {'nested': 'data'}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "test_data.json")
            
            self.file_manager.save_json(test_data, json_path)
            
            # Check file was created
            assert os.path.exists(json_path)
            
            # Check file can be loaded
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data

    def test_save_json_with_nested_directories(self):
        """Test JSON saving with nested directories."""
        test_data = {'test': 'data'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "nested", "deep", "test_data.json")
            
            self.file_manager.save_json(test_data, json_path)
            
            # Check file was created
            assert os.path.exists(json_path)

    def test_load_json_success(self):
        """Test successful JSON loading."""
        test_data = {
            'key1': 'value1',
            'key2': [1, 2, 3],
            'key3': {'nested': 'data'}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "test_data.json")
            with open(json_path, 'w') as f:
                json.dump(test_data, f)
            
            loaded_data = self.file_manager.load_json(json_path)
            assert loaded_data == test_data

    def test_load_json_file_not_found(self):
        """Test JSON loading when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.file_manager.load_json("nonexistent_data.json")

    def test_load_json_invalid_json(self):
        """Test JSON loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                self.file_manager.load_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_error_handling_in_load_config(self):
        """Test error handling in load_config method."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            self.file_manager.load_config("nonexistent.yaml")
        
        # Test with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [")
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                self.file_manager.load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_path_resolution_edge_cases(self):
        """Test path resolution with edge cases."""
        # Test with empty string
        empty_path = self.file_manager.get_absolute_path("")
        assert isinstance(empty_path, Path)
        
        # Test with None (should not happen in practice, but test robustness)
        with pytest.raises(TypeError):
            self.file_manager.get_absolute_path(None)

    def test_ensure_directory_exists_with_existing_directory(self):
        """Test ensure_directory_exists with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = os.path.join(temp_dir, "existing")
            os.makedirs(existing_dir)
            
            file_path = os.path.join(existing_dir, "test.txt")
            
            # Should not raise an error
            self.file_manager.ensure_directory_exists(file_path)
            
            # Directory should still exist
            assert os.path.exists(existing_dir)

    def test_save_model_with_exception(self):
        """Test save_model with exception during saving."""
        mock_model = Mock()
        
        with patch('joblib.dump') as mock_dump:
            mock_dump.side_effect = Exception("Save failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "test_model.pkl")
                
                with pytest.raises(Exception, match="Save failed"):
                    self.file_manager.save_model(mock_model, model_path)

    def test_load_model_with_exception(self):
        """Test load_model with exception during loading."""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = Exception("Load failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "test_model.pkl")
                # Create empty file
                with open(model_path, 'w') as f:
                    f.write("")
                
                with pytest.raises(Exception, match="Load failed"):
                    self.file_manager.load_model(model_path)

    def test_save_dataframe_with_exception(self):
        """Test save_dataframe with exception during saving."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            mock_to_csv.side_effect = Exception("Save failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = os.path.join(temp_dir, "test_data.csv")
                
                with pytest.raises(Exception, match="Save failed"):
                    self.file_manager.save_dataframe(df, csv_path)

    def test_load_dataframe_with_exception(self):
        """Test load_dataframe with exception during loading."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = Exception("Load failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = os.path.join(temp_dir, "test_data.csv")
                # Create empty file
                with open(csv_path, 'w') as f:
                    f.write("")
                
                with pytest.raises(Exception, match="Load failed"):
                    self.file_manager.load_dataframe(csv_path)

    def test_save_json_with_exception(self):
        """Test save_json with exception during saving."""
        test_data = {'test': 'data'}
        
        with patch('json.dump') as mock_dump:
            mock_dump.side_effect = Exception("Save failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                json_path = os.path.join(temp_dir, "test_data.json")
                
                with pytest.raises(Exception, match="Save failed"):
                    self.file_manager.save_json(test_data, json_path)

    def test_load_json_with_exception(self):
        """Test load_json with exception during loading."""
        with patch('json.load') as mock_load:
            mock_load.side_effect = Exception("Load failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                json_path = os.path.join(temp_dir, "test_data.json")
                # Create empty file
                with open(json_path, 'w') as f:
                    f.write("")
                
                with pytest.raises(Exception, match="Load failed"):
                    self.file_manager.load_json(json_path)
