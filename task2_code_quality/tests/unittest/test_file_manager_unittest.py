"""
Comprehensive unittest tests for FileManager class to increase coverage.
"""

import unittest
import tempfile
import os
import yaml
import json
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import numpy as np

# Add src to path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.file_manager import FileManager


class TestFileManagerUnittest(unittest.TestCase):
    """Comprehensive unittest test cases for FileManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.file_manager = FileManager()

    def test_init_default(self):
        """Test FileManager initialization with default parameters."""
        fm = FileManager()
        self.assertIsNotNone(fm)
        self.assertIsInstance(fm.base_path, Path)

    def test_init_with_base_path(self):
        """Test FileManager initialization with custom base path."""
        custom_path = "/custom/path"
        fm = FileManager(base_path=custom_path)
        self.assertEqual(fm.base_path, Path(custom_path))

    def test_get_absolute_path_relative(self):
        """Test path resolution for relative paths."""
        rel_path = "test_file.txt"
        abs_path = self.file_manager.get_absolute_path(rel_path)
        self.assertIsInstance(abs_path, Path)
        self.assertTrue(abs_path.is_absolute())

    def test_get_absolute_path_absolute(self):
        """Test path resolution for absolute paths."""
        abs_input = "/absolute/path/file.txt"
        result = self.file_manager.get_absolute_path(abs_input)
        self.assertTrue(result.is_absolute())

    def test_file_exists_true(self):
        """Test file existence checking when file exists."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        try:
            self.assertTrue(self.file_manager.file_exists(temp_file))
        finally:
            os.unlink(temp_file)

    def test_file_exists_false(self):
        """Test file existence checking when file doesn't exist."""
        self.assertFalse(self.file_manager.file_exists("nonexistent_file.txt"))

    def test_ensure_directory_exists(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "new", "nested", "directory")
            file_path = os.path.join(new_dir, "test.txt")
            
            # Directory shouldn't exist initially
            self.assertFalse(os.path.exists(new_dir))
            
            # Create directory
            self.file_manager.ensure_directory_exists(file_path)
            
            # Directory should exist now
            self.assertTrue(os.path.exists(new_dir))

    def test_ensure_directory_exists_existing(self):
        """Test ensure_directory_exists with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = os.path.join(temp_dir, "existing")
            os.makedirs(existing_dir)
            
            file_path = os.path.join(existing_dir, "test.txt")
            
            # Should not raise an error
            self.file_manager.ensure_directory_exists(file_path)
            
            # Directory should still exist
            self.assertTrue(os.path.exists(existing_dir))

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
            self.assertEqual(config, test_config)
        finally:
            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.file_manager.load_config("nonexistent_config.yaml")

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                self.file_manager.load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_config_with_default(self):
        """Test config loading with default fallback."""
        default_config = {'test': 'default'}
        config = self.file_manager.load_config("nonexistent.yaml", default_config)
        self.assertEqual(config, default_config)

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
            self.assertEqual(config, test_config)
        finally:
            os.unlink(temp_path)

    def test_load_training_config_default(self):
        """Test training config loading with default fallback."""
        config = self.file_manager.load_training_config("nonexistent.yaml")
        self.assertIn('data', config)
        self.assertIn('model', config)
        self.assertIn('mlflow', config)
        self.assertIn('training', config)

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
            self.assertEqual(config, test_config)
        finally:
            os.unlink(temp_path)

    def test_load_preprocessing_config_default(self):
        """Test preprocessing config loading with default fallback."""
        config = self.file_manager.load_preprocessing_config("nonexistent.yaml")
        self.assertIn('target_column', config)
        self.assertIn('columns_to_drop', config)
        self.assertIn('boolean_columns', config)
        self.assertIn('critical_columns', config)

    def test_save_model_success(self):
        """Test model saving."""
        test_model = {"test": "data", "value": 123}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            
            self.file_manager.save_model(test_model, model_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(model_path))
            
            # Check file can be loaded
            loaded_model = joblib.load(model_path)
            self.assertEqual(loaded_model, test_model)

    def test_save_model_with_nested_directories(self):
        """Test model saving with nested directories."""
        test_model = {"test": "data", "value": 123}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "nested", "deep", "test_model.pkl")
            
            self.file_manager.save_model(test_model, model_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(model_path))

    def test_load_model_success(self):
        """Test successful model loading."""
        test_model = {"test": "data", "value": 123}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            joblib.dump(test_model, model_path)
            
            loaded_model = self.file_manager.load_model(model_path)
            self.assertEqual(loaded_model, test_model)

    def test_load_model_file_not_found(self):
        """Test model loading when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.file_manager.load_model("nonexistent_model.pkl")

    def test_save_dataframe_success(self):
        """Test DataFrame saving."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "test_data.csv")
            
            self.file_manager.save_dataframe(df, csv_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(csv_path))
            
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
            self.assertTrue(os.path.exists(csv_path))
            
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

    def test_load_dataframe_file_not_found(self):
        """Test DataFrame loading when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.file_manager.load_dataframe("nonexistent_data.csv")

    def test_save_json_success(self):
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
            self.assertTrue(os.path.exists(json_path))
            
            # Check file can be loaded
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
            self.assertEqual(loaded_data, test_data)

    def test_save_json_with_nested_directories(self):
        """Test JSON saving with nested directories."""
        test_data = {'test': 'data'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "nested", "deep", "test_data.json")
            
            self.file_manager.save_json(test_data, json_path)
            
            # Check file was created
            self.assertTrue(os.path.exists(json_path))

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
            self.assertEqual(loaded_data, test_data)

    def test_load_json_file_not_found(self):
        """Test JSON loading when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.file_manager.load_json("nonexistent_data.json")

    def test_load_json_invalid_json(self):
        """Test JSON loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_path = f.name
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                self.file_manager.load_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_error_handling_in_load_config(self):
        """Test error handling in load_config method."""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.file_manager.load_config("nonexistent.yaml")
        
        # Test with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [")
            temp_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                self.file_manager.load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_path_resolution_edge_cases(self):
        """Test path resolution with edge cases."""
        # Test with empty string
        empty_path = self.file_manager.get_absolute_path("")
        self.assertIsInstance(empty_path, Path)
        
        # Test with None (should not happen in practice, but test robustness)
        with self.assertRaises(TypeError):
            self.file_manager.get_absolute_path(None)

    def test_save_model_with_exception(self):
        """Test save_model with exception during saving."""
        test_model = {"test": "data"}
        
        with patch('joblib.dump') as mock_dump:
            mock_dump.side_effect = Exception("Save failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "test_model.pkl")
                
                with self.assertRaises(Exception) as context:
                    self.file_manager.save_model(test_model, model_path)
                self.assertIn("Save failed", str(context.exception))

    def test_load_model_with_exception(self):
        """Test load_model with exception during loading."""
        with patch('joblib.load') as mock_load:
            mock_load.side_effect = Exception("Load failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "test_model.pkl")
                # Create empty file
                with open(model_path, 'w') as f:
                    f.write("")
                
                with self.assertRaises(Exception) as context:
                    self.file_manager.load_model(model_path)
                self.assertIn("Load failed", str(context.exception))

    def test_save_dataframe_with_exception(self):
        """Test save_dataframe with exception during saving."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            mock_to_csv.side_effect = Exception("Save failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = os.path.join(temp_dir, "test_data.csv")
                
                with self.assertRaises(Exception) as context:
                    self.file_manager.save_dataframe(df, csv_path)
                self.assertIn("Save failed", str(context.exception))

    def test_load_dataframe_with_exception(self):
        """Test load_dataframe with exception during loading."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = Exception("Load failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = os.path.join(temp_dir, "test_data.csv")
                # Create empty file
                with open(csv_path, 'w') as f:
                    f.write("")
                
                with self.assertRaises(Exception) as context:
                    self.file_manager.load_dataframe(csv_path)
                self.assertIn("Load failed", str(context.exception))

    def test_save_json_with_exception(self):
        """Test save_json with exception during saving."""
        test_data = {'test': 'data'}
        
        with patch('json.dump') as mock_dump:
            mock_dump.side_effect = Exception("Save failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                json_path = os.path.join(temp_dir, "test_data.json")
                
                with self.assertRaises(Exception) as context:
                    self.file_manager.save_json(test_data, json_path)
                self.assertIn("Save failed", str(context.exception))

    def test_load_json_with_exception(self):
        """Test load_json with exception during loading."""
        with patch('json.load') as mock_load:
            mock_load.side_effect = Exception("Load failed")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                json_path = os.path.join(temp_dir, "test_data.json")
                # Create empty file
                with open(json_path, 'w') as f:
                    f.write("")
                
                with self.assertRaises(Exception) as context:
                    self.file_manager.load_json(json_path)
                self.assertIn("Load failed", str(context.exception))

    def test_default_training_config_structure(self):
        """Test that default training config has expected structure."""
        config = self.file_manager.load_training_config("nonexistent.yaml")
        
        # Check main sections
        self.assertIn('data', config)
        self.assertIn('model', config)
        self.assertIn('mlflow', config)
        self.assertIn('training', config)
        
        # Check data section
        self.assertIn('test_size', config['data'])
        self.assertIn('train_size', config['data'])
        self.assertIn('val_size', config['data'])
        self.assertIn('random_state', config['data'])
        
        # Check model section
        self.assertIn('objective', config['model'])
        self.assertIn('metric', config['model'])
        self.assertIn('learning_rate', config['model'])
        
        # Check mlflow section
        self.assertIn('experiment_name', config['mlflow'])
        self.assertIn('tracking_uri', config['mlflow'])
        
        # Check training section
        self.assertIn('early_stopping_rounds', config['training'])
        self.assertIn('eval_metric', config['training'])

    def test_default_preprocessing_config_structure(self):
        """Test that default preprocessing config has expected structure."""
        config = self.file_manager.load_preprocessing_config("nonexistent.yaml")
        
        # Check main sections
        self.assertIn('target_column', config)
        self.assertIn('columns_to_drop', config)
        self.assertIn('boolean_columns', config)
        self.assertIn('critical_columns', config)
        
        # Check types
        self.assertIsInstance(config['columns_to_drop'], list)
        self.assertIsInstance(config['boolean_columns'], list)
        self.assertIsInstance(config['critical_columns'], list)

    def test_path_handling_with_different_types(self):
        """Test path handling with different input types."""
        # Test with Path object
        path_obj = Path("test/path")
        result = self.file_manager.get_absolute_path(path_obj)
        self.assertIsInstance(result, Path)
        
        # Test with string
        str_path = "test/path"
        result = self.file_manager.get_absolute_path(str_path)
        self.assertIsInstance(result, Path)

    def test_file_operations_with_unicode_paths(self):
        """Test file operations with unicode paths."""
        test_data = {'test': 'data'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path with unicode characters
            unicode_path = os.path.join(temp_dir, "test_ñ_中文.json")
            
            # Should not raise an error
            self.file_manager.save_json(test_data, unicode_path)
            self.assertTrue(os.path.exists(unicode_path))
            
            # Should be able to load it back
            loaded_data = self.file_manager.load_json(unicode_path)
            self.assertEqual(loaded_data, test_data)

    def test_large_data_handling(self):
        """Test handling of large data structures."""
        # Test with large DataFrame
        large_df = pd.DataFrame({
            'col1': range(10000),
            'col2': [f'value_{i}' for i in range(10000)]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "large_data.csv")
            
            # Should not raise an error
            self.file_manager.save_dataframe(large_df, csv_path)
            self.assertTrue(os.path.exists(csv_path))
            
            # Should be able to load it back
            loaded_df = self.file_manager.load_dataframe(csv_path)
            self.assertEqual(len(loaded_df), 10000)

    def test_concurrent_file_operations(self):
        """Test file operations that might be called concurrently."""
        test_data = {'test': 'data'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "concurrent_test.json")
            
            # Multiple operations on the same file manager should work
            self.file_manager.save_json(test_data, json_path)
            loaded_data = self.file_manager.load_json(json_path)
            self.assertEqual(loaded_data, test_data)
            
            # Should still work after multiple operations
            self.file_manager.save_json({'test2': 'data2'}, json_path)
            loaded_data2 = self.file_manager.load_json(json_path)
            self.assertEqual(loaded_data2, {'test2': 'data2'})


if __name__ == '__main__':
    unittest.main()
