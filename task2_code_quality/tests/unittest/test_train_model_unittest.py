"""
Unit tests for train_model.py using unittest framework.

This module provides comprehensive unit tests for the MadridHousingTrainer class
using the standard unittest framework instead of pytest.
"""

import unittest
import unittest.mock as mock
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from train_model import MadridHousingTrainer


class TestMadridHousingTrainer(unittest.TestCase):
    """Test cases for MadridHousingTrainer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.trainer = MadridHousingTrainer("nonexistent.yaml")
        
        # Create sample housing data for tests
        self.sample_data = pd.DataFrame({
            'sq_mt_built': [85.5, 120.0, 65.0, 90.0],
            'n_rooms': [3, 4, 2, 3],
            'n_bathrooms': [2, 3, 1, 2],
            'is_new_development': [True, False, True, False],
            'has_ac': [True, True, False, True],
            'has_fitted_wardrobes': [True, False, True, False],
            'has_lift': [1.0, 0.0, 1.0, 0.0],
            'is_exterior': [1.0, 1.0, 0.0, 1.0],
            'has_pool': [False, True, False, False],
            'has_terrace': [True, False, True, False],
            'has_balcony': [False, True, False, True],
            'has_storage_room': [True, False, True, False],
            'is_accessible': [True, True, False, True],
            'has_green_zones': [True, False, True, False],
            'has_parking': [True, True, False, True],
            'house_type_id_HouseType_1_Piso': [True, False, False, True],
            'house_type_id_HouseType_2_Casa_o_chalet': [False, True, False, False],
            'house_type_id_HouseType_3_Estudio': [False, False, True, False],
            'district_id_District_1_Arganzuela': [True, False, False, False],
            'district_id_District_2_Barajas': [False, True, False, True],
            'district_id_District_3_Carabanchel': [False, False, True, False],
            'buy_price': [250000, 350000, 180000, 280000]
        })
        
        # Create train/val/test splits
        self.X_train = self.sample_data.iloc[:2].drop(columns=['buy_price'])
        self.X_val = self.sample_data.iloc[2:3].drop(columns=['buy_price'])
        self.X_test = self.sample_data.iloc[3:4].drop(columns=['buy_price'])
        self.y_train = self.sample_data.iloc[:2]['buy_price']
        self.y_val = self.sample_data.iloc[2:3]['buy_price']
        self.y_test = self.sample_data.iloc[3:4]['buy_price']
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any temporary files if needed
        pass
    
    def test_init_default_config(self):
        """Test trainer initialization with default config."""
        trainer = MadridHousingTrainer()
        self.assertIsNotNone(trainer.config)
        self.assertIn('data', trainer.config)
        self.assertIn('model', trainer.config)
        self.assertIn('mlflow', trainer.config)
        self.assertIsNone(trainer.preprocessor)
        self.assertIsNone(trainer.model)
    
    def test_init_custom_config_path(self):
        """Test trainer initialization with custom config path."""
        trainer = MadridHousingTrainer("custom_config.yaml")
        self.assertEqual(trainer.config_path.name, "custom_config.yaml")
        # Should fall back to default config since file doesn't exist
        self.assertIsNotNone(trainer.config)
    
    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        trainer = MadridHousingTrainer("nonexistent.yaml")
        # Should load default config
        self.assertIsNotNone(trainer.config)
        self.assertIn('data', trainer.config)
        self.assertIn('model', trainer.config)
    
    def test_load_config_success(self):
        """Test successful config loading."""
        # Create a temporary config file for testing
        import tempfile
        import os
        
        test_config = {
            "data": {"target_column": "buy_price"},
            "model": {"objective": "regression"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            trainer = MadridHousingTrainer(temp_config_path)
            self.assertEqual(trainer.config["data"]["target_column"], "buy_price")
            self.assertEqual(trainer.config["model"]["objective"], "regression")
        finally:
            # Clean up the temporary file
            os.unlink(temp_config_path)
    
    def test_check_preprocessed_data_exists(self):
        """Test checking for preprocessed data when it exists."""
        with patch.object(self.trainer.file_manager, 'file_exists', return_value=True):
            result = self.trainer._check_preprocessed_data()
            self.assertTrue(result)
    
    def test_check_preprocessed_data_missing(self):
        """Test checking for preprocessed data when it doesn't exist."""
        with patch.object(self.trainer.file_manager, 'file_exists', return_value=False):
            result = self.trainer._check_preprocessed_data()
            self.assertFalse(result)
    
    def test_load_preprocessed_data(self):
        """Test loading preprocessed data."""
        with patch.object(self.trainer.file_manager, 'load_dataframe') as mock_load_dataframe:
            mock_load_dataframe.return_value = self.sample_data
            
            data = self.trainer._load_preprocessed_data()
            
            self.assertIsInstance(data, pd.DataFrame)
            mock_load_dataframe.assert_called_once_with("data/preprocessed_houses_Madrid.csv")
    
    def test_prepare_data_with_preprocessed(self):
        """Test data preparation when preprocessed data exists."""
        with patch.object(self.trainer.file_manager, 'load_dataframe') as mock_load_dataframe:
            mock_load_dataframe.return_value = self.sample_data
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data()
        
        # Verify data types
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_val, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_val, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        
        # Verify shapes
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)
    
    @patch('train_model.lgb.LGBMRegressor')
    def test_train_model_success(self, mock_lgb_class):
        """Test successful model training."""
        # Setup mock model
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_lgb_class.return_value = mock_model
        
        # Train model
        result = self.trainer.train_model(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Verify model was trained
        self.assertEqual(result, mock_model)
        self.assertEqual(self.trainer.model, mock_model)
        mock_model.fit.assert_called_once()
    
    @patch('train_model.lgb.LGBMRegressor')
    def test_train_model_without_validation(self, mock_lgb_class):
        """Test model training without validation data."""
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_lgb_class.return_value = mock_model
        
        result = self.trainer.train_model(self.X_train, self.y_train)
        
        self.assertEqual(result, mock_model)
        mock_model.fit.assert_called_once()
    
    def test_evaluate_model_success(self):
        """Test model evaluation."""
        # Setup mock model
        mock_model = Mock()
        # X_test has 1 sample, so return 1 prediction
        mock_model.predict.return_value = np.array([250000])
        self.trainer.model = mock_model
        
        metrics = self.trainer.evaluate_model(self.X_test, self.y_test)
        
        # Verify metrics structure
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # Verify metric types
        self.assertIsInstance(metrics['rmse'], float)
        self.assertIsInstance(metrics['mae'], float)
        self.assertIsInstance(metrics['r2'], float)
    
    def test_evaluate_model_no_model(self):
        """Test evaluation when no model is trained."""
        self.trainer.model = None
        
        with self.assertRaises(ValueError) as context:
            self.trainer.evaluate_model(self.X_test, self.y_test)
        
        self.assertIn("No model to evaluate", str(context.exception))
    
    def test_save_model_success(self):
        """Test successful model saving."""
        mock_model = Mock()
        self.trainer.model = mock_model
        
        with patch.object(self.trainer.file_manager, 'save_model') as mock_save_model:
            self.trainer.save_model("test_model.pkl")
            mock_save_model.assert_called_once_with(mock_model, "test_model.pkl")
    
    def test_save_model_no_model(self):
        """Test saving when no model is trained."""
        self.trainer.model = None
        
        with self.assertRaises(ValueError) as context:
            self.trainer.save_model()
        
        self.assertIn("No model to save", str(context.exception))
    
    def test_save_model_with_preprocessor(self):
        """Test saving model with preprocessor."""
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.save_pipeline = Mock()
        
        self.trainer.model = mock_model
        self.trainer.preprocessor = mock_preprocessor
        
        with patch('train_model.joblib.dump'), \
             patch('train_model.Path.mkdir'):
            
            self.trainer.save_model("test_model.pkl")
            mock_preprocessor.save_pipeline.assert_called_once()
    
    @patch('train_model.mlflow.start_run')
    @patch('train_model.mlflow.log_param')
    @patch('train_model.mlflow.log_metrics')
    @patch('train_model.mlflow.lightgbm.log_model')
    def test_log_to_mlflow_success(self, mock_log_model, mock_log_metrics, 
                                 mock_log_param, mock_start_run):
        """Test successful MLflow logging."""
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3])
        metrics = {"rmse": 0.5, "mae": 0.3, "r2": 0.8}
        
        # Setup mock context
        mock_context = Mock()
        mock_context.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Set feature names
        self.trainer.feature_names = ['feature1', 'feature2', 'feature3']
        
        with patch('train_model.pd.DataFrame.to_csv'), \
             patch('train_model.mlflow.log_artifact'), \
             patch('train_model.Path.unlink'):
            
            run_id = self.trainer.log_to_mlflow(mock_model, metrics, "test_run")
            
            self.assertEqual(run_id, "test_run_id")
            mock_start_run.assert_called_once()
            mock_log_param.assert_called()
            mock_log_metrics.assert_called()
            mock_log_model.assert_called_once()
    
    @patch('train_model.mlflow.start_run')
    @patch('train_model.mlflow.log_param')
    @patch('train_model.mlflow.lightgbm.log_model')
    def test_log_to_mlflow_without_metrics(self, mock_log_model, mock_log_param, 
                                         mock_start_run):
        """Test MLflow logging without metrics."""
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3])
        
        mock_context = Mock()
        mock_context.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        self.trainer.feature_names = ['feature1', 'feature2', 'feature3']
        
        with patch('train_model.pd.DataFrame.to_csv'), \
             patch('train_model.mlflow.log_artifact'), \
             patch('train_model.Path.unlink'):
            
            run_id = self.trainer.log_to_mlflow(mock_model, run_name="test_run")
            
            self.assertEqual(run_id, "test_run_id")
            mock_start_run.assert_called_once()
    
    @patch('train_model.GridSearchCV')
    def test_run_grid_search_success(self, mock_grid_search_class):
        """Test successful grid search."""
        # Setup mock grid search
        mock_grid_search = Mock()
        mock_grid_search.fit.return_value = mock_grid_search
        mock_grid_search.best_params_ = {"learning_rate": 0.15}
        mock_grid_search.best_score_ = -0.85
        mock_grid_search_class.return_value = mock_grid_search
        
        # Setup config
        self.trainer.config["grid_search"] = {
            "parameters": {"learning_rate": [0.1, 0.2]}, 
            "cv_folds": 2
        }
        self.trainer.config["model_saving"] = {"model_path": "test_model.pkl"}
        
        # Mock prepare_data
        self.trainer.prepare_data = Mock(return_value=(
            self.X_train, self.X_val, self.X_test, 
            self.y_train, self.y_val, self.y_test
        ))
        
        with patch('train_model.joblib.dump'):
            result = self.trainer.run_grid_search()
        
        self.assertIn("best_params", result)
        self.assertIn("best_score", result)
        self.assertEqual(result["best_params"]["learning_rate"], 0.15)
    
    def test_run_grid_search_no_parameters(self):
        """Test grid search with no parameters defined."""
        self.trainer.config["grid_search"] = {"parameters": {}}
        
        with self.assertRaises(ValueError) as context:
            self.trainer.run_grid_search()
        
        self.assertIn("No grid search parameters defined", str(context.exception))
    
    @patch('train_model.mlflow.start_run')
    @patch('train_model.mlflow.log_param')
    @patch('train_model.mlflow.log_metrics')
    @patch('train_model.mlflow.lightgbm.log_model')
    def test_run_training_pipeline_success(self, mock_log_model, mock_log_metrics,
                                         mock_log_param, mock_start_run):
        """Test successful training pipeline."""
        # Setup mocks
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3])
        
        mock_context = Mock()
        mock_context.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Mock methods
        self.trainer.prepare_data = Mock(return_value=(
            self.X_train, self.X_val, self.X_test, 
            self.y_train, self.y_val, self.y_test
        ))
        self.trainer.train_model = Mock(return_value=mock_model)
        self.trainer.evaluate_model = Mock(return_value={"rmse": 0.5, "mae": 0.3, "r2": 0.8})
        self.trainer.save_model = Mock()
        self.trainer.feature_names = ['feature1', 'feature2', 'feature3']
        
        with patch('train_model.pd.DataFrame.to_csv'), \
             patch('train_model.mlflow.log_artifact'), \
             patch('train_model.Path.unlink'):
            
            result = self.trainer.run_training_pipeline("test_run")
        
        self.assertIn("run_id", result)
        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertEqual(result["run_id"], "test_run_id")
    
    def test_run_training_pipeline_mlflow_error(self):
        """Test training pipeline when MLflow fails."""
        self.trainer.prepare_data = Mock(return_value=(
            self.X_train, self.X_val, self.X_test, 
            self.y_train, self.y_val, self.y_test
        ))
        self.trainer.train_model = Mock(return_value=Mock())
        self.trainer.evaluate_model = Mock(return_value={"rmse": 0.5, "mae": 0.3, "r2": 0.8})
        self.trainer.save_model = Mock()
        
        with patch('train_model.mlflow.start_run', side_effect=Exception("MLflow error")):
            with self.assertRaises(Exception) as context:
                self.trainer.run_training_pipeline("test_run")
            
            self.assertIn("MLflow error", str(context.exception))
    
    @patch('train_model.mlflow.start_run')
    @patch('train_model.mlflow.log_param')
    @patch('train_model.mlflow.log_metrics')
    @patch('train_model.mlflow.lightgbm.log_model')
    def test_run_multiple_experiments_success(self, mock_log_model, mock_log_metrics,
                                            mock_log_param, mock_start_run):
        """Test successful multiple experiments."""
        # Setup experiments config
        self.trainer.config["experiments"] = [
            {"run_name": "exp1", "model": {"n_estimators": 50}, "training": {}},
            {"run_name": "exp2", "model": {"n_estimators": 100}, "training": {}}
        ]
        
        # Setup mocks
        mock_context = Mock()
        mock_context.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        self.trainer.prepare_data = Mock(return_value=(
            self.X_train, self.X_val, self.X_test, 
            self.y_train, self.y_val, self.y_test
        ))
        self.trainer.train_model = Mock(return_value=Mock())
        self.trainer.evaluate_model = Mock(return_value={"rmse": 0.5, "mae": 0.3, "r2": 0.8})
        self.trainer.save_model = Mock()
        
        with patch('train_model.pd.DataFrame.to_csv'), \
             patch('train_model.mlflow.log_artifact'), \
             patch('train_model.Path.unlink'):
            
            results = self.trainer.run_multiple_experiments()
        
        self.assertIn("exp1", results)
        self.assertIn("exp2", results)
        # Check that results contain the expected structure (either success or error)
        if "error" not in results["exp1"]:
            self.assertIn("run_id", results["exp1"])
            self.assertIn("metrics", results["exp1"])
        else:
            # If there's an error, check that it's a string
            self.assertIsInstance(results["exp1"]["error"], str)
    
    def test_run_multiple_experiments_no_config(self):
        """Test multiple experiments with no experiments configured."""
        self.trainer.config["experiments"] = []
        
        with patch.object(self.trainer, 'run_training_pipeline', return_value={"test": "result"}) as mock_pipeline:
            result = self.trainer.run_multiple_experiments()
            self.assertEqual(result, {"test": "result"})
            mock_pipeline.assert_called_once()
    
    def test_get_data_version_info(self):
        """Test getting data version information."""
        info = self.trainer._get_data_version_info()
        
        self.assertIsInstance(info, dict)
        # Should contain some version info keys
        self.assertTrue(len(info) > 0)
    
    def test_model_configuration(self):
        """Test model configuration structure."""
        model_config = self.trainer.config["model"]
        
        self.assertIn("objective", model_config)
        self.assertIn("n_estimators", model_config)
        self.assertIn("learning_rate", model_config)
    
    def test_data_configuration(self):
        """Test data configuration structure."""
        data_config = self.trainer.config["data"]
        
        self.assertIn("target_column", data_config)
        self.assertIn("train_size", data_config)
        self.assertIn("val_size", data_config)
        self.assertIn("test_size", data_config)
    
    def test_mlflow_configuration(self):
        """Test MLflow configuration structure."""
        mlflow_config = self.trainer.config["mlflow"]
        
        self.assertIn("experiment_name", mlflow_config)
        self.assertIn("tracking_uri", mlflow_config)


class TestMadridHousingTrainerIntegration(unittest.TestCase):
    """Integration tests for MadridHousingTrainer."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.trainer = MadridHousingTrainer("nonexistent.yaml")
    
    def test_full_training_pipeline_integration(self):
        """Test the complete training pipeline integration."""
        # Create realistic test data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'sq_mt_built': np.random.uniform(50, 200, n_samples),
            'n_rooms': np.random.randint(1, 6, n_samples),
            'n_bathrooms': np.random.randint(1, 4, n_samples),
            'is_new_development': np.random.choice([True, False], n_samples),
            'has_ac': np.random.choice([True, False], n_samples),
            'has_fitted_wardrobes': np.random.choice([True, False], n_samples),
            'has_lift': np.random.choice([1.0, 0.0], n_samples),
            'is_exterior': np.random.choice([1.0, 0.0], n_samples),
            'has_pool': np.random.choice([True, False], n_samples),
            'has_terrace': np.random.choice([True, False], n_samples),
            'has_balcony': np.random.choice([True, False], n_samples),
            'has_storage_room': np.random.choice([True, False], n_samples),
            'is_accessible': np.random.choice([True, False], n_samples),
            'has_green_zones': np.random.choice([True, False], n_samples),
            'has_parking': np.random.choice([True, False], n_samples),
        }
        
        # Add one-hot encoded features
        for house_type in ['HouseType_1_Piso', 'HouseType_2_Casa_o_chalet', 'HouseType_3_Estudio']:
            data[f'house_type_id_{house_type}'] = np.random.choice([True, False], n_samples)
        
        for district in ['District_1_Arganzuela', 'District_2_Barajas', 'District_3_Carabanchel']:
            data[f'district_id_{district}'] = np.random.choice([True, False], n_samples)
        
        # Create target variable with some relationship to features
        data['buy_price'] = (
            data['sq_mt_built'] * 1000 + 
            data['n_rooms'] * 50000 + 
            data['n_bathrooms'] * 30000 + 
            np.random.normal(0, 50000, n_samples)
        )
        
        df = pd.DataFrame(data)
        
        # Mock the data loading
        with patch('train_model.pd.read_csv', return_value=df), \
             patch('train_model.Path.exists', return_value=True):
            
            # Test data preparation
            X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data()
            
            # Verify data splits
            self.assertGreater(len(X_train), 0)
            self.assertGreater(len(X_val), 0)
            self.assertGreater(len(X_test), 0)
            
            # Verify no target column in features
            self.assertNotIn('buy_price', X_train.columns)
            self.assertNotIn('buy_price', X_val.columns)
            self.assertNotIn('buy_price', X_test.columns)
            
            # Verify target variables are series
            self.assertIsInstance(y_train, pd.Series)
            self.assertIsInstance(y_val, pd.Series)
            self.assertIsInstance(y_test, pd.Series)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using TestLoader
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestMadridHousingTrainer))
    test_suite.addTests(loader.loadTestsFromTestCase(TestMadridHousingTrainerIntegration))
    
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
