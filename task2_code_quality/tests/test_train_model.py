import pandas as pd
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import MadridHousingTrainer


def test_init_with_default_config():
    """Test trainer initialization with default config."""
    trainer = MadridHousingTrainer()
    assert trainer.config_path.name == "training_config.yaml"
    assert "data" in trainer.config
    assert "model" in trainer.config
    assert "mlflow" in trainer.config
    assert trainer.preprocessor is None
    assert trainer.model is None


def test_init_with_custom_config(tmp_path):
    """Test trainer initialization with custom config path."""
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("""
target_column: price
model_params:
  objective: regression
  n_estimators: 100
""")
    
    trainer = MadridHousingTrainer(str(config_file))
    assert trainer.config_path == config_file
    assert trainer.config["target_column"] == "price"


def test_load_config_missing_file():
    """Test config loading when file doesn't exist."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    assert trainer.config["data"]["target_column"] == "buy_price"
    assert "model" in trainer.config


@patch('builtins.open', new_callable=lambda: Mock())
@patch('train_model.yaml.safe_load')
def test_load_config_success(mock_yaml, mock_file):
    """Test successful config loading."""
    mock_yaml.return_value = {
        "data": {"target_column": "buy_price"},
        "model": {"objective": "regression", "n_estimators": 100}
    }
    
    trainer = MadridHousingTrainer("test.yaml")
    assert trainer.config["data"]["target_column"] == "buy_price"
    assert trainer.config["model"]["objective"] == "regression"


def test_get_default_config():
    """Test default configuration generation."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    default_config = trainer._get_default_config()
    
    assert default_config["data"]["target_column"] == "buy_price"
    assert "model" in default_config
    assert "mlflow" in default_config


@patch('train_model.load_data')
def test_prepare_data_success(mock_load_data, sample_housing_data):
    """Test data preparation with successful data loading."""
    mock_load_data.return_value = sample_housing_data
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    trainer.preprocessor = Mock()
    trainer.preprocessor.prepare_data.return_value = sample_housing_data
    
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_val, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)


@patch('train_model.pd.read_csv')
@patch('train_model.Path.exists')
def test_prepare_data_with_preprocessor(mock_exists, mock_read_csv, sample_housing_data):
    """Test data preparation using preprocessor."""
    mock_exists.return_value = True  # Preprocessed data exists
    mock_read_csv.return_value = sample_housing_data
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data()
    
    # Verify data was loaded and split correctly
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    mock_read_csv.assert_called_once()


@patch('train_model.lgb.LGBMRegressor')
def test_train_model_success(mock_lgb, sample_housing_data):
    """Test successful model training."""
    mock_model = Mock()
    mock_model.fit.return_value = mock_model
    mock_lgb.return_value = mock_model
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    X_train = sample_housing_data.drop(columns=["buy_price"])
    y_train = sample_housing_data["buy_price"]
    
    trained_model = trainer.train_model(X_train, y_train)
    
    assert trained_model == mock_model
    mock_model.fit.assert_called_once()


@patch('train_model.lgb.LGBMRegressor')
def test_train_model_with_validation(mock_lgb, sample_housing_data):
    """Test model training with validation data."""
    mock_model = Mock()
    mock_model.fit.return_value = mock_model
    mock_lgb.return_value = mock_model
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    X_train = sample_housing_data.drop(columns=["buy_price"])
    y_train = sample_housing_data["buy_price"]
    X_val = sample_housing_data.drop(columns=["buy_price"])
    y_val = sample_housing_data["buy_price"]
    
    trained_model = trainer.train_model(X_train, y_train, X_val, y_val)
    
    assert trained_model == mock_model
    # Should call fit with eval_set when validation data provided
    mock_model.fit.assert_called_once()


def test_evaluate_model(sample_housing_data):
    """Test model evaluation."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    trainer.model = Mock()
    
    X_test = sample_housing_data.drop(columns=["buy_price"])
    y_test = sample_housing_data["buy_price"]
    
    trainer.model.predict.return_value = np.full(len(y_test), 200000)
    
    metrics = trainer.evaluate_model(X_test, y_test)
    
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_model_with_predictions():
    """Test model evaluation with custom predictions."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    trainer.model = Mock()
    predictions = np.array([100000, 200000])
    trainer.model.predict.return_value = predictions
    
    X_test = pd.DataFrame({"feature": [1, 2]})
    y_test = pd.Series([110000, 190000])
    
    metrics = trainer.evaluate_model(X_test, y_test)
    
    # Verify model.predict was called
    trainer.model.predict.assert_called_once_with(X_test)
    
    # Check metrics are reasonable
    assert metrics["r2"] > 0  # Should be positive for this example


def test_save_model(tmp_path):
    """Test model saving functionality."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    trainer.model = Mock()
    
    model_path = tmp_path / "test_model.pkl"
    
    with patch('train_model.joblib.dump') as mock_dump:
        trainer.save_model(str(model_path))
        
        mock_dump.assert_called_once_with(trainer.model, str(model_path))


def test_save_model_default_path():
    """Test model saving with default path."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    trainer.model = Mock()
    
    with patch('train_model.joblib.dump') as mock_dump:
        trainer.save_model()
        
        # Should use default path
        mock_dump.assert_called_once()



def test_get_data_version_info():
    """Test data version information retrieval."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    version_info = trainer._get_data_version_info()
    
    assert "data_rows" in version_info
    assert "data_version" in version_info


@patch('train_model.load_data')
def test_check_preprocessed_data_exists(mock_load_data, tmp_path):
    """Test checking for existing preprocessed data."""
    # Create a preprocessed file
    preprocessed_file = tmp_path / "preprocessed_houses_Madrid.csv"
    preprocessed_file.write_text("test,data\n1,2\n")
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    trainer.config["data_path"] = "test.csv"
    trainer.config["preprocessed_file"] = str(preprocessed_file)
    
    exists = trainer._check_preprocessed_data()
    assert exists is True


def test_check_preprocessed_data_missing():
    """Test checking for missing preprocessed data."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    with patch('train_model.Path.exists', return_value=False):
        assert trainer._check_preprocessed_data() is False



@patch('train_model.subprocess.run')
def test_prepare_data_if_needed_with_preprocessing(mock_subprocess_run, sample_housing_data):
    """Test data preparation when preprocessing is needed."""
    mock_subprocess_run.return_value = Mock(returncode=0)
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    trainer._check_preprocessed_data = Mock(return_value=False)
    
    trainer._prepare_data_if_needed()
    
    # Should call subprocess to run prepare_data script
    mock_subprocess_run.assert_called_once()


@patch('train_model.pd.read_csv')
def test_load_preprocessed_data(mock_read_csv, sample_housing_data):
    """Test loading preprocessed data."""
    mock_read_csv.return_value = sample_housing_data
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    data = trainer._load_preprocessed_data()
    
    assert isinstance(data, pd.DataFrame)
    # Should call with the default preprocessed file path
    mock_read_csv.assert_called_once()


@patch('train_model.GridSearchCV')
def test_run_grid_search(mock_grid_search):
    """Test hyperparameter tuning with grid search."""
    mock_grid_search_instance = Mock()
    mock_grid_search_instance.fit.return_value = mock_grid_search_instance
    mock_grid_search_instance.best_params_ = {"learning_rate": 0.15}
    mock_grid_search_instance.best_score_ = -0.85
    mock_grid_search.return_value = mock_grid_search_instance
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    # Create test data with proper housing features
    X_train = pd.DataFrame({
        'sq_mt_built': [85.5, 120.0],
        'n_rooms': [3, 4],
        'n_bathrooms': [2, 3],
        'is_new_development': [True, False],
        'has_ac': [True, True],
        'has_fitted_wardrobes': [True, False],
        'has_lift': [1.0, 0.0],
        'is_exterior': [1.0, 1.0],
        'has_pool': [False, True],
        'has_terrace': [True, False],
        'has_balcony': [False, True],
        'has_storage_room': [True, False],
        'is_accessible': [True, True],
        'has_green_zones': [True, False],
        'has_parking': [True, True],
        'house_type_id_HouseType_1_Piso': [True, False],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, True],
        'house_type_id_HouseType_3_Estudio': [False, False],
        'district_id_District_1_Arganzuela': [True, False],
        'district_id_District_2_Barajas': [False, True],
        'district_id_District_3_Carabanchel': [False, False]
    })
    X_val = pd.DataFrame({
        'sq_mt_built': [65.0, 90.0],
        'n_rooms': [2, 3],
        'n_bathrooms': [1, 2],
        'is_new_development': [True, False],
        'has_ac': [False, True],
        'has_fitted_wardrobes': [True, False],
        'has_lift': [1.0, 0.0],
        'is_exterior': [0.0, 1.0],
        'has_pool': [False, False],
        'has_terrace': [True, False],
        'has_balcony': [False, True],
        'has_storage_room': [True, False],
        'is_accessible': [False, True],
        'has_green_zones': [True, False],
        'has_parking': [False, True],
        'house_type_id_HouseType_1_Piso': [False, True],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, False],
        'house_type_id_HouseType_3_Estudio': [True, False],
        'district_id_District_1_Arganzuela': [False, True],
        'district_id_District_2_Barajas': [False, False],
        'district_id_District_3_Carabanchel': [True, False]
    })
    X_test = pd.DataFrame({
        'sq_mt_built': [75.0, 110.0],
        'n_rooms': [3, 4],
        'n_bathrooms': [2, 3],
        'is_new_development': [False, True],
        'has_ac': [True, False],
        'has_fitted_wardrobes': [False, True],
        'has_lift': [0.0, 1.0],
        'is_exterior': [1.0, 0.0],
        'has_pool': [True, False],
        'has_terrace': [False, True],
        'has_balcony': [True, False],
        'has_storage_room': [False, True],
        'is_accessible': [True, False],
        'has_green_zones': [False, True],
        'has_parking': [True, False],
        'house_type_id_HouseType_1_Piso': [True, False],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, True],
        'house_type_id_HouseType_3_Estudio': [False, False],
        'district_id_District_1_Arganzuela': [False, False],
        'district_id_District_2_Barajas': [True, False],
        'district_id_District_3_Carabanchel': [False, True]
    })
    y_train = pd.Series([250000, 350000])
    y_val = pd.Series([180000, 280000])
    y_test = pd.Series([220000, 320000])
    
    # Mock the prepare_data method
    trainer.prepare_data = Mock(return_value=(X_train, X_val, X_test, y_train, y_val, y_test))

    trainer.config["grid_search"] = {"parameters": {"learning_rate": [0.1, 0.2]}, "cv_folds": 2}
    trainer.config["model_saving"] = {"model_path": "test_model.pkl"}

    # Mock joblib.dump to prevent pickling issues
    with patch('train_model.joblib.dump'):
        result = trainer.run_grid_search()
    
    assert "best_params" in result
    assert "best_score" in result
    assert result["best_params"]["learning_rate"] == 0.15


def test_run_grid_search_disabled():
    """Test grid search when no parameters are defined."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    # Mock prepare_data to avoid subprocess calls
    trainer.prepare_data = Mock(return_value=(
        pd.DataFrame({"feature": [1, 2]}),  # X_train
        pd.DataFrame({"feature": [3, 4]}),  # X_val
        pd.DataFrame({"feature": [5, 6]}),  # X_test
        pd.Series([100, 200]),              # y_train
        pd.Series([300, 400]),              # y_val
        pd.Series([500, 600])               # y_test
    ))
    
    with pytest.raises(ValueError, match="No grid search parameters defined"):
        trainer.run_grid_search()


@patch('train_model.lgb.LGBMRegressor')
@patch('train_model.mlflow.start_run')
def test_run_training_pipeline(mock_start_run, mock_lgb):
    """Test complete training pipeline."""
    mock_model = Mock()
    mock_model.fit.return_value = mock_model
    mock_model.predict.return_value = np.array([250000, 350000])
    mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    mock_lgb.return_value = mock_model
    
    mock_context = Mock()
    mock_start_run.return_value.__enter__ = Mock(return_value=mock_context)
    mock_start_run.return_value.__exit__ = Mock(return_value=None)
    
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    # Create proper housing test data
    X_train = pd.DataFrame({
        'sq_mt_built': [85.5, 120.0],
        'n_rooms': [3, 4],
        'n_bathrooms': [2, 3],
        'is_new_development': [True, False],
        'has_ac': [True, True],
        'has_fitted_wardrobes': [True, False],
        'has_lift': [1.0, 0.0],
        'is_exterior': [1.0, 1.0],
        'has_pool': [False, True],
        'has_terrace': [True, False],
        'has_balcony': [False, True],
        'has_storage_room': [True, False],
        'is_accessible': [True, True],
        'has_green_zones': [True, False],
        'has_parking': [True, True],
        'house_type_id_HouseType_1_Piso': [True, False],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, True],
        'house_type_id_HouseType_3_Estudio': [False, False],
        'district_id_District_1_Arganzuela': [True, False],
        'district_id_District_2_Barajas': [False, True],
        'district_id_District_3_Carabanchel': [False, False]
    })
    X_val = pd.DataFrame({
        'sq_mt_built': [65.0, 90.0],
        'n_rooms': [2, 3],
        'n_bathrooms': [1, 2],
        'is_new_development': [True, False],
        'has_ac': [False, True],
        'has_fitted_wardrobes': [True, False],
        'has_lift': [1.0, 0.0],
        'is_exterior': [0.0, 1.0],
        'has_pool': [False, False],
        'has_terrace': [True, False],
        'has_balcony': [False, True],
        'has_storage_room': [True, False],
        'is_accessible': [False, True],
        'has_green_zones': [True, False],
        'has_parking': [False, True],
        'house_type_id_HouseType_1_Piso': [False, True],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, False],
        'house_type_id_HouseType_3_Estudio': [True, False],
        'district_id_District_1_Arganzuela': [False, True],
        'district_id_District_2_Barajas': [False, False],
        'district_id_District_3_Carabanchel': [True, False]
    })
    X_test = pd.DataFrame({
        'sq_mt_built': [75.0, 110.0],
        'n_rooms': [3, 4],
        'n_bathrooms': [2, 3],
        'is_new_development': [False, True],
        'has_ac': [True, False],
        'has_fitted_wardrobes': [False, True],
        'has_lift': [0.0, 1.0],
        'is_exterior': [1.0, 0.0],
        'has_pool': [True, False],
        'has_terrace': [False, True],
        'has_balcony': [True, False],
        'has_storage_room': [False, True],
        'is_accessible': [True, False],
        'has_green_zones': [False, True],
        'has_parking': [True, False],
        'house_type_id_HouseType_1_Piso': [True, False],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, True],
        'house_type_id_HouseType_3_Estudio': [False, False],
        'district_id_District_1_Arganzuela': [False, False],
        'district_id_District_2_Barajas': [True, False],
        'district_id_District_3_Carabanchel': [False, True]
    })
    y_train = pd.Series([250000, 350000])
    y_val = pd.Series([180000, 280000])
    y_test = pd.Series([220000, 320000])
    
    trainer.prepare_data = Mock(return_value=(X_train, X_val, X_test, y_train, y_val, y_test))
    
    # Mock save_model
    trainer.save_model = Mock()
    
    with patch('train_model.mlflow.log_param'), \
         patch('train_model.mlflow.log_metric'), \
         patch('train_model.mlflow.lightgbm.log_model'):
        
        result = trainer.run_training_pipeline("test_run")
    
    assert "model" in result
    assert "metrics" in result
    assert "run_id" in result


@patch('train_model.mlflow.end_run')
def test_run_training_pipeline_without_mlflow(mock_end_run):
    """Test training pipeline without MLflow."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    # Create proper housing test data
    X_train = pd.DataFrame({
        'sq_mt_built': [85.5, 120.0],
        'n_rooms': [3, 4],
        'n_bathrooms': [2, 3],
        'is_new_development': [True, False],
        'has_ac': [True, True],
        'has_fitted_wardrobes': [True, False],
        'has_lift': [1.0, 0.0],
        'is_exterior': [1.0, 1.0],
        'has_pool': [False, True],
        'has_terrace': [True, False],
        'has_balcony': [False, True],
        'has_storage_room': [True, False],
        'is_accessible': [True, True],
        'has_green_zones': [True, False],
        'has_parking': [True, True],
        'house_type_id_HouseType_1_Piso': [True, False],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, True],
        'house_type_id_HouseType_3_Estudio': [False, False],
        'district_id_District_1_Arganzuela': [True, False],
        'district_id_District_2_Barajas': [False, True],
        'district_id_District_3_Carabanchel': [False, False]
    })
    X_val = pd.DataFrame({
        'sq_mt_built': [65.0, 90.0],
        'n_rooms': [2, 3],
        'n_bathrooms': [1, 2],
        'is_new_development': [True, False],
        'has_ac': [False, True],
        'has_fitted_wardrobes': [True, False],
        'has_lift': [1.0, 0.0],
        'is_exterior': [0.0, 1.0],
        'has_pool': [False, False],
        'has_terrace': [True, False],
        'has_balcony': [False, True],
        'has_storage_room': [True, False],
        'is_accessible': [False, True],
        'has_green_zones': [True, False],
        'has_parking': [False, True],
        'house_type_id_HouseType_1_Piso': [False, True],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, False],
        'house_type_id_HouseType_3_Estudio': [True, False],
        'district_id_District_1_Arganzuela': [False, True],
        'district_id_District_2_Barajas': [False, False],
        'district_id_District_3_Carabanchel': [True, False]
    })
    X_test = pd.DataFrame({
        'sq_mt_built': [75.0, 110.0],
        'n_rooms': [3, 4],
        'n_bathrooms': [2, 3],
        'is_new_development': [False, True],
        'has_ac': [True, False],
        'has_fitted_wardrobes': [False, True],
        'has_lift': [0.0, 1.0],
        'is_exterior': [1.0, 0.0],
        'has_pool': [True, False],
        'has_terrace': [False, True],
        'has_balcony': [True, False],
        'has_storage_room': [False, True],
        'is_accessible': [True, False],
        'has_green_zones': [False, True],
        'has_parking': [True, False],
        'house_type_id_HouseType_1_Piso': [True, False],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, True],
        'house_type_id_HouseType_3_Estudio': [False, False],
        'district_id_District_1_Arganzuela': [False, False],
        'district_id_District_2_Barajas': [True, False],
        'district_id_District_3_Carabanchel': [False, True]
    })
    y_train = pd.Series([250000, 350000])
    y_val = pd.Series([180000, 280000])
    y_test = pd.Series([220000, 320000])
    
    trainer.prepare_data = Mock(return_value=(X_train, X_val, X_test, y_train, y_val, y_test))
    
    # Mock other methods
    trainer.train_model = Mock(return_value=Mock())
    trainer.evaluate_model = Mock(return_value={"rmse": 0.5, "mae": 0.3, "r2": 0.8})
    trainer.save_model = Mock()
    
    # Mock MLflow to prevent active run conflicts
    with patch('train_model.mlflow.start_run') as mock_start_run:
        mock_start_run.side_effect = Exception("MLflow not available")
        
        # The implementation should raise the exception
        with pytest.raises(Exception, match="MLflow not available"):
            trainer.run_training_pipeline("test_run")


def test_model_params_configuration():
    """Test model parameters are properly configured."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    model_params = trainer.config["model"]
    
    assert "objective" in model_params
    assert "metric" in model_params
    assert "learning_rate" in model_params
    assert "boosting_type" in model_params


def test_data_configuration():
    """Test data configuration."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    data_config = trainer.config["data"]
    
    assert "source_path" in data_config
    assert "target_column" in data_config
    assert "test_size" in data_config
    assert data_config["target_column"] == "buy_price"


def test_mlflow_configuration():
    """Test MLflow configuration."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    mlflow_config = trainer.config["mlflow"]
    
    assert "experiment_name" in mlflow_config
    assert "tracking_uri" in mlflow_config


def test_save_model_no_model():
    """Test save_model when no model is trained."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    trainer.model = None
    
    with pytest.raises(ValueError, match="No model to save"):
        trainer.save_model()


def test_save_model_with_preprocessor():
    """Test save_model when preprocessor is available."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    trainer.model = Mock()
    trainer.preprocessor = Mock()
    trainer.preprocessor.save_pipeline = Mock()
    
    with patch('train_model.joblib.dump'), \
         patch('train_model.Path.mkdir'):
        trainer.save_model("test_model.pkl")
    
    # Verify preprocessor was saved
    trainer.preprocessor.save_pipeline.assert_called_once()


def test_log_to_mlflow_with_feature_importances():
    """Test MLflow logging with feature importances."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    # Create mock model with feature importances
    mock_model = Mock()
    mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    
    metrics = {"rmse": 0.5, "mae": 0.3, "r2": 0.8}
    
    # Set feature names
    trainer.feature_names = ['sq_mt_built', 'n_rooms', 'n_bathrooms', 'is_new_development', 'has_ac', 'has_fitted_wardrobes', 'has_lift', 'is_exterior', 'has_pool', 'has_terrace', 'has_balcony', 'has_storage_room', 'is_accessible', 'has_green_zones', 'has_parking', 'house_type_id_HouseType_1_Piso', 'house_type_id_HouseType_2_Casa_o_chalet', 'house_type_id_HouseType_3_Estudio', 'district_id_District_1_Arganzuela', 'district_id_District_2_Barajas', 'district_id_District_3_Carabanchel']
    
    with patch('train_model.mlflow.start_run') as mock_start_run, \
         patch('train_model.mlflow.log_param') as mock_log_param, \
         patch('train_model.mlflow.log_metrics') as mock_log_metrics, \
         patch('train_model.mlflow.lightgbm.log_model') as mock_log_model, \
         patch('train_model.pd.DataFrame.to_csv'), \
         patch('train_model.mlflow.log_artifact'), \
         patch('train_model.Path.unlink'):
        
        mock_context = Mock()
        mock_context.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        run_id = trainer.log_to_mlflow(mock_model, metrics, "test_run")
        
        # Verify MLflow calls were made
        mock_start_run.assert_called_once()
        mock_log_param.assert_called()
        mock_log_metrics.assert_called()
        mock_log_model.assert_called_once()
        assert run_id == "test_run_id"


def test_run_multiple_experiments():
    """Test running multiple experiments."""
    trainer = MadridHousingTrainer("nonexistent.yaml")
    
    # Create test data
    X_train = pd.DataFrame({'feature': [1, 2]})
    X_val = pd.DataFrame({'feature': [3, 4]})
    X_test = pd.DataFrame({'feature': [5, 6]})
    y_train = pd.Series([100, 200])
    y_val = pd.Series([300, 400])
    y_test = pd.Series([500, 600])
    
    trainer.prepare_data = Mock(return_value=(X_train, X_val, X_test, y_train, y_val, y_test))
    
    # Set up experiments config with proper structure
    trainer.config["experiments"] = [
        {"run_name": "exp1", "model": {"n_estimators": 50}},
        {"run_name": "exp2", "model": {"n_estimators": 100}}
    ]
    
    # Mock other methods
    trainer.train_model = Mock(return_value=Mock())
    trainer.evaluate_model = Mock(return_value={"rmse": 0.5, "mae": 0.3, "r2": 0.8})
    trainer.save_model = Mock()
    
    with patch('train_model.mlflow.start_run') as mock_start_run, \
         patch('train_model.mlflow.log_param'), \
         patch('train_model.mlflow.log_metrics'), \
         patch('train_model.mlflow.lightgbm.log_model'):
        
        mock_context = Mock()
        mock_context.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        results = trainer.run_multiple_experiments()
        
        assert "exp1" in results
        assert "exp2" in results
        assert "best_experiment" in results
