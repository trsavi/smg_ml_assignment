"""
Pytest configuration and shared fixtures for Madrid Housing Market ML Pipeline tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import tempfile
import yaml
import joblib
from faker import Faker

# Initialize Faker for generating test data
fake = Faker()


@pytest.fixture
def sample_housing_data() -> pd.DataFrame:
    """Create sample housing data for testing."""
    np.random.seed(42)
    
    data = {
        'sq_mt_built': np.random.uniform(50, 200, 100),
        'n_rooms': np.random.randint(1, 5, 100),
        'n_bathrooms': np.random.randint(1, 3, 100),
        'is_new_development': np.random.choice([True, False], 100),
        'has_ac': np.random.choice([True, False], 100),
        'has_fitted_wardrobes': np.random.choice([True, False], 100),
        'has_lift': np.random.choice([0.0, 1.0], 100),
        'is_exterior': np.random.choice([0.0, 1.0], 100),
        'has_pool': np.random.choice([True, False], 100),
        'has_terrace': np.random.choice([True, False], 100),
        'has_balcony': np.random.choice([True, False], 100),
        'has_storage_room': np.random.choice([True, False], 100),
        'is_accessible': np.random.choice([True, False], 100),
        'has_green_zones': np.random.choice([True, False], 100),
        'has_parking': np.random.choice([True, False], 100),
        'house_type_id_HouseType_1_Pisos': np.random.choice([True, False], 100),
        'house_type_id_HouseType_2_Chalets': np.random.choice([True, False], 100),
        'house_type_id_HouseType_3_Estudios': np.random.choice([True, False], 100),
        'house_type_id_HouseType_4_Duplex': np.random.choice([True, False], 100),
        'house_type_id_HouseType_5_Planta baja': np.random.choice([True, False], 100),
        'house_type_id_HouseType_6_Aticos': np.random.choice([True, False], 100),
        'house_type_id_HouseType_7_Lofts': np.random.choice([True, False], 100),
        'district_id_District_1_Arganzuela': np.random.choice([True, False], 100),
        'district_id_District_2_Barajas': np.random.choice([True, False], 100),
        'district_id_District_3_Carabanchel': np.random.choice([True, False], 100),
        'district_id_District_4_Centro': np.random.choice([True, False], 100),
        'district_id_District_5_Chamartin': np.random.choice([True, False], 100),
        'district_id_District_6_Chamberi': np.random.choice([True, False], 100),
        'district_id_District_7_Ciudad Lineal': np.random.choice([True, False], 100),
        'district_id_District_8_Fuencarral-El Pardo': np.random.choice([True, False], 100),
        'district_id_District_9_Hortaleza': np.random.choice([True, False], 100),
        'district_id_District_10_Latina': np.random.choice([True, False], 100),
        'district_id_District_11_Moncloa-Aravaca': np.random.choice([True, False], 100),
        'district_id_District_12_Moratalaz': np.random.choice([True, False], 100),
        'district_id_District_13_Puente de Vallecas': np.random.choice([True, False], 100),
        'district_id_District_14_Retiro': np.random.choice([True, False], 100),
        'district_id_District_15_Salamanca': np.random.choice([True, False], 100),
        'district_id_District_16_San Blas-Canillejas': np.random.choice([True, False], 100),
        'district_id_District_17_Tetuan': np.random.choice([True, False], 100),
        'district_id_District_18_Usera': np.random.choice([True, False], 100),
        'district_id_District_19_Vicalvaro': np.random.choice([True, False], 100),
        'district_id_District_20_Villa de Vallecas': np.random.choice([True, False], 100),
        'district_id_District_21_Villaverde': np.random.choice([True, False], 100),
        'buy_price': np.random.uniform(100000, 800000, 100)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_prediction_request() -> Dict[str, Any]:
    """Create a sample prediction request for API testing."""
    return {
        'sq_mt_built': 85.5,
        'n_rooms': 3,
        'n_bathrooms': 2,
        'is_new_development': True,
        'has_ac': True,
        'has_fitted_wardrobes': True,
        'has_lift': 1.0,
        'is_exterior': 1.0,
        'has_pool': False,
        'has_terrace': True,
        'has_balcony': False,
        'has_storage_room': True,
        'is_accessible': True,
        'has_green_zones': True,
        'has_parking': True,
        'house_type_id_HouseType_1_Pisos': True,
        'house_type_id_HouseType_2_Chalets': False,
        'house_type_id_HouseType_3_Estudios': False,
        'house_type_id_HouseType_4_Duplex': False,
        'house_type_id_HouseType_5_Planta baja': False,
        'house_type_id_HouseType_6_Aticos': False,
        'house_type_id_HouseType_7_Lofts': False,
        'district_id_District_1_Arganzuela': False,
        'district_id_District_2_Barajas': False,
        'district_id_District_3_Carabanchel': False,
        'district_id_District_4_Centro': True,
        'district_id_District_5_Chamartin': False,
        'district_id_District_6_Chamberi': False,
        'district_id_District_7_Ciudad Lineal': False,
        'district_id_District_8_Fuencarral-El Pardo': False,
        'district_id_District_9_Hortaleza': False,
        'district_id_District_10_Latina': False,
        'district_id_District_11_Moncloa-Aravaca': False,
        'district_id_District_12_Moratalaz': False,
        'district_id_District_13_Puente de Vallecas': False,
        'district_id_District_14_Retiro': False,
        'district_id_District_15_Salamanca': False,
        'district_id_District_16_San Blas-Canillejas': False,
        'district_id_District_17_Tetuan': False,
        'district_id_District_18_Usera': False,
        'district_id_District_19_Vicalvaro': False,
        'district_id_District_20_Villa de Vallecas': False,
        'district_id_District_21_Villaverde': False
    }


@pytest.fixture
def sample_batch_prediction_requests() -> List[Dict[str, Any]]:
    """Create sample batch prediction requests for API testing."""
    requests = []
    for i in range(3):
        request = sample_prediction_request()
        request['sq_mt_built'] += i * 10  # Vary the size
        requests.append(request)
    return requests


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([250000.0])
    return mock_model


@pytest.fixture
def mock_model_info():
    """Create mock model info for testing."""
    return {
        'model_type': 'LightGBM',
        'training_date': '2023-01-01',
        'version': '1.0.0',
        'features': ['sq_mt_built', 'n_rooms', 'n_bathrooms'],
        'target': 'buy_price'
    }


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    config = {
        'target_column': 'buy_price',
        'columns_to_drop': ['id'],
        'boolean_columns': ['is_new_development', 'has_ac', 'has_fitted_wardrobes'],
        'critical_columns': ['sq_mt_built', 'n_rooms', 'buy_price']
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([250000.0])
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(mock_model, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    data = pd.DataFrame({
        'sq_mt_built': [85.5, 120.0, 65.0],
        'n_rooms': [3, 4, 2],
        'n_bathrooms': [2, 3, 1],
        'buy_price': [250000, 350000, 180000]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_df():
    """Create a small sample dataframe for preprocessing tests."""
    return pd.DataFrame({
        'sq_mt_built': [85.5, 120.0, 65.0],
        'n_rooms': [3, 4, 2],
        'n_bathrooms': [2, 3, 1],
        'is_new_development': [True, False, True],
        'has_ac': [True, True, False],
        'has_fitted_wardrobes': [True, False, True],
        'has_lift': [1.0, 0.0, 1.0],
        'is_exterior': [1.0, 1.0, 0.0],
        'has_pool': [False, True, False],
        'has_terrace': [True, False, True],
        'has_balcony': [False, True, False],
        'has_storage_room': [True, False, True],
        'is_accessible': [True, True, False],
        'has_green_zones': [True, False, True],
        'has_parking': [True, True, False],
        'house_type_id_HouseType_1_Piso': [True, False, False],
        'house_type_id_HouseType_2_Casa_o_chalet': [False, True, False],
        'house_type_id_HouseType_3_Estudio': [False, False, True],
        'district_id_District_1_Arganzuela': [True, False, False],
        'district_id_District_2_Barajas': [False, True, False],
        'district_id_District_3_Carabanchel': [False, False, True],
        'buy_price': [250000, 350000, 180000]
    })


@pytest.fixture
def mock_mlflow():
    """Create mock MLflow for testing."""
    with pytest.Mock() as mock:
        mock.start_run.return_value.__enter__ = Mock()
        mock.start_run.return_value.__exit__ = Mock()
        mock.log_param = Mock()
        mock.log_metric = Mock()
        mock.log_model = Mock()
        yield mock
