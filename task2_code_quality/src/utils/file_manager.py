"""
File management utilities for Madrid Housing Market ML pipeline.

This module provides a centralized FileManager class for loading configuration files,
managing file paths, and handling file operations across the ML pipeline.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import joblib
import json

# Setup logging
logger = logging.getLogger(__name__)


class FileManager:
    """Centralized file management for configuration and data files."""
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize FileManager with optional base path.
        
        Args:
            base_path: Base directory path for relative file operations.
                      If None, uses current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
    def load_config(self, config_path: Union[str, Path], 
                   default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with fallback to default config.
        
        Args:
            config_path: Path to the YAML configuration file.
            default_config: Default configuration to use if file loading fails.
            
        Returns:
            Dict containing the loaded configuration.
        """
        config_path = Path(config_path)
        
        # If path is relative, make it relative to base_path
        if not config_path.is_absolute():
            config_path = self.base_path / config_path
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            if default_config:
                logger.info("Using provided default configuration")
                return default_config
            else:
                raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            if default_config:
                logger.info("Using provided default configuration due to YAML error")
                return default_config
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error loading config from {config_path}: {e}")
            if default_config:
                logger.info("Using provided default configuration due to error")
                return default_config
            else:
                raise
    
    def load_training_config(self, config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
        """
        Load training configuration with default fallback.
        
        Args:
            config_path: Path to training configuration file.
            
        Returns:
            Dict containing training configuration.
        """
        default_config = {
            'data': {
                'source_path': 'houses_Madrid.csv',
                'target_column': 'buy_price',
                'test_size': 0.2,
                'val_size': 0.2,
                'train_size': 0.6,
                'random_state': 42
            },
            'mlflow': {
                'experiment_name': 'housing_price_experiments',
                'tracking_uri': './mlruns'
            },
            'model': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_estimators': 300,
                'min_child_samples': 4,
                'max_depth': 8,
                'num_leaves': 20,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            },
            'training': {
                'early_stopping_rounds': 10,
                'eval_metric': 'rmse',
                'verbose_eval': 100
            },
            'model_saving': {
                'model_path': 'models/madrid_housing_model.pkl'
            }
        }
        
        return self.load_config(config_path, default_config)
    
    def load_preprocessing_config(self, config_path: str = "configs/preprocessing_config.yaml") -> Dict[str, Any]:
        """
        Load preprocessing configuration with default fallback.
        
        Args:
            config_path: Path to preprocessing configuration file.
            
        Returns:
            Dict containing preprocessing configuration.
        """
        default_config = {
            'target_column': 'buy_price',
            'columns_to_drop': [],
            'boolean_columns': [],
            'critical_columns': []
        }
        
        return self.load_config(config_path, default_config)
    
    def ensure_directory_exists(self, directory_path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            directory_path: Path to directory.
            
        Returns:
            Path object of the directory.
        """
        directory_path = Path(directory_path)
        
        # If path is relative, make it relative to base_path
        if not directory_path.is_absolute():
            directory_path = self.base_path / directory_path
            
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {directory_path}")
        return directory_path
    
    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file exists.
        
        Args:
            file_path: Path to file.
            
        Returns:
            True if file exists, False otherwise.
        """
        file_path = Path(file_path)
        
        # If path is relative, make it relative to base_path
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
            
        return file_path.exists()
    
    def get_absolute_path(self, file_path: Union[str, Path]) -> Path:
        """
        Get absolute path for a file, resolving relative paths against base_path.
        
        Args:
            file_path: Path to file (relative or absolute).
            
        Returns:
            Absolute Path object.
        """
        file_path = Path(file_path)
        
        # If path is relative, make it relative to base_path
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
            
        return file_path.resolve()
    
    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> str:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save.
            file_path: Path where to save the file.
            
        Returns:
            String path of the saved file.
        """
        
        file_path = self.get_absolute_path(file_path)
        
        # Ensure directory exists
        self.ensure_directory_exists(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON data saved to: {file_path}")
        return str(file_path)
    
    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file.
            
        Returns:
            Dict containing loaded data.
        """
        
        file_path = self.get_absolute_path(file_path)
        
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"JSON data loaded from: {file_path}")
        return data
    
    def save_model(self, model: Any, file_path: Union[str, Path]) -> str:
        """
        Save model using joblib.
        
        Args:
            model: Model object to save.
            file_path: Path where to save the model.
            
        Returns:
            String path of the saved model.
        """
        file_path = self.get_absolute_path(file_path)
        
        # Ensure directory exists
        self.ensure_directory_exists(file_path.parent)
        
        joblib.dump(model, file_path)
        logger.info(f"Model saved to: {file_path}")
        return str(file_path)
    
    def load_model(self, file_path: Union[str, Path]) -> Any:
        """
        Load model using joblib.
        
        Args:
            file_path: Path to model file.
            
        Returns:
            Loaded model object.
        """
        file_path = self.get_absolute_path(file_path)
        
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        model = joblib.load(file_path)
        logger.info(f"Model loaded from: {file_path}")
        return model
    
    def save_dataframe(self, df: pd.DataFrame, file_path: Union[str, Path], 
                      index: bool = False) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save.
            file_path: Path where to save the CSV.
            index: Whether to include index in CSV.
            
        Returns:
            String path of the saved file.
        """
        file_path = self.get_absolute_path(file_path)
        
        # Ensure directory exists
        self.ensure_directory_exists(file_path.parent)
        
        df.to_csv(file_path, index=index)
        logger.info(f"DataFrame saved to: {file_path}")
        return str(file_path)
    
    def load_dataframe(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load DataFrame from CSV file.
        
        Args:
            file_path: Path to CSV file.
            
        Returns:
            Loaded DataFrame.
        """
        file_path = self.get_absolute_path(file_path)
        
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"DataFrame loaded from: {file_path}")
        return df
