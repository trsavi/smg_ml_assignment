"""
Data loader module for Madrid Housing Market dataset.

This module provides functions to load and split the Madrid Housing Market dataset
from CSV files with proper data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """
    Load Madrid Housing Market dataset from CSV file.
    
    This function loads the dataset from the specified CSV path, performs basic
    data validation, and returns a validated pandas DataFrame.
    
    Args:
        path (str): Path to the CSV file containing the Madrid Housing Market dataset.
                   Can be relative or absolute path.
    
    Returns:
        pd.DataFrame: Loaded and validated dataset with proper data types.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file is not a valid CSV or contains no data.
        pd.errors.EmptyDataError: If the CSV file is empty.
        
    Example:
        >>> df = load_data("data/madrid_housing.csv")
        >>> print(df.shape)
        (1000, 15)
    """
    file_path = Path(path)
    
    # Validate file existence
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    
    # Validate file extension
    if file_path.suffix.lower() != '.csv':
        raise ValueError(f"Expected CSV file, got: {file_path.suffix}")
    
    try:
        # Load the dataset
        logger.info(f"Loading Madrid Housing Market dataset from: {path}")
        df = pd.read_csv(file_path)
        
        # Validate dataset is not empty
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")
        
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Basic data validation
        _validate_housing_data(df)
        
        # Basic cleaning - remove duplicates only
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        logger.info(f"Final dataset shape: {df.shape}")
        return df
        
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise ValueError(f"Failed to load dataset: {str(e)}")


def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input 'df' must be a non-empty pandas DataFrame.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    logger.info(f"Splitting dataset with target column: '{target_column}' and test_size: {test_size}")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check for missing values in target
    if y.isnull().any():
        logger.warning(f"Found {y.isnull().sum()} missing values in target column. Removing them.")
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]

        # Reset indices after filtering
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Split completed - Training: {X_train.shape}, Testing: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def _validate_housing_data(df: pd.DataFrame) -> None:
    """
    Validate that the loaded data appears to be a Madrid Housing Market dataset.
    
    Args:
        df (pd.DataFrame): The loaded dataset to validate.
        
    Raises:
        ValueError: If the dataset doesn't meet expected criteria.
    """
    # Check minimum expected columns for housing data
    expected_columns = ['buy_price', 'sq_mt_built', 'n_rooms', 'n_bathrooms']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Expected housing columns not found: {missing_columns}")
        logger.info(f"Available columns: {list(df.columns)}")
    
    # Check for reasonable data ranges
    if 'buy_price' in df.columns:
        price_col = df['buy_price']
        if price_col.min() < 0:
            logger.warning("Found negative prices in the dataset")
        if price_col.max() > 10000000:  # 10M euros seems very high for Madrid
            logger.warning("Found very high prices (>10M euros) in the dataset")
    
    if 'sq_mt_built' in df.columns:
        surface_col = df['sq_mt_built']
        if surface_col.min() < 0:
            logger.warning("Found negative surface areas in the dataset")
        if surface_col.max() > 10000:  # 10k sqm seems very large for residential
            logger.warning("Found very large surface areas (>10k sqm) in the dataset")


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Load data (this would fail if file doesn't exist)
        df = load_data("houses_Madrid.csv")
        print(f"Dataset loaded successfully: {df.shape}")
        
    except FileNotFoundError:
        print("Sample dataset file not found. This is expected for the example.")
        print("To use this module, provide a valid path to a Madrid Housing Market CSV file.")
    except Exception as e:
        print(f"Error: {e}")
