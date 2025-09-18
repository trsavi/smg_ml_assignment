"""
Preprocessing module for Madrid Housing Market dataset.

This module provides preprocessing pipelines using scikit-learn for handling
missing values, categorical encoding, and feature scaling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MadridHousingPreprocessor:
    """Preprocessing pipeline for Madrid Housing Market dataset."""
    
    def __init__(self, config_path: str = "configs/preprocessing_config.yaml"):
        """Initialize preprocessor with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.preprocessing_pipeline = None
        self.feature_columns = None
        self.target_column = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load preprocessing configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Use default configuration if file not found
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            'target_column': 'buy_price',
            'columns_to_drop': [
                'Unnamed: 0', 'id', 'title', 'subtitle', 'sq_mt_useful', 'n_floors',
                'sq_mt_allotment', 'latitude', 'longitude', 'raw_address',
                'is_exact_address_hidden', 'street_name', 'street_number', 'portal',
                'floor', 'is_floor_under', 'door', 'operation', 'rent_price',
                'rent_price_by_area', 'is_rent_price_known', 'buy_price_by_area',
                'is_buy_price_known', 'has_central_heating', 'has_individual_heating',
                'are_pets_allowed', 'is_furnished', 'is_kitchen_equipped', 'has_garden',
                'is_renewal_needed', 'energy_certificate', 'has_private_parking',
                'has_public_parking', 'is_parking_included_in_price', 'parking_price',
                'is_orientation_north', 'is_orientation_west', 'is_orientation_south',
                'is_orientation_east'
            ],
            'boolean_columns': [
                'has_ac', 'has_fitted_wardrobes', 'has_pool', 'has_terrace',
                'has_balcony', 'has_storage_room', 'is_accessible', 'has_green_zones'
            ],
            'categorical_columns': ['house_type_id', 'neighborhood_id'],
            'critical_columns': ['sq_mt_built', 'n_bathrooms'],
            'numeric_columns': ['sq_mt_built', 'n_rooms', 'n_bathrooms', 'built_year']
        }
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe by dropping unnecessary columns and handling critical missing values."""
        logger.info("Preparing dataframe...")
        
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = self.config['columns_to_drop']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        df_processed = df_processed.drop(columns=existing_columns_to_drop)
        logger.info(f"Dropped {len(existing_columns_to_drop)} columns")
        
        # Drop rows with missing critical values
        critical_columns = self.config['critical_columns']
        initial_rows = len(df_processed)
        
        for col in critical_columns:
            if col in df_processed.columns:
                df_processed = df_processed[df_processed[col].notna()]
        
        dropped_rows = initial_rows - len(df_processed)
        logger.info(f"Dropped {dropped_rows} rows with missing critical columns: {critical_columns}")
        
        # Handle boolean columns - fill NaN with False
        boolean_columns = self.config['boolean_columns']
        for col in boolean_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(False)
        
        # Handle is_new_development based on built_year
        if 'is_new_development' in df_processed.columns and 'built_year' in df_processed.columns:
            mask = (df_processed['is_new_development'].isnull()) & (df_processed['built_year'].notna())
            df_processed.loc[mask, 'is_new_development'] = False
        
        # Drop built_year column
        if 'built_year' in df_processed.columns:
            df_processed = df_processed.drop(columns='built_year')
        
        # Fill is_new_development with mode
        if 'is_new_development' in df_processed.columns:
            dev_mode = df_processed['is_new_development'].mode()
            df_processed['is_new_development'] = df_processed['is_new_development'].fillna(
                dev_mode.iloc[0] if len(dev_mode) > 0 else False
            )
        
        # Fill house_type_id with "Misc"
        if 'house_type_id' in df_processed.columns:
            df_processed['house_type_id'] = df_processed['house_type_id'].fillna("Misc")
        
        # Extract district and neighborhood from neighborhood_id
        if 'neighborhood_id' in df_processed.columns:
            df_processed['district_id'] = df_processed['neighborhood_id'].copy()
            df_processed['district_id'] = df_processed['district_id'].str.extract(r'(District \d+)')
            df_processed['neighborhood_id'] = df_processed['neighborhood_id'].str.extract(r'(Neighborhood \d+)')
            df_processed['district_id'] = df_processed['district_id'].str.extract(r'(\d+)')
            df_processed['neighborhood_id'] = df_processed['neighborhood_id'].str.extract(r'(\d+)')
            df_processed = df_processed.drop(columns='neighborhood_id')
        
        # Convert has_lift and is_exterior to numeric
        if 'has_lift' in df_processed.columns:
            df_processed['has_lift'] = pd.to_numeric(df_processed['has_lift'], errors='coerce')
        if 'is_exterior' in df_processed.columns:
            df_processed['is_exterior'] = pd.to_numeric(df_processed['is_exterior'], errors='coerce')
        
        logger.info(f"Dataframe prepared. Shape: {df_processed.shape}")
        return df_processed
    
    def _create_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create scikit-learn preprocessing pipeline."""
        logger.info("Creating preprocessing pipeline...")
        
        # Identify column types
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if present
        target_col = self.config['target_column']
        if target_col in numeric_columns:
            numeric_columns.remove(target_col)
        if target_col in categorical_columns:
            categorical_columns.remove(target_col)
        
        logger.info(f"Numeric columns: {numeric_columns}")
        logger.info(f"Categorical columns: {categorical_columns}")
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        
        return pipeline
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the preprocessing pipeline and transform the data."""
        logger.info("Fitting and transforming data...")
        
        # Prepare dataframe
        X_processed = self._prepare_dataframe(X)
        
        # Store target column
        self.target_column = self.config['target_column']
        
        # Create and fit preprocessing pipeline
        self.preprocessing_pipeline = self._create_preprocessing_pipeline(X_processed)
        
        # Fit and transform features
        X_transformed = self.preprocessing_pipeline.fit_transform(X_processed)
        
        # Get feature names
        self.feature_columns = self._get_feature_names()
        
        logger.info(f"Data transformed. Shape: {X_transformed.shape}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        
        return X_transformed, y.values if y is not None else None
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted pipeline."""
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        
        # Prepare dataframe
        X_processed = self._prepare_dataframe(X)
        
        # Transform features
        X_transformed = self.preprocessing_pipeline.transform(X_processed)
        
        logger.info(f"Data transformed. Shape: {X_transformed.shape}")
        return X_transformed
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if self.preprocessing_pipeline is None:
            return []
        
        # Get feature names from the preprocessor
        preprocessor = self.preprocessing_pipeline.named_steps['preprocessor']
        
        feature_names = []
        
        # Add numeric feature names
        numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        feature_names.extend(numeric_features)
        
        # Add categorical feature names
        categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        feature_names.extend(categorical_features)
        
        return feature_names.tolist()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        return self.feature_columns
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the preprocessing pipeline to disk."""
        if self.preprocessing_pipeline is None:
            raise ValueError("No pipeline to save. Call fit_transform first.")
        
        joblib.dump(self.preprocessing_pipeline, filepath)
        logger.info(f"Preprocessing pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load a preprocessing pipeline from disk."""
        self.preprocessing_pipeline = joblib.load(filepath)
        logger.info(f"Preprocessing pipeline loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    from data_loader import load_data, split_data
    
    # Load data
    df = load_data("houses_Madrid.csv")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, 'buy_price', test_size=0.2)
    
    # Initialize preprocessor
    preprocessor = MadridHousingPreprocessor()
    
    # Fit and transform training data
    X_train_transformed, y_train_transformed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)
    
    print(f"Training data shape: {X_train_transformed.shape}")
    print(f"Test data shape: {X_test_transformed.shape}")
    print(f"Feature names: {len(preprocessor.get_feature_names())}")
