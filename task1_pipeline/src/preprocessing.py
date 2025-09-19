"""
Preprocessing module for Madrid Housing Market dataset.

This module prepares the dataset following the specified steps and
stores preprocessing parameters for consistent training/test handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging
import yaml
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MadridHousingPreprocessor:
    """Preprocessing pipeline for Madrid Housing Market dataset."""

    def __init__(self, config_path: str = "configs/preprocessing_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.preprocessing_params: Dict[str, Any] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load preprocessing configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            return {'target_column': 'buy_price', 'columns_to_drop': [], 
                    'boolean_columns': [], 'critical_columns': []}

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Prepare data for ML models like LightGBM."""
        df_processed = df.copy()

        # Step 1: Drop specified columns
        drop_cols = [c for c in self.config.get('columns_to_drop', []) if c in df_processed]
        df_processed.drop(columns=drop_cols, inplace=True)

        # Step 2: Drop rows with missing critical values
        for col in self.config.get('critical_columns', []):
            if col in df_processed:
                df_processed = df_processed[df_processed[col].notna()]

        # Step 3: Fill NaN in boolean columns with False
        for col in self.config.get('boolean_columns', []):
            if col in df_processed:
                # Convert to boolean first, then fill NaN
                df_processed[col] = df_processed[col].astype('boolean')
                df_processed[col] = df_processed[col].fillna(False)

        # Step 4: Handle `is_new_development`
        if {'is_new_development', 'built_year'}.issubset(df_processed):
            mask = df_processed['is_new_development'].isna() & df_processed['built_year'].notna()
            df_processed.loc[mask, 'is_new_development'] = False
            df_processed.drop(columns='built_year', inplace=True)

            if is_training:
                mode_val = df_processed['is_new_development'].mode(dropna=True)
                self.preprocessing_params['is_new_development_mode'] = mode_val.iat[0] if not mode_val.empty else False
            # Convert to boolean first, then fill NaN
            df_processed['is_new_development'] = df_processed['is_new_development'].astype('boolean')
            df_processed['is_new_development'] = df_processed['is_new_development'].fillna(
                self.preprocessing_params.get('is_new_development_mode', False)
            )

        # Step 5: Fill house_type_id
        if 'house_type_id' in df_processed:
            df_processed['house_type_id'] = df_processed['house_type_id'].fillna("Misc").infer_objects(copy=False)

        # Step 6: Extract district/neighborhood
        if 'neighborhood_id' in df_processed:
            df_processed['district_id'] = df_processed['neighborhood_id'].str.extract(r'District (\d+)')
            df_processed.drop(columns='neighborhood_id', inplace=True)

        # Step 7: Convert lift/exterior to numeric
        for col in ['has_lift', 'is_exterior']:
            if col in df_processed:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # Step 8: Clean categorical columns and create dummies
        cat_cols = [c for c in ['house_type_id', 'district_id'] if c in df_processed]
        if cat_cols:
            # Clean categorical values to remove special characters
            for col in cat_cols:
                if col in df_processed:
                    # Replace special characters with underscores
                    df_processed[col] = df_processed[col].astype(str).str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
                    # Remove multiple consecutive underscores
                    df_processed[col] = df_processed[col].str.replace(r'_+', '_', regex=True)
                    # Remove leading/trailing underscores
                    df_processed[col] = df_processed[col].str.strip('_')
            
            df_processed = pd.get_dummies(df_processed, columns=cat_cols)

        # Step 9: Handle has_lift / is_exterior
        for col in ['has_lift', 'is_exterior']:
            if col in df_processed:
                if is_training:
                    mode_val = df_processed[col].mode(dropna=True)
                    self.preprocessing_params[f'{col}_mode'] = mode_val.iat[0] if not mode_val.empty else 0
                df_processed[col] = df_processed[col].fillna(self.preprocessing_params.get(f'{col}_mode', 0)).infer_objects(copy=False)

        # Step 10: Drop final dummy columns
        final_drop = ['house_type_id_Misc', 'district_id_21']
        drop_final = [c for c in final_drop if c in df_processed]
        if is_training:
            self.preprocessing_params['columns_to_drop_final'] = drop_final
        else:
            drop_final = self.preprocessing_params.get('columns_to_drop_final', drop_final)
        df_processed.drop(columns=drop_final, inplace=True, errors='ignore')

        logger.info(f"Prepared data shape: {df_processed.shape}")
        return df_processed

    def fit(self, X: pd.DataFrame) -> None:
        """Fit preprocessing params on training data."""
        _ = self.prepare_data(X, is_training=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing using fitted params."""
        return self.prepare_data(X, is_training=False)

    def save_pipeline(self, filepath: str) -> None:
        joblib.dump(self.preprocessing_params, filepath)
        logger.info(f"Saved preprocessing params → {filepath}")

    def load_pipeline(self, filepath: str) -> None:
        self.preprocessing_params = joblib.load(filepath)
        logger.info(f"Loaded preprocessing params ← {filepath}")


# Example usage
if __name__ == "__main__":
    from data_loader import load_data, split_data

    df = load_data("houses_Madrid.csv")
    X_train, X_test, y_train, y_test = split_data(df, 'buy_price', test_size=0.2)

    preprocessor = MadridHousingPreprocessor()
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print(f"Train: {X_train_proc.shape}, Test: {X_test_proc.shape}")
