#!/usr/bin/env python3
"""
Model versioning utilities for Madrid Housing Market pipeline.

This module provides utilities for managing model versions, saving models with versioning,
and tracking model metadata.
"""

# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Local imports
from utils.file_manager import FileManager


class ModelVersioningManager:
    """Utility class for managing model versioning and storage."""
    
    def __init__(self, file_manager: FileManager = None):
        """Initialize the model versioning manager.
        
        Args:
            file_manager: FileManager instance for file operations
        """
        self.file_manager = file_manager or FileManager()
    
    def save_model_with_versioning(self, model, model_path: str = "models/madrid_housing_model.pkl") -> None:
        """Save trained model and preprocessor with versioning.
        
        Args:
            model: Trained model to save
            model_path: Path for the best model (default: models/madrid_housing_model.pkl)
        """
        if model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create trained_models directory if it doesn't exist
        trained_models_dir = Path("trained_models")
        trained_models_dir.mkdir(exist_ok=True)
        
        # Generate versioned model name
        model_name = Path(model_path).stem  # Get name without extension
        model_ext = Path(model_path).suffix  # Get extension
        versioned_model_name = f"{model_name}_{timestamp}{model_ext}"
        versioned_model_path = trained_models_dir / versioned_model_name
        
        # Save versioned model in trained_models directory
        self.file_manager.save_model(model, str(versioned_model_path))
        print(f"Versioned model saved to {versioned_model_path}")
        
        # Save the same model as the "best" model in models directory
        self.file_manager.save_model(model, model_path)
        print(f"Best model saved to {model_path}")
        
        # Store versioning information
        self._save_version_info(timestamp, str(versioned_model_path), model_path, model)
    
    def save_experiment_model(self, model, experiment_name: str, metrics: Dict[str, float] = None) -> str:
        """Save an experiment model with versioning.
        
        Args:
            model: Trained model to save
            experiment_name: Name of the experiment
            metrics: Performance metrics for the model
            
        Returns:
            Path to the saved model
        """
        if model is None:
            raise ValueError("No model to save.")
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create trained_models directory if it doesn't exist
        trained_models_dir = Path("trained_models")
        trained_models_dir.mkdir(exist_ok=True)
        
        # Generate experiment model name
        exp_model_name = f"madrid_housing_model_{experiment_name}_{timestamp}.pkl"
        exp_model_path = trained_models_dir / exp_model_name
        
        # Save versioned model
        self.file_manager.save_model(model, str(exp_model_path))
        print(f"Experiment model saved to {exp_model_path}")
        
        # Save version info for this experiment
        exp_version_info = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "versioned_model_path": str(exp_model_path),
            "model_type": type(model).__name__,
            "created_at": datetime.now().isoformat()
        }
        
        # Add metrics if provided
        if metrics:
            exp_version_info.update(metrics)
        
        exp_version_info_path = trained_models_dir / f"version_info_{experiment_name}_{timestamp}.json"
        self.file_manager.save_json(exp_version_info, str(exp_version_info_path))
        print(f"Experiment version info saved to {exp_version_info_path}")
        
        return str(exp_model_path)
    
    def save_best_model_from_experiments(self, model, best_experiment_name: str, 
                                       best_val_rmse: float, model_path: str = "models/madrid_housing_model.pkl") -> None:
        """Save the best model from experiments to the models directory.
        
        Args:
            model: Best model to save
            best_experiment_name: Name of the best experiment
            best_val_rmse: Validation RMSE of the best model
            model_path: Path for the best model
        """
        if model is None:
            raise ValueError("No model to save.")
        
        # Save with versioning (this will save to both trained_models/ and models/)
        self.save_model_with_versioning(model, model_path)
        
        # Update the latest version info to reflect this is the best from experiments
        latest_version_info = self.get_latest_model_info()
        if latest_version_info:
            latest_version_info['best_from_experiment'] = best_experiment_name
            latest_version_info['best_val_rmse'] = best_val_rmse
            latest_version_path = Path("trained_models") / "latest_version.json"
            self.file_manager.save_json(latest_version_info, str(latest_version_path))
    
    def _save_version_info(self, timestamp: str, versioned_path: str, best_path: str, model) -> None:
        """Save versioning information to a JSON file.
        
        Args:
            timestamp: Timestamp for versioning
            versioned_path: Path to the versioned model
            best_path: Path to the best model
            model: Model instance for type information
        """
        version_info = {
            "timestamp": timestamp,
            "versioned_model_path": versioned_path,
            "best_model_path": best_path,
            "model_type": type(model).__name__ if model else "Unknown",
            "created_at": datetime.now().isoformat()
        }
        
        # Save version info to trained_models directory
        version_info_path = Path("trained_models") / f"version_info_{timestamp}.json"
        self.file_manager.save_json(version_info, str(version_info_path))
        print(f"Version info saved to {version_info_path}")
        
        # Update the latest version info
        latest_version_path = Path("trained_models") / "latest_version.json"
        self.file_manager.save_json(version_info, str(latest_version_path))
        print(f"Latest version info updated at {latest_version_path}")
    
    def list_trained_models(self) -> List[Dict[str, Any]]:
        """List all trained models with their version information.
        
        Returns:
            List of model information dictionaries
        """
        trained_models_dir = Path("trained_models")
        if not trained_models_dir.exists():
            print("No trained_models directory found")
            return []
        
        models = []
        for version_file in trained_models_dir.glob("version_info_*.json"):
            try:
                version_info = self.file_manager.load_json(str(version_file))
                models.append(version_info)
            except Exception as e:
                print(f"Error loading version info from {version_file}: {e}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return models
    
    def get_latest_model_info(self) -> Dict[str, Any]:
        """Get information about the latest trained model.
        
        Returns:
            Dictionary with latest model information
        """
        latest_version_path = Path("trained_models") / "latest_version.json"
        if not latest_version_path.exists():
            print("No latest version info found")
            return {}
        
        try:
            return self.file_manager.load_json(str(latest_version_path))
        except Exception as e:
            print(f"Error loading latest version info: {e}")
            return {}
    
    def find_best_model_from_experiments(self, experiment_results: Dict[str, Any]) -> tuple:
        """Find the best model from experiment results based on validation RMSE.
        
        Args:
            experiment_results: Dictionary of experiment results
            
        Returns:
            Tuple of (best_model, best_experiment_name, best_val_rmse)
        """
        best_model = None
        best_val_rmse = float('inf')
        best_experiment_name = None
        
        for exp_name, result in experiment_results.items():
            if 'error' not in result and 'metrics' in result:
                val_rmse = result['metrics']['val']['val_rmse']
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model = result['model']
                    best_experiment_name = exp_name
        
        return best_model, best_experiment_name, best_val_rmse
