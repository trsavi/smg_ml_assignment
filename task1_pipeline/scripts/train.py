#!/usr/bin/env python3
"""
Training script interface for Madrid Housing Market pipeline.

This script provides a simple interface to run training with different modes:
- Single model training
- Multiple experiments (from config)
- Grid search hyperparameter tuning (from config)
"""

import sys
import os
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train_model import MadridHousingTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to run training with different modes."""
    parser = argparse.ArgumentParser(
        description='Train Madrid Housing Market model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s single                    # Train single model
  %(prog)s experiments               # Run multiple experiments from config
  %(prog)s grid-search               # Run grid search from config
  %(prog)s single --run-name "exp1"  # Train with custom run name
  %(prog)s single --config custom_config.yaml  # Use custom config
        """
    )
    
    parser.add_argument('mode', choices=['single', 'experiments', 'grid-search'],
                       help='Training mode: single model, multiple experiments, or grid search')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training config file (default: configs/training_config.yaml)')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom name for the training run')
    
    args = parser.parse_args()
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Madrid Housing Market Training - Mode: {args.mode}")
    print(f"Working directory: {project_root}")
    print(f"Config file: {args.config}")
    
    if args.run_name:
        print(f"Run name: {args.run_name}")
    
    try:
        # Initialize trainer
        trainer = MadridHousingTrainer(config_path=args.config)
        
        # Run the appropriate training mode
        if args.mode == 'single':
            print("\n" + "=" * 60)
            print("SINGLE MODEL TRAINING")
            print("=" * 60)
            
            results = trainer.run_training_pipeline(run_name=args.run_name)
            
            print(f"\nTraining completed successfully!")
            print(f"Run ID: {results['run_id']}")
            print(f"Model saved to: models/madrid_housing_model.pkl")
            print("To evaluate the model, run: python scripts/evaluate.py")
            print("To view MLflow UI: mlflow ui --backend-store-uri ./mlruns --port 5000")
            
        elif args.mode == 'experiments':
            print("\n" + "=" * 60)
            print("MULTIPLE EXPERIMENTS TRAINING")
            print("=" * 60)
            
            # Check if experiments are configured
            if 'experiments' not in trainer.config or len(trainer.config['experiments']) <= 1:
                print("No multiple experiments configured. Running single training instead.")
                results = trainer.run_training_pipeline(run_name=args.run_name)
                print(f"\nTraining completed! Run ID: {results['run_id']}")
                return
            
            results = trainer.run_multiple_experiments()
            
            print(f"\nMultiple experiments training completed!")
            print(f"Total experiments: {len(results)}")
            print("\nResults Summary:")
            print("-" * 60)
            
            for exp_name, result in results.items():
                if 'error' not in result:
                    print(f"  {exp_name}:")
                    print(f"    Run ID: {result['run_id']}")
                    if result.get('description'):
                        print(f"    Description: {result['description']}")
                    print()
                else:
                    print(f"  {exp_name}: FAILED - {result['error']}")
                    print()
            
            print("To evaluate the models, run: python scripts/evaluate.py")
            print("To view MLflow UI: mlflow ui --backend-store-uri ./mlruns --port 5000")
            
        elif args.mode == 'grid-search':
            print("\n" + "=" * 60)
            print("GRID SEARCH HYPERPARAMETER TUNING")
            print("=" * 60)
            
            # Check if grid search is enabled in config
            grid_config = trainer.config.get('grid_search', {})
            if not grid_config.get('enabled', False):
                print("Grid search is not enabled in config.")
                print("To enable grid search, set 'grid_search.enabled: true' in your config file.")
                print("Example config:")
                print("""
                    grid_search:
                    enabled: true
                    parameters:
                        learning_rate: [0.05, 0.1, 0.15]
                        num_leaves: [31, 50, 100]
                        max_depth: [6, 8, 10]
                        feature_fraction: [0.8, 0.9, 1.0]
                        """)
                sys.exit(1)
            
            results = trainer.run_grid_search()
            
            print(f"\nGrid search completed!")
            print(f"Best parameters: {results['best_params']}")
            print(f"Best validation RMSE: {results['best_score']:.2f}")
            print(f"Best run ID: {results['best_run_id']}")
            print("Best model saved to: models/madrid_housing_model.pkl")
            print("To evaluate the best model, run: python scripts/evaluate.py")
            print("To view MLflow UI: mlflow ui --backend-store-uri ./mlruns --port 5000")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()