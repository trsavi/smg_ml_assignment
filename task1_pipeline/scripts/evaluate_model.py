#!/usr/bin/env python3
"""
Model evaluation script for Madrid Housing Market pipeline.

This script evaluates the trained model and shows performance metrics.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train import MadridHousingTrainer

def main():
    """Evaluate the trained model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Madrid Housing Market model')
    parser.add_argument('--mode', type=str, choices=['single', 'experiments'], default='single',
                       help='Evaluation mode: "single" for single training pipeline, "experiments" for multiple experiments (default: single)')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Custom name for the training run (only used in single mode)')
    
    args = parser.parse_args()
    
    print(f"Evaluating model in {args.mode} mode...")
    
    try:
        # Initialize trainer
        trainer = MadridHousingTrainer()
        
        # Check if preprocessed data exists, if not prepare it
        if not trainer._check_preprocessed_data():
            print("Preprocessed data not found. Preparing data first...")
            trainer._prepare_data_if_needed()
        
        if args.mode == 'single':
            # Run single training pipeline (which includes evaluation)
            results = trainer.run_training_pipeline(run_name=args.run_name)
            
            print(f"Single evaluation completed!")
            print(f"Test RMSE: {results['metrics']['rmse']:.2f}")
            print(f"Test MAE: {results['metrics']['mae']:.2f}")
            print(f"Test R²: {results['metrics']['r2']:.3f}")
            print(f"Test MAPE: {results['metrics']['mape']:.2f}%")
            
        elif args.mode == 'experiments':
            # Run multiple experiments
            results = trainer.run_multiple_experiments()
            
            print(f"Multiple experiments evaluation completed!")
            print(f"Total experiments: {len(results)}")
            print("\nResults Summary:")
            print("-" * 60)
            
            for exp_name, result in results.items():
                if 'error' not in result:
                    print(f"  {exp_name}:")
                    print(f"    RMSE: {result['metrics']['rmse']:.2f}")
                    print(f"    MAE: {result['metrics']['mae']:.2f}")
                    print(f"    R²: {result['metrics']['r2']:.3f}")
                    print(f"    MAPE: {result['metrics']['mape']:.2f}%")
                    print(f"    Run ID: {result['run_id']}")
                    if result.get('description'):
                        print(f"    Description: {result['description']}")
                    print()
                else:
                    print(f"  {exp_name}: FAILED - {result['error']}")
                    print()
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
