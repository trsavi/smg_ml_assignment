#!/usr/bin/env python3
"""
Data preparation script for Madrid Housing Market pipeline.

This script prepares and preprocesses the dataset for training.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import load_data
from preprocessing import MadridHousingPreprocessor


def main():
    """Prepare and preprocess the dataset."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare and preprocess Madrid Housing Market dataset')
    parser.add_argument('--store', action='store_true', 
                       help='Store the preprocessed dataset to a file')
    parser.add_argument('--output-path', type=str, default='data/preprocessed_houses_Madrid.csv',
                       help='Path to store the preprocessed dataset (default: data/preprocessed_houses_Madrid.csv)')
    
    args = parser.parse_args()
    
    print("Preparing data...")

    try:
        # Load data
        df = load_data("src/houses_Madrid.csv")
        print(f"Data loaded: {df.shape}")

        # Initialize preprocessor
        preprocessor = MadridHousingPreprocessor()

        # Transform data (this will fit and transform in one step)
        df_processed = preprocessor.prepare_data(df, is_training=True)

        print(f"Data prepared: {df_processed.shape}")
        print(f"Number of features: {df_processed.shape[1]}")
        
        # Store preprocessed data if requested
        if args.store:
            # Create output directory if it doesn't exist
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save preprocessed data
            df_processed.to_csv(output_path, index=False)
            print(f"Preprocessed data saved to: {output_path}")
        
        print("Data preparation completed successfully!")

    except Exception as e:
        print(f"Error preparing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()