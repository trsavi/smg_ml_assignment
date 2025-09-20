# SMG Engineer Assessment

This repository contains the implementation of two main tasks for the SMG Engineer Assessment:

## Project Structure

```
smg_ml_assignment/
│
├── task1_pipeline/          # Task 1: Production-Ready ML Pipeline
│   ├── src/                 # Source code modules
│   ├── configs/             # Configuration files
│   ├── data/                # Preprocessed datasets
│   ├── models/              # Trained models and artifacts
│   ├── scripts/             # Utility scripts
│   ├── test_cases/          # Test case files
│   ├── api_test_cases/      # API test cases
│   ├── mlruns/              # MLflow tracking data
│   ├── Makefile             # Build automation
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Container configuration
│   ├── docker-compose.yml   # Multi-container setup
│   └── README.md            # Task 1 documentation
│
├── task2_code_quality/      # Task 2: Refactoring & Engineering Excellence
│   ├── src/                 # Refactored source code
│   └── tests/               # Test suite
│
└── README.md                # This file
```

## Quick Start

### Task 1: Production-Ready ML Pipeline
A complete machine learning pipeline for Madrid Housing Market price prediction with MLflow tracking and FastAPI serving.

```bash
cd task1_pipeline
make install
make pipeline
make serve
```

### Task 2: Code Quality & Engineering Excellence
Refactored code with proper testing, linting, and CI/CD integration.

```bash
cd task2_code_quality
# Implementation details in task2_code_quality/README.md
```

## Task Overview

### Task 1: Production-Ready ML Pipeline
- Data loading and validation
- Scikit-learn preprocessing pipeline
- LightGBM training with MLflow
- FastAPI model serving
- Comprehensive documentation
- Automation with Makefile
- Docker containerization
- MLflow experiment tracking
- API testing framework

### Task 2: Code Quality & Engineering Excellence
- Code refactoring and optimization
- Comprehensive testing suite
- Linting and formatting (Black, Flake8, MyPy)
- Pre-commit hooks
- CI/CD pipeline
- Documentation improvements

## Requirements

- Python 3.8+
- See individual task requirements.txt files for specific dependencies

## Contact

For questions about this assessment, please refer to the individual task README files.
