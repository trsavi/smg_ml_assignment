# SMG Engineer Assessment

This repository contains the implementation of two main tasks for the SMG Engineer Assessment:

## Project Structure

```
smg_ml_assignment/
│
├── task1_pipeline/          # Task 1: Production-Ready ML Pipeline
│   ├── src/                 # Source code modules
│   │   ├── data_loader.py   # Data loading utilities
│   │   ├── preprocessing.py # Data preprocessing pipeline
│   │   ├── train_model.py   # Model training with MLflow
│   │   └── api.py          # FastAPI model serving
│   ├── configs/             # Configuration files
│   │   ├── preprocessing_config.yaml
│   │   └── training_config.yaml
│   ├── data/                # Preprocessed datasets
│   ├── models/              # Trained models and artifacts
│   ├── scripts/             # Utility scripts
│   │   ├── data_prep.py     # Data preparation
│   │   ├── train.py         # Training interface
│   │   ├── evaluate.py      # Model evaluation
│   │   └── serve.py         # API serving
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
│   │   ├── data_loader.py   # Data loading utilities
│   │   ├── preprocessing.py # Data preprocessing pipeline
│   │   ├── train_model.py   # Core training logic (refactored)
│   │   ├── api.py          # FastAPI model serving
│   │   └── utils/           # Utility modules (NEW)
│   │       ├── data_utils.py           # Data management
│   │       ├── metrics_utils.py        # Metrics calculation
│   │       ├── model_versioning_utils.py # Model versioning
│   │       ├── mlflow_utils.py         # MLflow logging
│   │       ├── evaluation_utils.py     # Model evaluation
│   │       ├── file_manager.py         # File operations
│   │       └── api/                    # API utilities
│   │           ├── config_loader.py    # Configuration loading
│   │           ├── model_manager.py    # Model management
│   │           ├── prediction_service.py # Prediction logic
│   │           ├── json_handler.py     # JSON handling
│   │           └── models.py           # Pydantic models
│   ├── configs/             # Configuration files
│   │   ├── preprocessing_config.yaml
│   │   ├── training_config.yaml
│   │   └── api_config.json  # API configuration (NEW)
│   ├── data/                # Preprocessed datasets
│   ├── models/              # Best models (production ready)
│   ├── trained_models/      # All model versions (NEW)
│   │   ├── madrid_housing_model_*.pkl  # Versioned models
│   │   ├── version_info_*.json         # Version metadata
│   │   └── latest_version.json         # Latest model info
│   ├── scripts/             # Utility scripts
│   │   ├── data_prep.py     # Data preparation
│   │   ├── train.py         # Training interface
│   │   ├── evaluate.py      # Model evaluation
│   │   └── serve.py         # API serving
│   ├── tests/               # Comprehensive test suite
│   │   ├── unittest/        # Unit tests
│   │   └── integration/     # Integration tests
│   ├── api_test_cases/      # API test cases
│   ├── mlruns/              # MLflow tracking data
│   ├── htmlcov/             # Coverage reports
│   ├── Makefile             # Build automation
│   ├── requirements.txt     # Python dependencies
│   ├── pytest.ini          # Pytest configuration
│   ├── Dockerfile           # Container configuration
│   ├── docker-compose.yml   # Multi-container setup
│   ├── API_REFACTORING_GUIDE.md # API refactoring documentation
│   ├── REFACTORING_SUMMARY.md   # Refactoring documentation
│   ├── TESTING_SUMMARY.md       # Testing documentation
│   ├── UNITTEST_ORGANIZATION_SUMMARY.md # Test organization docs
│   └── README.md            # Task 2 documentation
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

# Install dependencies
make install

# Run all tests with coverage
make test-coverage

# View HTML coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS

# Run specific test modules
make test-file-manager
make test-api
make test-train

# Full pipeline with testing
make pipeline
```

**Test Coverage**: 80% overall coverage with 230+ unit tests
**Coverage Reports**: Interactive HTML reports and XML for CI/CD

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
- **Modular Architecture**: Refactored code with utility modules
- **Model Versioning System**: Automatic versioning with best model selection
- **Comprehensive Testing**: 230+ unit tests with 80% code coverage
- **API Refactoring**: Clean separation of concerns with utility modules
- **Code Quality**: Linting, formatting, and type hints
- **Documentation**: Extensive documentation and guides
- **Performance Tracking**: Enhanced MLflow integration with comprehensive metrics
- **Docker Support**: Production-ready containerization
- **Test Coverage**: HTML and XML coverage reports with detailed analysis
- **Future Improvements**: Detailed roadmap for enhancements

## Key Improvements in Task 2

### 🏗️ Modular Architecture
- **Utility Modules**: Separated helper methods into focused utility classes
- **Clean Separation**: Core training logic isolated from helper functions
- **Reusability**: Utility modules can be used across different parts of the system

### 🔄 Model Versioning System
- **Automatic Versioning**: Every training creates timestamped model versions
- **Best Model Selection**: Experiments automatically select the best performing model
- **Dual Storage**: Best model in `models/`, all versions in `trained_models/`
- **Performance Tracking**: Each version includes comprehensive metrics
- **Metadata Management**: JSON files with version information and performance data

### 🧪 Enhanced Testing
- **Test Separation**: Unit tests and integration tests in separate directories
- **Comprehensive Coverage**: 80% overall code coverage with 230+ unit tests
- **Test Organization**: Clear structure for different types of tests
- **Coverage Reports**: HTML and XML coverage reports for detailed analysis
- **Test Commands**: Comprehensive Makefile and pytest commands for testing

### 🔧 API Refactoring
- **Configuration Management**: Externalized API configuration
- **Service Separation**: Prediction logic separated from API routes
- **Modular Design**: Clean separation of concerns with utility modules

### 📊 Performance Tracking
- **Comprehensive Metrics**: Train/validation/test metrics logging
- **MLflow Integration**: Enhanced experiment tracking
- **Model Registry**: Production-ready model management

## Requirements

- Python 3.8+
- See individual task requirements.txt files for specific dependencies

## Contact

For questions about this assessment, please refer to the individual task README files.
