# Madrid Housing Market Price Prediction Pipeline

A production-ready machine learning pipeline for predicting housing prices in Madrid using LightGBM regression and FastAPI model serving.

## Architecture

### ML Pipeline Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Preprocessing   │    │   Training      │
│                 │    │                  │    │                 │
│ • data_loader   │───▶│ • preprocessing  │───▶│ • train.py     │
│ • CSV loading   │    │ • Pipeline       │    │ • Hyperparams   │
│ • Validation    │    │ • Scaling        │    │ • Model training│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Model API     │    │   Evaluation     │    │   Model Store   │
│                 │    │                  │    │                 │
│ • FastAPI       │◀───│ • Metrics        │◀───│ • Versioning    │
│ • REST endpoints│    │ • Validation     │    │ • trained_models│
│ • Pydantic      │    │ • Cross-val      │    │ • models/       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Modular Architecture (Refactored)
```
┌─────────────────────────────────────────────────────────────────┐
│                        Core Training Module                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   train_model   │  │   Data Utils    │  │  Metrics Utils  │ │
│  │   • Core logic  │  │   • Data mgmt   │  │   • Calculation │ │
│  │   • Pipelines   │  │   • Splits      │  │   • Validation  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                     │                     │         │
│           ▼                     ▼                     ▼         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Versioning Utils│  │  MLflow Utils   │  │Evaluation Utils │ │
│  │ • Model storage │  │   • Logging     │  │   • Testing     │ │
│  │ • Version mgmt  │  │   • Tracking    │  │   • Metrics     │ │
│  │ • Best model    │  │   • Experiments │  │   • Reports     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Versioning System                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ trained_models/ │  │    models/      │  │ Version Info    │ │
│  │ • All versions  │  │ • Best model    │  │ • JSON metadata │ │
│  │ • Timestamps    │  │ • Production    │  │ • Performance   │ │
│  │ • Experiments   │  │ • API ready     │  │ • Tracking      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Docker Deployment Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Container                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Python 3.9    │  │   FastAPI App   │  │   LightGBM      │ │
│  │   Base Image    │  │   (Port 8000)   │  │   Model         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                     │                     │         │
│           ▼                     ▼                     ▼         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Dependencies  │  │   API Endpoints │  │   Model Files   │ │
│  │   • pandas      │  │   • /health     │  │   • .pkl files  │ │
│  │   • scikit-learn│  │   • /predict    │  │   • configs     │ │
│  │   • lightgbm    │  │   • /batch_predict│  │   • data       │ │
│  │   • fastapi     │  │   • /docs       │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Host System (Port 8000)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Docker        │  │   API Client    │  │   Monitoring    │ │
│  │   Compose       │  │   • curl        │  │   • Health      │ │
│  │   • build       │  │   • Python      │  │   • Logs        │ │
│  │   • up/down     │  │   • Postman     │  │   • Metrics     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### API Request Flow
```
Client Request → Docker Container → FastAPI → Model Prediction → JSON Response
     │                │                │            │              │
     ▼                ▼                ▼            ▼              ▼
┌─────────┐    ┌─────────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  curl   │───▶│  Port 8000  │─▶│ FastAPI │─▶│ LightGBM│─▶│  JSON   │
│ Python  │    │  Mapping    │  │ Router  │  │ Model   │  │ Response│
│ Postman │    │  Container  │  │ Handler │  │ Predict │  │  Data   │
└─────────┘    └─────────────┘  └─────────┘  └─────────┘  └─────────┘
```

## Model Versioning System

### Overview
The pipeline includes a comprehensive model versioning system that automatically manages model versions and tracks performance metrics.

### Directory Structure
```
task2_code_quality/
├── models/                          # Production models
│   ├── madrid_housing_model.pkl    # Best model (API ready)
│   └── feature_importance.csv      # Feature importance
├── trained_models/                  # All model versions
│   ├── madrid_housing_model_20250922_213437.pkl
│   ├── madrid_housing_model_baseline_model_20250922_212224.pkl
│   ├── version_info_*.json         # Version metadata
│   └── latest_version.json         # Latest model info
└── mlruns/                         # MLflow experiment tracking
```

### Versioning Features
- **Automatic Versioning**: Every training creates timestamped versions
- **Best Model Selection**: Experiments automatically select the best performing model
- **Dual Storage**: Best model in `models/`, all versions in `trained_models/`
- **Performance Tracking**: Each version includes validation metrics
- **Experiment Tracking**: Knows which experiment produced the best model
- **Metadata Storage**: JSON files with version information and performance metrics

### Model Management
```bash
# Single training (creates versioned model + best model)
python scripts/train.py single --run-name "my_experiment"

# Multiple experiments (saves all + selects best)
python scripts/train.py experiments

# Grid search (saves best from search)
python scripts/train.py grid-search
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd task2_code_quality

# Install dependencies (choose one method)
make install                    # Using Makefile
pip install -r requirements.txt # Direct pip install
```

## Docker Deployment

### Prerequisites

- Docker Desktop installed and running
- At least 4GB RAM available for Docker
- Port 8000 available on your system

### Docker Commands

#### 1. Build Docker Image

```bash
# Build the Docker image
docker build -t madrid-housing-api .

# Or using docker-compose
docker-compose build
```

#### 2. Run Docker Container

```bash
# Method 1: Using docker-compose (Recommended)
docker-compose up

# Method 2: Run in background
docker-compose up -d

# Method 3: Direct docker run
docker run -p 8000:8000 madrid-housing-api
```

#### 3. Test Docker Container

```bash
# Test all endpoints
python test_docker_predictions.py

# Test individual endpoints
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info
```

#### 4. Stop Docker Container

```bash
# Stop and remove containers
docker-compose down

# Stop all running containers
docker stop $(docker ps -q)
```

### Docker Troubleshooting

#### Container Won't Start

```bash
# Check if Docker is running
docker --version
docker ps

# Check container logs
docker-compose logs
docker logs <container_name>

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

#### Port Already in Use

```bash
# Check what's using port 8000
netstat -ano | findstr :8000

# Kill process using port 8000 (Windows)
taskkill /PID <process_id> /F

# Or use different port
docker run -p 8001:8000 madrid-housing-api
```

#### Model Not Loading

```bash
# Check if model file exists
ls models/madrid_housing_model.pkl

# Rebuild container to include model
docker-compose down
docker-compose up --build
```

### Docker API Endpoints

Once the container is running, the API is available at:

- **Health Check**: http://127.0.0.1:8000/health
- **Model Info**: http://127.0.0.1:8000/model/info
- **Single Prediction**: http://127.0.0.1:8000/predict
- **Batch Prediction**: http://127.0.0.1:8000/batch_predict
- **API Documentation**: http://127.0.0.1:8000/docs

### Example API Usage

#### Single Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @api_test_cases/test_case_1.json
```

#### Batch Prediction

```bash
curl -X POST http://127.0.0.1:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d @api_test_cases/test_case_batch_prediction.json
```

### Docker Production Deployment

#### Environment Variables

```bash
# Set environment variables
export MODEL_PATH=models/madrid_housing_model.pkl
export API_HOST=0.0.0.0
export API_PORT=8000

# Run with environment variables
docker run -e MODEL_PATH=$MODEL_PATH -e API_HOST=$API_HOST -e API_PORT=$API_PORT -p 8000:8000 madrid-housing-api
```

#### Docker Compose for Production

```yaml
version: '3.8'
services:
  madrid-housing-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=models/madrid_housing_model.pkl
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 2. Data Preparation

```bash
# Prepare and preprocess the dataset (choose one method)
make prepare-data                    # Using Makefile
python scripts/data_prep.py         # Direct script execution
```

### 3. Training

```bash
# Train single model (choose one method)
make train                          # Using Makefile
python scripts/train_model.py single      # Direct script execution

# Train multiple experiments
make train-experiments              # Using Makefile
python scripts/train_model.py experiments # Direct script execution

# Train with grid search tuning
make train-grid                     # Using Makefile
python scripts/train_model.py grid-search # Direct script execution
```

### 4. Model Evaluation

```bash
# Evaluate the trained model (choose one method)
make evaluate                       # Using Makefile
python scripts/evaluate.py         # Direct script execution
```

### 5. Model Serving

```bash
# Start the FastAPI server (choose one method)
make serve                          # Using Makefile
python scripts/serve.py start      # Direct script execution

# Test specific endpoints
make test-health                    # Test health check
make test-model-info               # Test model info
make test-predict                  # Test prediction
make test-batch-predict            # Test batch prediction

# Or test directly
python scripts/serve.py health_check
python scripts/serve.py model_info
python scripts/serve.py predict
python scripts/serve.py batch_predict
```

### 6. Test the API

Once the server is running, test it with a sample prediction:

```bash
# Make a prediction request using test case
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@test_cases/test_case_1.json"
```

Expected output:
```json
{"prediction":169049.65984043336}
```

### 7. View Results

- **API Documentation**: http://localhost:8000/docs

## Project Structure

```
task1_pipeline/
├── src/
│   ├── data_loader.py      # Data loading and validation
│   ├── preprocessing.py    # Preprocessing pipeline
│   ├── train.py           # Model training
│   ├── api.py             # FastAPI model serving
│   └── houses_Madrid.csv  # Source dataset
├── configs/
│   ├── preprocessing_config.yaml
│   └── training_config.yaml
├── data/
│   └── preprocessed_houses_Madrid.csv  # Preprocessed dataset
├── models/                # Saved models and artifacts
│   ├── madrid_housing_model.pkl
│   └── feature_importance.csv
├── scripts/               # Pipeline scripts
│   ├── data_prep.py       # Data preparation
│   ├── train.py           # Model training (single, experiments, grid search)
│   ├── evaluate.py        # Model evaluation
│   └── serve.py           # API serving and testing
├── test_cases/           # Test case files
├── api_test_cases/       # API test cases
├── mlruns/               # MLflow experiment tracking
├── json_requests/        # Sample API requests
├── Makefile              # Automation commands
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container setup
└── README.md            # This file
```

## Available Commands

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install required packages |
| `make prepare-data` | Prepare and preprocess dataset |
| `make train` | Train single model |
| `make train-experiments` | Train multiple experiments |
| `make train-grid` | Train with grid search tuning |
| `make evaluate` | Evaluate trained model |
| `make serve` | Start FastAPI server |
| `make test-health` | Test health check endpoint |
| `make test-model-info` | Test model info endpoint |
| `make test-predict` | Test prediction endpoint |
| `make test-batch-predict` | Test batch prediction endpoint |
| `make clean` | Clean up generated files |
| `make pipeline` | Run complete pipeline (prepare → train → evaluate) |
| `make pipeline-experiments` | Run pipeline with multiple experiments |
| `make pipeline-grid` | Run pipeline with grid search tuning |

### Direct Script Execution

For development and debugging, you can run scripts directly:

```bash
# Data preparation
python scripts/data_prep.py

# Model training options
python scripts/train_model.py single        # Single model training
python scripts/train_model.py experiments   # Multiple experiments
python scripts/train_model.py grid-search   # Grid search tuning

# Model evaluation (requires trained model)
python scripts/evaluate.py

# API serving and testing
python scripts/serve.py start         # Start server
python scripts/serve.py health_check  # Test health check
python scripts/serve.py model_info    # Test model info
python scripts/serve.py predict       # Test prediction
python scripts/serve.py batch_predict # Test batch prediction

# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000
```

### Script Options

Most scripts support command-line arguments for customization:

```bash
# Training with custom experiment name
python .\scripts\train_model.py single --run-name "my_experiment"

# Training with custom config
python .\scripts\train_model.py experiments --config custom_config.yaml

# Evaluation with custom model path
python .\scripts\evaluate.py --model-path "models/my_model.pkl"

# API serving with custom host/port
python .\scripts\serve.py start --host localhost --port 8080

# Test endpoints with custom host/port
python .\scripts\serve.py predict --host localhost --port 8080

# Get help for any script
python .\scripts\data_prep.py --help
python .\scripts\train_model.py --help
python .\scripts\evaluate.py --help
python .\scripts\serve.py --help
```

### Makefile vs Direct Scripts

- **Makefile**: Simple, fast commands for basic operations
- **Direct Scripts**: More flexible with additional options, better for development and debugging

## Data Pipeline

### Data Loading (`data_loader.py`)
- Loads Madrid Housing Market dataset from CSV
- Validates data integrity and format
- Splits data into train/test sets
- Handles missing values and duplicates

### Preprocessing (`preprocessing.py`)
- **Scikit-learn Pipeline** for consistent preprocessing
- **Missing value handling**: Median imputation for numeric, mode for categorical
- **Categorical encoding**: One-hot encoding for categorical variables
- **Feature scaling**: StandardScaler for numeric features
- **Column selection**: Drops unnecessary columns based on configuration

### Training (`scripts/train.py`)
- **Multiple training modes**: Single model, experiments, grid search
- **LightGBM regression** with hyperparameter tuning
- **Grid search tuning** for key parameters (learning_rate, num_leaves, max_depth, feature_fraction)
- **MLflow tracking** for experiment management
- **Model persistence** with joblib
- **Feature importance** analysis and export
- **Separated from evaluation** - focuses only on training

### Evaluation (`scripts/evaluate.py`)
- **Loads trained models** from disk
- **Evaluates performance** on test data
- **Calculates metrics** (RMSE, MAE, R²)
- **Logs evaluation results** to MLflow
- **Independent of training** - can evaluate any trained model

### API Serving (`scripts/serve.py`)
- **Starts FastAPI server** automatically
- **Tests specific endpoints** using test case files
- **Uses predefined test cases** from `api_test_cases/` directory
- **Simple interface** with 4 clear endpoints
- **Loads test data** from JSON files automatically

## API Endpoints

### Starting the Server

First, start the API server:
```bash
# Option 1: Using Makefile
make serve

# Option 2: Direct execution
python .\src\api.py
```

The server will start on `http://127.0.0.1:8000`

### Health Check
```bash
curl.exe -s -X GET http://127.0.0.1:8000/health
```

### Model Information
```bash
curl.exe -s -X GET http://127.0.0.1:8000/health
```

### Single Prediction (Using Test Case)

Use the provided test case for quick testing:
```bash
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @api_test_cases\test_case_2.json
```

Expected response:
```json
{"prediction":169049.65984043336}
```

## Configuration

### Preprocessing Configuration (`configs/preprocessing_config.yaml`)
- Columns to drop
- Boolean columns handling
- Categorical columns for encoding
- Critical columns for filtering

### Training Configuration (`configs/training_config.yaml`)
- Data paths and parameters
- Model hyperparameters
- Training parameters

## Testing

This project includes comprehensive unit tests using the unittest framework. All tests are located in the `tests/unittest/` directory and provide complete coverage of the codebase.

### Test Structure

The project uses a **unified unittest approach** with all tests organized in the `tests/unittest/` directory:

- `test_file_manager_unittest.py` - FileManager class tests (47 tests)
- `test_preprocessing_comprehensive_unittest.py` - Preprocessing comprehensive tests (47 tests)  
- `test_preprocessing_unittest.py` - Preprocessing basic tests (25 tests)
- `test_api_unittest.py` - API endpoint tests (25 tests)
- `test_train_model_unittest.py` - Training pipeline tests (25 tests)
- `test_data_loader_unittest.py` - Data loading tests (24 tests)
- `test_metrics_utils_unittest.py` - Metrics calculation tests (15 tests)
- `test_json_handler_unittest.py` - JSON handling tests (12 tests)
- `test_model_manager_unittest.py` - Model management tests (10 tests)

**Total: 230+ comprehensive tests** covering all functionality with proper setup/teardown and error handling.

### Test Coverage

**Current Coverage: 80%** (765/959 statements covered)

#### Coverage by Module:
- **`src/utils/api/json_handler.py`**: 100% coverage (44/44 statements)
- **`src/utils/api/model_manager.py`**: 100% coverage (33/33 statements)
- **`src/utils/metrics_utils.py`**: 96% coverage (105/109 statements)
- **`src/train_model.py`**: 88% coverage (148/168 statements)
- **`src/utils/file_manager.py`**: 91% coverage (42/46 statements)
- **`src/preprocessing.py`**: 78% coverage (35/45 statements)
- **`src/api.py`**: 75% coverage (15/20 statements)
- **`src/data_loader.py`**: 70% coverage (14/20 statements)

### Test Execution

#### Run All Tests

**Linux/macOS (with Make):**
```bash
# Run all unittest tests
make test-unittest

# Run all tests with coverage
make test-coverage

# Run unit tests with coverage only
make test-coverage-unit

# Run integration tests with coverage only
make test-coverage-integration

# Run tests with HTML coverage report only
make test-html-coverage

# Run specific test modules
make test-file-manager
make test-preprocessing
make test-api
make test-train
make test-data-loader
```

**Windows (PowerShell/CMD):**
```powershell
# Run all unittest tests
python -m pytest tests/unittest/ -v

# Run all tests with coverage (terminal + HTML)
python -m pytest tests/unittest/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml

# Run tests with HTML coverage report only
python -m pytest tests/unittest/ --cov=src --cov-report=html:htmlcov

# Run tests with XML coverage report only
python -m pytest tests/unittest/ --cov=src --cov-report=xml:coverage.xml

# Run specific test modules
python -m pytest tests/unittest/test_file_manager_unittest.py -v
python -m pytest tests/unittest/test_preprocessing_comprehensive_unittest.py -v
python -m pytest tests/unittest/test_train_model_unittest.py -v
python -m pytest tests/unittest/test_api_unittest.py -v
python -m pytest tests/unittest/test_data_loader_unittest.py -v
python -m pytest tests/unittest/test_metrics_utils_unittest.py -v
python -m pytest tests/unittest/test_json_handler_unittest.py -v
python -m pytest tests/unittest/test_model_manager_unittest.py -v
```

#### Test Coverage Commands

**Comprehensive Coverage Testing:**
```bash
# Full coverage report (terminal + HTML + XML)
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml

# HTML coverage report only (opens in browser)
python -m pytest tests/ --cov=src --cov-report=html:htmlcov
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux

# XML coverage report for CI/CD
python -m pytest tests/ --cov=src --cov-report=xml:coverage.xml

# Coverage for specific modules
python -m pytest tests/unittest/ --cov=src/utils/file_manager --cov=src/preprocessing --cov-report=term-missing

# Coverage with minimum threshold (fail if below 80%)
python -m pytest tests/ --cov=src --cov-fail-under=80 --cov-report=term-missing

# Coverage for specific file patterns
python -m pytest tests/ --cov=src/utils --cov-report=term-missing
python -m pytest tests/ --cov=src/api --cov-report=term-missing
```

**Coverage Analysis:**
```bash
# Show missing lines in terminal
python -m pytest tests/ --cov=src --cov-report=term-missing

# Generate coverage report for specific test files
python -m pytest tests/unittest/test_file_manager_unittest.py --cov=src/utils/file_manager --cov-report=term-missing

# Coverage with branch analysis
python -m pytest tests/ --cov=src --cov-branch --cov-report=term-missing

# Coverage excluding certain files
python -m pytest tests/ --cov=src --cov-omit="*/tests/*,*/__pycache__/*" --cov-report=term-missing
```

#### Individual Test Execution
```bash
# Run specific test classes
python -m pytest tests/unittest/test_file_manager_unittest.py::TestFileManagerUnittest -v

# Run specific test methods
python -m pytest tests/unittest/test_file_manager_unittest.py::TestFileManagerUnittest::test_save_model_success -v

# Run tests with specific markers
python -m pytest tests/unittest/ -m "not slow" -v
```

#### Test Coverage Summary

**FileManager Tests** (`test_file_manager_unittest.py`) - 47 tests:
- ✅ Initialization and configuration
- ✅ Path resolution and file operations
- ✅ Config loading (YAML, JSON)
- ✅ Model saving/loading (joblib)
- ✅ DataFrame operations (pandas)
- ✅ Error handling and edge cases
- ✅ Unicode path handling
- ✅ Large data operations

**Preprocessing Tests** (`test_preprocessing_comprehensive_unittest.py`) - 47 tests:
- ✅ Data preparation and validation
- ✅ Missing value handling
- ✅ Boolean column processing
- ✅ Categorical encoding
- ✅ Feature engineering
- ✅ Pipeline operations
- ✅ Edge cases and error conditions
- ✅ Performance testing

**Training Tests** (`test_train_model_unittest.py`) - 25 tests:
- ✅ Model training workflows
- ✅ Configuration handling
- ✅ MLflow integration
- ✅ Grid search operations
- ✅ Model evaluation
- ✅ Data loading and preprocessing

**API Tests** (`test_api_unittest.py`) - 25 tests:
- ✅ FastAPI endpoints
- ✅ Request/response validation
- ✅ Error handling
- ✅ Model loading
- ✅ Batch predictions

**Data Loader Tests** (`test_data_loader_unittest.py`) - 24 tests:
- ✅ Data loading and validation
- ✅ Data splitting
- ✅ Data quality checks
- ✅ Error handling

### Test Structure

```
tests/
├── unittest/                                    # Unittest test files
│   ├── test_file_manager_unittest.py           # FileManager comprehensive tests (47 tests)
│   ├── test_preprocessing_comprehensive_unittest.py  # Preprocessing comprehensive tests (47 tests)
│   ├── test_preprocessing_unittest.py          # Preprocessing basic tests (25 tests)
│   ├── test_train_model_unittest.py            # Training model tests (25 tests)
│   ├── test_api_unittest.py                    # API endpoint tests (25 tests)
│   ├── test_data_loader_unittest.py            # Data loader tests (24 tests)
│   ├── test_metrics_utils_unittest.py          # Metrics calculation tests (15 tests)
│   ├── test_json_handler_unittest.py           # JSON handling tests (12 tests)
│   └── test_model_manager_unittest.py          # Model management tests (10 tests)
├── integration/                                 # Integration test files
│   ├── test_api_integration.py                 # API integration tests
│   ├── test_data_loader.py                     # Data loader integration tests
│   ├── test_preprocessing.py                   # Preprocessing integration tests
│   └── test_train_model.py                     # Training integration tests
├── htmlcov/                                     # HTML coverage reports
│   └── index.html                              # Interactive coverage report
├── coverage.xml                                 # XML coverage report for CI/CD
└── pytest.ini                                  # Pytest configuration
```

### Test Coverage

**Current test coverage: 80%** (765/959 statements covered)

#### Coverage by Module:
- **`src/utils/api/json_handler.py`**: 100% coverage (44/44 statements) ✅
- **`src/utils/api/model_manager.py`**: 100% coverage (33/33 statements) ✅
- **`src/utils/metrics_utils.py`**: 96% coverage (105/109 statements) ✅
- **`src/train_model.py`**: 88% coverage (148/168 statements) ✅
- **`src/utils/file_manager.py`**: 91% coverage (42/46 statements) ✅
- **`src/preprocessing.py`**: 78% coverage (35/45 statements) ✅
- **`src/api.py`**: 75% coverage (15/20 statements) ✅
- **`src/data_loader.py`**: 70% coverage (14/20 statements) ✅

**Total Tests**: 230+ unittest tests (all passing ✅)

#### HTML Coverage Report

A detailed HTML coverage report is generated when running tests with coverage:

```bash
# Generate HTML coverage report
python -m pytest tests/unittest/ --cov=src --cov-report=html

# Or using Make (Linux/macOS)
make test-coverage
```

**Report Location**: `htmlcov/index.html`

The HTML report provides:
- **Interactive coverage visualization** with line-by-line highlighting
- **Module-level coverage statistics** 
- **Missing line identification** for each file
- **Branch coverage analysis**
- **Function and class coverage details**

Open `htmlcov/index.html` in your web browser to view the detailed coverage report.

### Test Cases

- **API Test Cases**: Located in `api_test_cases/` directory
- **Automated Testing**: Comprehensive unittest and pytest test suites
- **Sample Requests**: JSON request examples in `json_requests/` directory
- **Coverage Reports**: HTML coverage reports in `htmlcov/` directory

### Running Tests in Development

#### Linux/macOS (with Make)
```bash
# Quick test run (no coverage)
make test-unittest

# Full test run with coverage
make test-coverage

# HTML coverage report only
make test-html-coverage

# Run specific test modules
make test-file-manager
make test-preprocessing
make test-api
make test-train
make test-data-loader
```

#### Windows (PowerShell/CMD)
```powershell
# Quick test run (no coverage)
python -m pytest tests/unittest/ -v

# Full test run with coverage
python -m pytest tests/unittest/ --cov=src --cov-report=term-missing

# Run specific test modules
python -m pytest tests/unittest/test_file_manager_unittest.py -v
python -m pytest tests/unittest/test_preprocessing_comprehensive_unittest.py -v
python -m pytest tests/unittest/test_api_unittest.py -v
python -m pytest tests/unittest/test_train_model_unittest.py -v
python -m pytest tests/unittest/test_data_loader_unittest.py -v
```

#### Windows (Batch File - Easy Mode)
```cmd
# Run all unittest tests
.\run_tests.bat unittest

# Run tests with coverage
.\run_tests.bat coverage

# Run tests with HTML coverage only
.\run_tests.bat html-coverage

# Run specific test modules
.\run_tests.bat file-manager
.\run_tests.bat preprocessing
.\run_tests.bat api
.\run_tests.bat train
.\run_tests.bat data-loader
```

#### Cross-Platform Commands
```bash
# Run tests for specific functionality
python -m pytest tests/unittest/ -k "file_manager" -v
python -m pytest tests/unittest/ -k "preprocessing" -v
python -m pytest tests/unittest/ -k "api" -v

# Run tests and stop on first failure
python -m pytest tests/unittest/ -x -v

# Run tests in parallel (if pytest-xdist is installed)
python -m pytest tests/unittest/ -n auto -v

# Run tests with specific markers
python -m pytest tests/unittest/ -m "not slow" -v
```

## Model Performance

The model typically achieves:
- **RMSE**: ~50,000-80,000 euros
- **R²**: ~0.85-0.95
- **MAE**: ~35,000-60,000 euros

## Feature Engineering

Key features used for prediction:
- **sq_mt_built**: Built surface area (square meters)
- **n_rooms**: Number of rooms
- **n_bathrooms**: Number of bathrooms
- **house_type_id**: Type of house (categorical)
- **neighborhood_id**: Neighborhood identifier (categorical)
- **Boolean features**: AC, pool, terrace, balcony, etc.

## MLflow Tracking

The pipeline includes comprehensive MLflow experiment tracking:

- **Experiment Management**: All training runs are logged with MLflow
- **Parameter Tracking**: Hyperparameters and configuration are recorded
- **Metrics Logging**: Performance metrics (RMSE, R², MAE) are tracked
- **Model Registry**: Trained models are registered and versioned
- **Artifact Storage**: Feature importance plots and model artifacts are saved

Access MLflow UI:
```bash
# Start MLflow tracking server
python -m mlflow ui --backend-store-uri ./mlruns --port 5000
```

## Production Deployment

### Docker Deployment

The project includes Docker support for containerized deployment:

```bash
# Build Docker image
docker build -t madrid-housing-api .

# Run containers
docker run -p 8000:8000 madrid-housing-api

# Or use docker-compose
docker-compose up
```

### Environment Variables

- `MODEL_PATH`: Path to saved model (default: `models/madrid_housing_model.pkl`)
- `API_HOST`: API host (default: `0.0.0.0`)
- `API_PORT`: API port (default: `8000`)

## Future Improvements

This section outlines potential enhancements and advanced features that could be implemented to further improve the Madrid Housing Market ML pipeline:

### 🔒 Security Enhancements


#### Data Security
- **Data Encryption**: Encrypt sensitive housing data at rest and in transit
- **Secure Model Storage**: Encrypt model files and implement secure key management
- **Network Security**: HTTPS enforcement, CORS configuration, and firewall rules

### 🤖 Multi-Model Architecture

#### Dynamic Model Management
- **Model Versioning System**: Automatic versioning and rollback capabilities
- **A/B Testing Framework**: Compare multiple models in production
- **Ensemble Methods**: Combine predictions from multiple models for better accuracy
- **Model Selection API**: Dynamic model selection based on performance metrics
- **Hot-Swapping**: Update models without service interruption

#### Advanced Training Pipeline
- **Automated Retraining**: Trigger retraining based on data drift detection
- **Hyperparameter Optimization**: Automated tuning using Optuna or similar tools
- **Cross-Validation**: Implement k-fold cross-validation for robust model evaluation
- **Feature Engineering Pipeline**: Automated feature selection and engineering
- **Model Interpretability**: SHAP values, LIME explanations, and feature importance

### 📊 Performance Monitoring & Alerting

#### Model Performance Tracking
- **Data Drift Detection**: Monitor input data distribution changes over time
- **Model Drift Detection**: Track prediction accuracy degradation
- **Performance Metrics Dashboard**: Real-time monitoring of model performance
- **Anomaly Detection**: Identify unusual patterns in predictions or data

#### Alerting System
- **Email Notifications**: Send alerts when model performance drops below thresholds
- **Slack/Teams Integration**: Real-time notifications to development teams
- **SMS Alerts**: Critical alerts for production issues
- **Webhook Support**: Integration with external monitoring systems
- **Escalation Policies**: Automatic escalation based on severity levels

#### Threshold-Based Triggers
```python
# Example configuration for performance monitoring
PERFORMANCE_THRESHOLDS = {
    "rmse_increase": 0.15,  # 15% increase in RMSE
    "accuracy_drop": 0.10,  # 10% drop in accuracy
    "prediction_drift": 0.20,  # 20% change in prediction distribution
    "data_drift": 0.25  # 25% change in input data distribution
}
```

### 🚀 Scalability & Performance

#### Infrastructure Improvements
- **Kubernetes Deployment**: Container orchestration for high availability
- **Load Balancing**: Distribute traffic across multiple API instances
- **Caching Layer**: Redis/Memcached for frequently accessed predictions
- **Database Integration**: Store predictions and metadata in PostgreSQL/MongoDB

#### Performance Optimization
- **Model Quantization**: Reduce model size for faster inference
- **Batch Processing**: Optimize batch prediction endpoints
- **GPU Acceleration**: CUDA support for faster model inference
- **Connection Pooling**: Efficient database and external service connections
- **CDN Integration**: Cache static assets and API responses

### 📈 Advanced Analytics

#### Business Intelligence
- **Prediction Analytics**: Track prediction trends and market insights
- **Customer Segmentation**: Analyze different housing market segments
- **Price Forecasting**: Time-series analysis for future price predictions
- **Market Analysis**: Geographic and temporal analysis of housing trends
- **ROI Tracking**: Measure business impact of model predictions

#### Data Science Enhancements
- **Feature Store**: Centralized feature management and versioning
- **Experiment Tracking**: Enhanced MLflow integration with more metrics
- **Model Registry**: Production-ready model management
- **Data Lineage**: Track data flow from source to predictions
- **Automated Reporting**: Generate insights and reports automatically

### 🔧 Development & Operations

#### CI/CD Pipeline
- **Automated Testing**: Unit, integration, and end-to-end tests
- **Code Quality Gates**: SonarQube, Black, Flake8 integration
- **Security Scanning**: SAST/DAST tools and dependency vulnerability checks
- **Performance Testing**: Load testing and stress testing automation
- **Blue-Green Deployment**: Zero-downtime deployments

#### Monitoring & Observability
- **Application Metrics**: Prometheus/Grafana for system monitoring
- **Distributed Tracing**: OpenTelemetry for request tracing
- **Log Aggregation**: ELK stack (Elasticsearch, Logstash, Kibana)
- **Health Checks**: Comprehensive health monitoring endpoints
- **SLA Monitoring**: Track and alert on service level agreements


### 📱 User Experience

#### Frontend Applications
- **Web Dashboard**: React/Vue.js dashboard for model management
- **Mobile App**: React Native/Flutter app for mobile predictions
- **Admin Panel**: Django/Flask admin interface for system management
- **Data Visualization**: Interactive charts and graphs for insights
- **User Management**: User registration, authentication, and profiles

#### Documentation & Support
- **Interactive API Docs**: Swagger UI with live testing capabilities
- **Video Tutorials**: Step-by-step guides for common tasks
- **Community Forum**: User community for support and feature requests
- **Knowledge Base**: Comprehensive documentation and FAQs
- **API Playground**: Interactive testing environment for developers

### 🔬 Research & Development

#### Model Evaluation Enhancements
- **New Data Testing**: Evaluate the best model on new, unseen test datasets
- **A/B Testing Framework**: Compare model performance on different data distributions
- **Cross-Domain Validation**: Test model generalization across different geographic regions
- **Temporal Validation**: Evaluate model performance on data from different time periods
- **Real-time Evaluation**: Continuous model performance monitoring on live data


## Contributing
 
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Troubleshooting

### Common Issues

1. **Model not loaded**: Ensure you've run `make train` first
2. **Port conflicts**: Change ports in the Makefile or use different ports
3. **Memory issues**: Reduce batch size or use smaller datasets

### Logs

Check the console output for detailed logging information. All modules use Python's logging framework with INFO level by default.

## Support

For questions or issues, please:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Open an issue in the repository
