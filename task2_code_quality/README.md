# Madrid Housing Market Price Prediction Pipeline

A production-ready machine learning pipeline for predicting housing prices in Madrid using LightGBM regression and FastAPI model serving.

## Architecture

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
│ • FastAPI       │◀───│ • Metrics        │◀───│ • joblib       │
│ • REST endpoints│    │ • Validation     │    │ • Model saving  │
│ • Pydantic      │    │ • Cross-val      │    │ • Artifacts     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd task1_pipeline

# Install dependencies (choose one method)
make install                    # Using Makefile
pip install -r requirements.txt # Direct pip install
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

**Total: 201 comprehensive tests** covering all functionality with proper setup/teardown and error handling.

### Test Execution

#### Run All Tests

**Linux/macOS (with Make):**
```bash
# Run all unittest tests
make test-unittest

# Run all tests with coverage
make test-coverage

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

# Run all tests with coverage
python -m pytest tests/unittest/ --cov=src --cov-report=term-missing --cov-report=html

# Run specific test modules
python -m pytest tests/unittest/test_file_manager_unittest.py -v
python -m pytest tests/unittest/test_preprocessing_comprehensive_unittest.py -v
python -m pytest tests/unittest/test_train_model_unittest.py -v
python -m pytest tests/unittest/test_api_unittest.py -v
python -m pytest tests/unittest/test_data_loader_unittest.py -v
```

#### Test Coverage
```bash
# Generate coverage report for specific modules
python -m pytest tests/unittest/ --cov=src/utils/file_manager --cov=src/preprocessing --cov-report=term-missing

# Generate HTML coverage report
python -m pytest tests/unittest/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser to view detailed coverage
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
│   ├── test_file_manager_unittest.py           # FileManager comprehensive tests
│   ├── test_preprocessing_comprehensive_unittest.py  # Preprocessing comprehensive tests
│   ├── test_train_model_unittest.py            # Training model tests
│   ├── test_api_unittest.py                    # API endpoint tests
│   └── test_data_loader_unittest.py            # Data loader tests
├── test_api_integration.py                     # Integration tests
├── test_data_loader.py                         # Data loader tests
├── test_preprocessing.py                       # Preprocessing tests
└── test_train_model.py                         # Training tests
```

### Test Coverage

Current test coverage:
- **FileManager**: 91% coverage
- **Preprocessing**: 78% coverage
- **Overall Project**: 87% coverage
- **Total Tests**: 201 unittest tests (all passing ✅)

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

# Run container
docker run -p 8000:8000 madrid-housing-api

# Or use docker-compose
docker-compose up
```

### Environment Variables

- `MODEL_PATH`: Path to saved model (default: `models/madrid_housing_model.pkl`)
- `API_HOST`: API host (default: `0.0.0.0`)
- `API_PORT`: API port (default: `8000`)

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
