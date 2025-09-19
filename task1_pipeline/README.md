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
python scripts/train.py single      # Direct script execution

# Train multiple experiments
make train-experiments              # Using Makefile
python scripts/train.py experiments # Direct script execution

# Train with grid search tuning
make train-grid                     # Using Makefile
python scripts/train.py grid-search # Direct script execution
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
python scripts/train.py single        # Single model training
python scripts/train.py experiments   # Multiple experiments
python scripts/train.py grid-search   # Grid search tuning

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
python scripts/train.py single --run-name "my_experiment"

# Training with custom config
python scripts/train.py experiments --config custom_config.yaml

# Evaluation with custom model path
python scripts/evaluate.py --model-path "models/my_model.pkl"

# API serving with custom host/port
python scripts/serve.py start --host localhost --port 8080

# Test endpoints with custom host/port
python scripts/serve.py predict --host localhost --port 8080

# Get help for any script
python scripts/data_prep.py --help
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/serve.py --help
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
- **Calculates metrics** (RMSE, MAE, R², MAPE)
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
curl -X GET "http://localhost:8000/health"
```

### Model Information
```bash
curl -X GET "http://localhost:8000/model/info"
```

### Single Prediction (Using Test Case)

Use the provided test case for quick testing:
```bash
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@test_cases/test_case_1.json"
```

Expected response:
```json
{"prediction":169049.65984043336}
```

### Single Prediction (Custom Data)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sq_mt_built": 100,
    "n_rooms": 3,
    "n_bathrooms": 2,
    "house_type_id": "HouseType 1: Piso",
    "neighborhood_id": "Neighborhood 1",
    "has_ac": true,
    "has_terrace": false
  }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "sq_mt_built": 100,
        "n_rooms": 3,
        "n_bathrooms": 2,
        "house_type_id": "HouseType 1: Piso",
        "neighborhood_id": "Neighborhood 1"
      },
      {
        "sq_mt_built": 150,
        "n_rooms": 4,
        "n_bathrooms": 3,
        "house_type_id": "HouseType 2: Casa o chalet",
        "neighborhood_id": "Neighborhood 2"
      }
    ]
  }'
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

The project includes comprehensive testing capabilities:

```bash
# Run all tests
make test

# Test API endpoints
make test-api

# Run specific test cases
python scripts/test_api.py
```

### Test Cases

- **API Test Cases**: Located in `api_test_cases/` directory
- **Test Data**: Sample data for testing in `test_cases/` directory
- **Automated Testing**: Scripts for API endpoint validation
- **Sample Requests**: JSON request examples in `json_requests/` directory

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
- **Metrics Logging**: Performance metrics (RMSE, R², MAE, MAPE) are tracked
- **Model Registry**: Trained models are registered and versioned
- **Artifact Storage**: Feature importance plots and model artifacts are saved

Access MLflow UI:
```bash
# Start MLflow tracking server
mlflow ui --backend-store-uri sqlite:///mlruns.db
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
