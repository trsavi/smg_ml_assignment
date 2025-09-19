# Madrid Housing Market Price Prediction Pipeline

A production-ready machine learning pipeline for predicting housing prices in Madrid using LightGBM regression and FastAPI model serving.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Preprocessing   │    │   Training      │
│                 │    │                  │    │                 │
│ • data_loader   │───▶│ • preprocessing  │───▶│ • train.py      │
│ • CSV loading   │    │ • Pipeline       │    │ • Hyperparams   │
│ • Validation    │    │ • Scaling        │    │ • Model training│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Model API     │    │   Evaluation     │    │   Model Store   │
│                 │    │                  │    │                 │
│ • FastAPI       │◀───│ • Metrics        │◀───│ • joblib        │
│ • REST endpoints│    │ • Validation     │    │ • Model saving  │
│ • Pydantic      │    │ • Cross-val      │    │ • Artifacts     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd task1_pipeline

# Install dependencies
make install
# or
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Prepare and preprocess the dataset
make prepare-data
```

### 3. Training

```bash
# Train the model
make train
```

### 4. Model Serving

```bash
# Start the FastAPI server
make serve
```

### 5. View Results

- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
task1_pipeline/
├── src/
│   ├── data_loader.py      # Data loading and validation
│   ├── preprocessing.py    # Preprocessing pipeline
│   ├── train.py           # Model training
│   └── api.py             # FastAPI model serving
├── configs/
│   ├── preprocessing_config.yaml
│   └── training_config.yaml
├── models/                # Saved models (created after training)
├── Makefile              # Automation commands
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Available Commands

| Command | Description |
|---------|-------------|
| `make install` | Install required packages |
| `make prepare-data` | Prepare and preprocess dataset |
| `make train` | Train model |
| `make evaluate` | Evaluate trained model |
| `make serve` | Start FastAPI server |
| `make serve-dev` | Start development server with auto-reload |
| `make test` | Run tests |
| `make clean` | Clean up generated files |
| `make pipeline` | Run complete pipeline (prepare → train → evaluate) |

## 📊 Data Pipeline

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

### Training (`train.py`)
- **LightGBM regression** with hyperparameter tuning
- **Cross-validation** for robust evaluation
- **Model persistence** with joblib

## 🚀 API Endpoints

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Model Information
```bash
curl -X GET "http://localhost:8000/model/info"
```

### Single Prediction
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


## ⚙️ Configuration

### Preprocessing Configuration (`configs/preprocessing_config.yaml`)
- Columns to drop
- Boolean columns handling
- Categorical columns for encoding
- Critical columns for filtering

### Training Configuration (`configs/training_config.yaml`)
- Data paths and parameters
- Model hyperparameters
- Training parameters

## 🧪 Testing

```bash
# Run all tests
make test

# Test API endpoints
make test-api
```

## 📊 Model Performance

The model typically achieves:
- **RMSE**: ~50,000-80,000 euros
- **R²**: ~0.85-0.95
- **MAE**: ~35,000-60,000 euros

## 🔍 Feature Engineering

Key features used for prediction:
- **sq_mt_built**: Built surface area (square meters)
- **n_rooms**: Number of rooms
- **n_bathrooms**: Number of bathrooms
- **house_type_id**: Type of house (categorical)
- **neighborhood_id**: Neighborhood identifier (categorical)
- **Boolean features**: AC, pool, terrace, balcony, etc.

## 🚀 Production Deployment

### Docker Deployment (Optional)

```bash
# Build Docker image
make docker-build

# Run container
make docker-run
```

### Environment Variables

- `MODEL_PATH`: Path to saved model (default: `models/madrid_housing_model.pkl`)
- `API_HOST`: API host (default: `0.0.0.0`)
- `API_PORT`: API port (default: `8000`)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Model not loaded**: Ensure you've run `make train` first
2. **Port conflicts**: Change ports in the Makefile or use different ports
3. **Memory issues**: Reduce batch size or use smaller datasets

### Logs

Check the console output for detailed logging information. All modules use Python's logging framework with INFO level by default.

## 📞 Support

For questions or issues, please:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Open an issue in the repository
