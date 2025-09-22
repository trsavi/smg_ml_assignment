# API Refactoring Guide

## Overview

The Madrid Housing Market API has been refactored to follow proper separation of concerns and improve maintainability. The refactoring separates configuration, model logic, and request/response handling into dedicated modules.

## New Structure

### Configuration
- **`configs/api_config.json`**: Centralized API configuration including host, port, input/output schemas, and logging settings.

### API Modules (`src/api/`)
- **`config_loader.py`**: Handles loading and managing API configuration from JSON files.
- **`model_manager.py`**: Manages model loading, metadata extraction, and model state.
- **`prediction_service.py`**: Handles prediction logic, input validation, and data processing.
- **`json_handler.py`**: Manages request/response JSON operations and file saving.
- **`models.py`**: Defines Pydantic models for request/response validation.
- **`__init__.py`**: Package initialization and exports.

### Refactored Files
- **`src/api.py`**: Clean API script with only endpoint definitions.
- **`scripts/serve.py`**: Updated serve script supporting both legacy and refactored APIs.

## Key Improvements

### 1. Separation of Concerns
- **API Routes**: Only endpoint definitions and request/response handling
- **Model Logic**: Separated into `ModelManager` class
- **Prediction Logic**: Separated into `PredictionService` class
- **Configuration**: Centralized in JSON config file
- **JSON Handling**: Separated into `JSONHandler` class

### 2. Configuration Management
- All hardcoded values moved to `configs/api_config.json`
- Dynamic configuration loading using existing `FileManager`
- Easy environment-specific configuration changes

### 3. Better Error Handling
- Centralized error handling in each module
- Proper exception propagation
- Detailed logging throughout the application

### 4. Improved Maintainability
- Each module has a single responsibility
- Easy to test individual components
- Clear interfaces between modules
- Reusable components

## Usage

### Starting the Refactored API
```powershell
# Start refactored API server
python .\scripts\serve.py start

# Start legacy API server (for comparison)
python .\scripts\serve.py start --legacy

# Start with custom host/port
python .\scripts\serve.py start --host 0.0.0.0 --port 8080
```

### Testing Endpoints
```powershell
# Test health check
python .\scripts\serve.py health_check

# Test model info
python .\scripts\serve.py model_info

# Test single prediction
python .\scripts\serve.py predict

# Test batch prediction
python .\scripts\serve.py batch_predict
```

### Direct API Usage
```powershell
# Run refactored API directly
cd src
python .\api.py

# Run legacy API directly (if api_legacy.py exists)
cd src
python .\api_legacy.py
```

## Configuration

### API Configuration (`configs/api_config.json`)
```json
{
  "api": {
    "host": "127.0.0.1",
    "port": 8000,
    "title": "Madrid Housing Price Prediction API",
    "version": "1.0.0"
  },
  "model": {
    "path": "../models/madrid_housing_model.pkl",
    "name": "Madrid Housing Price Prediction",
    "algorithm": "LightGBM"
  },
  "input_schema": {
    "type": "object",
    "properties": {
      "sq_mt_built": {"type": "number"},
      "n_rooms": {"type": "number"},
      // ... other fields
    }
  },
  "output_schema": {
    "single_prediction": {
      "type": "object",
      "properties": {
        "prediction": {"type": "number"}
      }
    }
  }
}
```

## Module Details

### ModelManager
- **Purpose**: Manages model loading and state
- **Key Methods**:
  - `load_model(path)`: Load model from file
  - `get_model()`: Get loaded model
  - `is_model_loaded()`: Check if model is loaded
  - `predict(data)`: Make predictions

### PredictionService
- **Purpose**: Handles prediction logic and validation
- **Key Methods**:
  - `validate_input(data)`: Validate input data
  - `make_single_prediction(data)`: Make single prediction
  - `make_batch_prediction(data_list)`: Make batch predictions

### JSONHandler
- **Purpose**: Manages JSON operations and file saving
- **Key Methods**:
  - `save_request(data, type)`: Save request for debugging
  - `format_single_response(prediction)`: Format single response
  - `format_batch_response(predictions, count)`: Format batch response

### APIConfigLoader
- **Purpose**: Loads and manages API configuration
- **Key Methods**:
  - `load_config(path)`: Load configuration from file
  - `get_api_config()`: Get API settings
  - `get_model_config()`: Get model settings

## Benefits

1. **Maintainability**: Each module has a clear, single responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Reusability**: Modules can be reused in other parts of the application
4. **Configuration**: Easy to modify settings without code changes
5. **Error Handling**: Centralized and consistent error handling
6. **Logging**: Comprehensive logging throughout the application
7. **Type Safety**: Full type hints and Pydantic models for validation

## Migration from Legacy API

The refactored API maintains full backward compatibility with the legacy API. All endpoints work exactly the same way, but the internal structure is much cleaner and more maintainable.

### Key Differences
- **Configuration**: Now loaded from JSON file instead of hardcoded
- **Error Handling**: More detailed and consistent
- **Logging**: Enhanced logging throughout the application
- **Code Organization**: Clear separation of concerns
- **Type Safety**: Full type hints and validation

## Future Enhancements

The refactored structure makes it easy to add new features:

1. **Authentication**: Add to `APIConfigLoader` and implement in endpoints
2. **Rate Limiting**: Add to configuration and implement in middleware
3. **Caching**: Add to `ModelManager` or create new `CacheManager`
4. **Monitoring**: Add to `JSONHandler` or create new `MonitoringService`
5. **Multiple Models**: Extend `ModelManager` to handle multiple models
6. **API Versioning**: Add version handling to `APIConfigLoader`

## Testing

The refactored API can be tested using the same test cases as the legacy API. The `serve.py` script provides comprehensive testing capabilities for all endpoints.

## Conclusion

The refactored API provides a much cleaner, more maintainable, and more extensible architecture while maintaining full backward compatibility. The separation of concerns makes it easier to understand, test, and modify individual components without affecting the entire system.
