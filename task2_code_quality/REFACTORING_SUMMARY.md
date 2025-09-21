# FileManager Refactoring Summary

## Overview
This document summarizes the refactoring of configuration loading and file management functionality into a centralized `FileManager` class.

## Changes Made

### 1. Created New Utils Package
- **Location**: `src/utils/`
- **Files**:
  - `__init__.py` - Package initialization
  - `file_manager.py` - Main FileManager class

### 2. FileManager Class Features
The `FileManager` class provides centralized functionality for:

#### Configuration Loading
- `load_config()` - Generic YAML config loading with fallback
- `load_training_config()` - Training-specific config with defaults
- `load_preprocessing_config()` - Preprocessing-specific config with defaults

#### File Operations
- `file_exists()` - Check if file exists
- `get_absolute_path()` - Resolve relative paths
- `ensure_directory_exists()` - Create directories as needed

#### Data Persistence
- `save_model()` / `load_model()` - Model serialization with joblib
- `save_dataframe()` / `load_dataframe()` - CSV file operations
- `save_json()` / `load_json()` - JSON file operations

### 3. Refactored Classes

#### MadridHousingTrainer (`train_model.py`)
**Before**:
- Had `_load_config()` and `_get_default_config()` methods
- Manual file operations for data loading and model saving
- Direct YAML file handling

**After**:
- Uses `FileManager` instance for all file operations
- Removed duplicate config loading code
- Cleaner, more maintainable code

#### MadridHousingPreprocessor (`preprocessing.py`)
**Before**:
- Had `_load_config()` method
- Manual joblib operations for pipeline saving/loading

**After**:
- Uses `FileManager` for config loading and pipeline operations
- Consistent error handling and logging

#### API Module (`api.py`)
**Before**:
- Direct joblib and JSON file operations
- Manual directory creation

**After**:
- Uses `FileManager` for model loading and JSON saving
- Consistent file path handling

## Benefits

### 1. Code Reusability
- Single source of truth for file operations
- Consistent error handling across all modules
- Easy to extend with new file types

### 2. Maintainability
- Centralized configuration management
- Reduced code duplication
- Easier to update file handling logic

### 3. Error Handling
- Consistent error messages and logging
- Graceful fallbacks for missing files
- Better debugging capabilities

### 4. Path Management
- Automatic resolution of relative paths
- Consistent base path handling
- Cross-platform compatibility

## Usage Examples

### Basic FileManager Usage
```python
from utils.file_manager import FileManager

# Initialize with custom base path
file_manager = FileManager(base_path="/path/to/project")

# Load configuration
config = file_manager.load_training_config("configs/training_config.yaml")

# Check file existence
if file_manager.file_exists("data/processed.csv"):
    df = file_manager.load_dataframe("data/processed.csv")

# Save model
file_manager.save_model(model, "models/my_model.pkl")
```

### Integration in Classes
```python
class MyMLClass:
    def __init__(self, config_path="configs/my_config.yaml"):
        self.file_manager = FileManager()
        self.config = self.file_manager.load_config(config_path)
    
    def save_results(self, data, path):
        self.file_manager.save_json(data, path)
```

## Migration Notes

### Removed Methods
- `MadridHousingTrainer._load_config()`
- `MadridHousingTrainer._get_default_config()`
- `MadridHousingPreprocessor._load_config()`

### Updated Imports
All affected files now import:
```python
from utils.file_manager import FileManager
```

### Configuration Loading
All classes now use:
```python
self.file_manager = FileManager()
self.config = self.file_manager.load_training_config(config_path)
# or
self.config = self.file_manager.load_preprocessing_config(config_path)
```

## Testing
The refactoring has been tested to ensure:
- ✅ Configuration loading works correctly
- ✅ File operations function as expected
- ✅ All existing functionality is preserved
- ✅ Error handling is consistent
- ✅ No breaking changes to existing APIs

## Future Enhancements
The `FileManager` class can be easily extended to support:
- Additional file formats (pickle, parquet, etc.)
- Configuration validation
- File versioning
- Caching mechanisms
- Remote file operations
