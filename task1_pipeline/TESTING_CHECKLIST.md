# 🧪 Madrid Housing ML Pipeline - Testing Checklist

## 📋 Overview
This checklist covers comprehensive testing of all scripts, modes, and configurations in the Madrid Housing Market ML Pipeline.

---

## ✅ **1. Environment Setup**

### Prerequisites
- [ ] Navigate to `task1_pipeline` directory
- [ ] Install dependencies: `make install` or `pip install -r requirements.txt`
- [ ] Verify Python environment is working
- [ ] Check all required files exist in project structure

---

## ✅ **2. Data Preparation Script (`scripts/data_prep.py`)**

### Basic Functionality
- [ ] **Test 1**: Basic data preparation
  ```bash
  python scripts/data_prep.py
  ```
  - [ ] Creates `data/preprocessed_houses_Madrid.csv`
  - [ ] No console errors
  - [ ] File size > 0

- [ ] **Test 2**: Using Makefile
  ```bash
  make prepare-data
  ```
  - [ ] Same results as Test 1

- [ ] **Test 3**: Help command
  ```bash
  python scripts/data_prep.py --help
  ```
  - [ ] Shows help message properly

---

## ✅ **3. Training Script (`scripts/train.py`)**

### Single Model Training
- [ ] **Test 1**: Basic single training
  ```bash
  python scripts/train.py single
  ```
  - [ ] Creates model file `models/madrid_housing_model.pkl`
  - [ ] Logs to MLflow
  - [ ] No training errors

- [ ] **Test 2**: Custom run name
  ```bash
  python scripts/train.py single --run-name "test_single_model"
  ```
  - [ ] Uses custom run name in MLflow
  - [ ] Training completes successfully

- [ ] **Test 3**: Custom config
  ```bash
  python scripts/train.py single --config configs/training_config.yaml
  ```
  - [ ] Loads custom config
  - [ ] Training uses config parameters

- [ ] **Test 4**: Makefile command
  ```bash
  make train
  ```
  - [ ] Same results as Test 1

### Multiple Experiments
- [ ] **Test 1**: Basic experiments
  ```bash
  python scripts/train.py experiments
  ```
  - [ ] Runs all configured experiments
  - [ ] Each experiment logged to MLflow
  - [ ] No experiment failures

- [ ] **Test 2**: Makefile command
  ```bash
  make train-experiments
  ```
  - [ ] Same results as Test 1

### Grid Search Training
- [ ] **Test 1**: Enable grid search in config
  - [ ] Edit `configs/training_config.yaml`
  - [ ] Set `grid_search.enabled: true`

- [ ] **Test 2**: Run grid search
  ```bash
  python scripts/train.py grid-search
  ```
  - [ ] Tests multiple parameter combinations
  - [ ] Finds best parameters
  - [ ] Logs all runs to MLflow
  - [ ] Saves best model

- [ ] **Test 3**: Makefile command
  ```bash
  make train-grid
  ```
  - [ ] Same results as Test 2

### Error Handling
- [ ] **Test 1**: Invalid mode
  ```bash
  python scripts/train.py invalid_mode
  ```
  - [ ] Shows error with valid choices

- [ ] **Test 2**: Invalid config
  ```bash
  python scripts/train.py single --config nonexistent.yaml
  ```
  - [ ] Shows config file not found error

- [ ] **Test 3**: Grid search disabled
  ```bash
  python scripts/train.py grid-search
  ```
  - [ ] Shows grid search not enabled error

- [ ] **Test 4**: Help command
  ```bash
  python scripts/train.py --help
  ```
  - [ ] Shows comprehensive help

---

## ✅ **4. Evaluation Script (`scripts/evaluate.py`)**

### Basic Evaluation
- [ ] **Test 1**: Basic evaluation
  ```bash
  python scripts/evaluate.py
  ```
  - [ ] Loads trained model
  - [ ] Calculates metrics (RMSE, MAE, R², MAPE)
  - [ ] Logs to MLflow
  - [ ] Prints metrics to console

- [ ] **Test 2**: Custom model path
  ```bash
  python scripts/evaluate.py --model-path models/madrid_housing_model.pkl
  ```
  - [ ] Loads specified model
  - [ ] Evaluation completes successfully

- [ ] **Test 3**: Custom run name
  ```bash
  python scripts/evaluate.py --run-name "test_evaluation"
  ```
  - [ ] Uses custom run name in MLflow

- [ ] **Test 4**: Makefile command
  ```bash
  make evaluate
  ```
  - [ ] Same results as Test 1

### Error Handling
- [ ] **Test 1**: No model file
  ```bash
  # Delete model first, then:
  python scripts/evaluate.py
  ```
  - [ ] Shows model not found error

- [ ] **Test 2**: Invalid model path
  ```bash
  python scripts/evaluate.py --model-path nonexistent.pkl
  ```
  - [ ] Shows file not found error

- [ ] **Test 3**: Help command
  ```bash
  python scripts/evaluate.py --help
  ```
  - [ ] Shows help message

---

## ✅ **5. API Serving Script (`scripts/serve.py`)**

### Server Start
- [ ] **Test 1**: Start server
  ```bash
  python scripts/serve.py start
  ```
  - [ ] Server starts without errors
  - [ ] Shows "Press Ctrl+C to stop" message
  - [ ] Can stop with Ctrl+C

- [ ] **Test 2**: Custom host/port
  ```bash
  python scripts/serve.py start --host localhost --port 8080
  ```
  - [ ] Server starts on specified host/port

- [ ] **Test 3**: Makefile command
  ```bash
  make serve
  ```
  - [ ] Same results as Test 1

### Endpoint Testing
*Note: Start server in one terminal, then test endpoints in another*

- [ ] **Test 1**: Health check
  ```bash
  python scripts/serve.py health_check
  make test-health
  ```
  - [ ] Returns status JSON
  - [ ] No connection errors

- [ ] **Test 2**: Model info
  ```bash
  python scripts/serve.py model_info
  make test-model-info
  ```
  - [ ] Returns model details JSON
  - [ ] Shows model information

- [ ] **Test 3**: Single prediction
  ```bash
  python scripts/serve.py predict
  make test-predict
  ```
  - [ ] Returns prediction with confidence
  - [ ] Uses test_case_1.json data

- [ ] **Test 4**: Batch prediction
  ```bash
  python scripts/serve.py batch_predict
  make test-batch-predict
  ```
  - [ ] Returns array of predictions
  - [ ] Uses test_case_batch_prediction.json data

### Error Handling
- [ ] **Test 1**: Test without server
  ```bash
  # Don't start server, then:
  python scripts/serve.py health_check
  ```
  - [ ] Shows connection error

- [ ] **Test 2**: Wrong port
  ```bash
  python scripts/serve.py health_check --port 9999
  ```
  - [ ] Shows connection error

- [ ] **Test 3**: Help command
  ```bash
  python scripts/serve.py --help
  ```
  - [ ] Shows comprehensive help

---

## ✅ **6. Makefile Commands**

### All Commands
- [ ] **Test 1**: Help
  ```bash
  make help
  ```
  - [ ] Shows all available commands

- [ ] **Test 2**: Individual commands
  ```bash
  make install
  make prepare-data
  make train
  make evaluate
  make serve
  make test-health
  make test-model-info
  make test-predict
  make test-batch-predict
  make clean
  ```
  - [ ] All commands execute without errors

- [ ] **Test 3**: MLflow UI
  ```bash
  make mlflow-ui
  ```
  - [ ] Opens MLflow UI at http://localhost:5000
  - [ ] Can view experiments and runs

- [ ] **Test 4**: Pipeline commands
  ```bash
  make pipeline
  make pipeline-experiments
  make pipeline-grid
  make train-only
  ```
  - [ ] All pipeline combinations work

---

## ✅ **7. MLflow Integration**

### Tracking Verification
- [ ] **Test 1**: Start MLflow UI
  ```bash
  make mlflow-ui
  ```
  - [ ] UI opens at http://localhost:5000

- [ ] **Test 2**: View training runs
  - [ ] Navigate to http://localhost:5000
  - [ ] See training runs with parameters
  - [ ] See logged metrics

- [ ] **Test 3**: View evaluation runs
  - [ ] See evaluation runs in MLflow
  - [ ] Verify metrics are logged

- [ ] **Test 4**: Model registry
  - [ ] Check registered models
  - [ ] Verify model versions

---

## ✅ **8. Error Recovery & Clean Testing**

### Clean Restart
- [ ] **Test 1**: Clean everything
  ```bash
  make clean
  ```
  - [ ] Removes models, mlruns, cache files

- [ ] **Test 2**: Fresh start
  ```bash
  make install
  make prepare-data
  make train
  make evaluate
  ```
  - [ ] Complete pipeline works after clean

---

## ✅ **9. File Structure Verification**

### Generated Files
- [ ] **Check existence**:
  - [ ] `data/preprocessed_houses_Madrid.csv`
  - [ ] `models/madrid_housing_model.pkl`
  - [ ] `mlruns/` directory with runs
  - [ ] `api_test_cases/test_case_1.json`
  - [ ] `api_test_cases/test_case_batch_prediction.json`

### File Contents
- [ ] **Verify**:
  - [ ] Preprocessed data has expected columns
  - [ ] Model file is not corrupted
  - [ ] Test case files have valid JSON structure

---

## ✅ **10. Integration Testing**

### Complete Workflow
- [ ] **Test 1**: Full pipeline
  ```bash
  make pipeline
  ```
  - [ ] Data prep → Training → Evaluation
  - [ ] All steps complete successfully

- [ ] **Test 2**: API integration
  ```bash
  # Terminal 1:
  make serve
  
  # Terminal 2:
  make test-health
  make test-model-info
  make test-predict
  make test-batch-predict
  ```
  - [ ] Server starts and all endpoints work

- [ ] **Test 3**: MLflow integration
  ```bash
  # Terminal 1:
  make mlflow-ui
  
  # Terminal 2:
  python scripts/train.py single --run-name "integration_test"
  python scripts/evaluate.py --run-name "integration_eval"
  ```
  - [ ] Runs appear in MLflow UI
  - [ ] All data logged correctly

---

## 📊 **Testing Summary**

### Results Tracking
- [ ] **Total Tests**: ___ / ___ passed
- [ ] **Critical Issues**: ___ found
- [ ] **Minor Issues**: ___ found
- [ ] **Overall Status**: ✅ PASS / ❌ FAIL

### Notes Section
```
Add any issues, observations, or notes here:

1. 
2. 
3. 

```

---

## 🚀 **Quick Test Commands**

For rapid testing, use these command sequences:

```bash
# Quick smoke test
make install && make prepare-data && make train && make evaluate

# Quick API test
make serve &
sleep 5
make test-health && make test-predict

# Quick MLflow test
make mlflow-ui &
python scripts/train.py single --run-name "quick_test"
```

---

*Last Updated: $(date)*
*Tested By: ___________*
