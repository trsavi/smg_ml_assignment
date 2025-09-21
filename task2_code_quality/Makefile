# Makefile for Madrid Housing Market ML Pipeline

.PHONY: help install prepare-data train train-experiments train-grid evaluate serve test-health test-model-info test-predict test-batch-predict clean test test-unittest test-coverage test-html-coverage test-file-manager test-preprocessing

# Default target
help: ## Show this help message
	@echo "Madrid Housing Market ML Pipeline"
	@echo "================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install required packages
	pip install -r requirements.txt

# Data preparation
prepare-data: ## Prepare and preprocess the dataset
	@echo "Preparing data..."
	python scripts/data_prep.py

# Training
train: ## Train the model (single training)
	@echo "Training model..."
	python scripts/train.py single

# Training with experiments
train-experiments: ## Train multiple experiments
	@echo "Training experiments..."
	python scripts/train.py experiments

# Training with grid search
train-grid: ## Train with grid search
	@echo "Training with grid search..."
	python scripts/train.py grid-search

# Evaluation
evaluate: ## Evaluate the trained model
	@echo "Evaluating model..."
	python scripts/evaluate.py

# Model serving
serve: ## Start the FastAPI server
	@echo "Starting API server..."
	python scripts/serve.py start

# Test specific endpoints
test-health: ## Test health check endpoint
	@echo "Testing health check..."
	python scripts/serve.py health_check

test-model-info: ## Test model info endpoint
	@echo "Testing model info..."
	python scripts/serve.py model_info

test-predict: ## Test prediction endpoint
	@echo "Testing prediction..."
	python scripts/serve.py predict

test-batch-predict: ## Test batch prediction endpoint
	@echo "Testing batch prediction..."
	python scripts/serve.py batch_predict

# MLflow UI
mlflow-ui: ## Start MLflow UI to view experiments
	@echo "Starting MLflow UI..."
	mlflow ui --backend-store-uri ./mlruns --port 5000

# Testing
test: ## Run all tests
	@echo "Running all tests..."
	python -m pytest tests/ -v

test-unittest: ## Run unittest tests
	@echo "Running unittest tests..."
	python -m pytest tests/unittest/ -v

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	python -m pytest tests/unittest/ --cov=src --cov-report=term-missing --cov-report=html
	@echo "HTML coverage report generated in htmlcov/index.html"

test-html-coverage: ## Run tests with HTML coverage report only
	@echo "Running tests with HTML coverage report..."
	python -m pytest tests/unittest/ --cov=src --cov-report=html
	@echo "HTML coverage report generated in htmlcov/index.html"

test-file-manager: ## Run FileManager tests
	@echo "Running FileManager tests..."
	python -m pytest tests/unittest/test_file_manager_unittest.py -v

test-preprocessing: ## Run Preprocessing tests
	@echo "Running Preprocessing tests..."
	python -m pytest tests/unittest/test_preprocessing_comprehensive_unittest.py -v

test-api: ## Run API tests
	@echo "Running API tests..."
	python -m pytest tests/unittest/test_api_unittest.py -v

test-train: ## Run Training tests
	@echo "Running Training tests..."
	python -m pytest tests/unittest/test_train_model_unittest.py -v

test-data-loader: ## Run Data Loader tests
	@echo "Running Data Loader tests..."
	python -m pytest tests/unittest/test_data_loader_unittest.py -v

# Clean up
clean: ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf models/
	rm -rf mlruns/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Training only (without evaluation)
train-only: prepare-data train ## Train model only (without evaluation)

# Full pipeline
pipeline: prepare-data train evaluate ## Run the complete pipeline (train then evaluate)

# Pipeline with experiments
pipeline-experiments: prepare-data train-experiments evaluate ## Run pipeline with multiple experiments

# Pipeline with grid search
pipeline-grid: prepare-data train-grid evaluate ## Run pipeline with grid search tuning

# Docker commands
docker-build: ## Build Docker image
	docker build -t madrid-housing-api .

docker-run: ## Run Docker container
	docker run -p 8000:8000 madrid-housing-api

docker-compose-up: ## Start services with docker-compose
	docker-compose up --build

docker-compose-down: ## Stop docker-compose services
	docker-compose down

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f


# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	@echo "API documentation available at: http://localhost:8000/docs"
