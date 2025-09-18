# Makefile for Madrid Housing Market ML Pipeline

.PHONY: help install prepare-data train evaluate serve clean test

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
	python -c "from src.preprocessing import MadridHousingPreprocessor; from src.data_loader import load_data; df = load_data('src/houses_Madrid.csv'); preprocessor = MadridHousingPreprocessor(); X, y = preprocessor.fit_transform(df.drop(columns=['buy_price']), df['buy_price']); print(f'Data prepared: {X.shape}')"

# Training
train: ## Train the model with MLflow tracking
	@echo "Training model..."
	cd src && python train.py

# Evaluation
evaluate: ## Evaluate the trained model
	@echo "Evaluating model..."
	cd src && python -c "from train import MadridHousingTrainer; trainer = MadridHousingTrainer(); results = trainer.run_training_pipeline(); print(f'Evaluation completed. RMSE: {results[\"metrics\"][\"rmse\"]:.2f}')"

# Model serving
serve: ## Start the FastAPI server
	@echo "Starting API server..."
	cd src && python api.py

# Development server
serve-dev: ## Start the FastAPI server in development mode
	@echo "Starting development server..."
	cd src && uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# MLflow UI
mlflow-ui: ## Start MLflow UI to view experiments
	@echo "Starting MLflow UI..."
	mlflow ui --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5000

# Testing
test: ## Run tests
	@echo "Running tests..."
	python -m pytest tests/ -v

# Clean up
clean: ## Clean up generated files
	@echo "Cleaning up..."
	rm -rf models/
	rm -rf mlruns/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Full pipeline
pipeline: prepare-data train evaluate ## Run the complete pipeline

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

# API testing
test-api: ## Test API endpoints
	@echo "Testing API endpoints..."
	@echo "Health check:"
	curl -X GET "http://localhost:8000/health"
	@echo ""
	@echo "Model info:"
	curl -X GET "http://localhost:8000/model/info"
	@echo ""
	@echo "Single prediction:"
	curl -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"sq_mt_built": 100, "n_rooms": 3, "n_bathrooms": 2, "house_type_id": "HouseType 1: Piso", "neighborhood_id": "Neighborhood 1"}'

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	@echo "API documentation available at: http://localhost:8000/docs"
	@echo "MLflow UI available at: http://localhost:5000"
