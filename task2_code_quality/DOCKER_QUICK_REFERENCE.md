# Docker Quick Reference

## Essential Commands

### Build & Run
```bash
# Build image
docker build -t madrid-housing-api .

# Run container
docker run -p 8000:8000 madrid-housing-api

# Using docker-compose
docker-compose up
docker-compose up -d  # background
```

### Testing
```bash
# Test all endpoints
python test_docker_predictions.py

# Test health
curl http://127.0.0.1:8000/health

# Test prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @api_test_cases/test_case_1.json
```

### Management
```bash
# Stop containers
docker-compose down

# View logs
docker-compose logs

# Rebuild
docker-compose up --build

# Check status
docker ps
docker images
```

### Troubleshooting
```bash
# Check port usage
netstat -ano | findstr :8000

# Kill process on port 8000
taskkill /PID <process_id> /F

# Clean up
docker system prune
```

## API Endpoints
- Health: http://127.0.0.1:8000/health
- Model Info: http://127.0.0.1:8000/model/info
- Predict: http://127.0.0.1:8000/predict
- Batch Predict: http://127.0.0.1:8000/batch_predict
- Docs: http://127.0.0.1:8000/docs
