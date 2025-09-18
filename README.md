# SMG Engineer Assessment

This repository contains the implementation of two main tasks for the SMG Engineer Assessment:

## 📁 Project Structure

```
smg-engineer-assessment/
│
├── task1_pipeline/          # Task 1: Production-Ready ML Pipeline
│   ├── src/
│   ├── configs/
│   ├── Makefile
│   ├── requirements.txt
│   └── README.md
│
├── task2_code_quality/      # Task 2: Refactoring & Engineering Excellence
│   ├── src/
│   ├── tests/
│   ├── .pre-commit-config.yaml
│   ├── pyproject.toml
│   └── README.md
│
└── .github/workflows/       # CI/CD (shared)
```

## 🚀 Quick Start

### Task 1: Production-Ready ML Pipeline
A complete machine learning pipeline for Madrid Housing Market price prediction with MLflow tracking and FastAPI serving.

```bash
cd task1_pipeline
make install
make pipeline
make serve
```

### Task 2: Code Quality & Engineering Excellence
Refactored code with proper testing, linting, and CI/CD integration.

```bash
cd task2_code_quality
# Implementation details in task2_code_quality/README.md
```

## 📋 Task Overview

### Task 1: Production-Ready ML Pipeline
- ✅ Data loading and validation
- ✅ Scikit-learn preprocessing pipeline
- ✅ LightGBM training with MLflow
- ✅ FastAPI model serving
- ✅ Comprehensive documentation
- ✅ Automation with Makefile

### Task 2: Code Quality & Engineering Excellence
- 📋 Code refactoring and optimization
- 📋 Comprehensive testing suite
- 📋 Linting and formatting (Black, Flake8, MyPy)
- 📋 Pre-commit hooks
- 📋 CI/CD pipeline
- 📋 Documentation improvements

## 🔧 Requirements

- Python 3.8+
- See individual task requirements.txt files for specific dependencies

## 📞 Contact

For questions about this assessment, please refer to the individual task README files.
