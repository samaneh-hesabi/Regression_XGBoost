<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Tests Directory</div>

This directory contains test files for the California Housing Price Prediction project.

# 1. Test Structure

## 1.1 Test Categories
- Unit tests
- Integration tests
- End-to-end tests
- Performance tests

# 2. Testing Guidelines

## 2.1 Test Files
- Name test files with `test_` prefix
- Group tests by functionality
- Include both positive and negative test cases

## 2.2 Test Coverage
- Model training tests
- Data preprocessing tests
- Prediction pipeline tests
- Utility function tests

## 2.3 Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test_file.py

# Run with coverage report
python -m pytest --cov=src
``` 