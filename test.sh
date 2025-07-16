#!/bin/bash
# Run all Python tests with coverage and show a summary

set -e

# Activate virtual environment if exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d ".resume-classifier" ]; then
  source .resume-classifier/bin/activate
fi

# Run tests with coverage
pytest --cov=src --cov-report=term-missing tests/

# Done
