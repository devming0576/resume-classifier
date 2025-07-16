#!/bin/bash
# Clean build, cache, and virtual environment files for the project

set -e

# Remove Python cache and build artifacts
rm -rf __pycache__/ src/**/__pycache__/ tests/__pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .nox/ .tox/ htmlcov/ .coverage coverage.xml

# Remove virtual environments
rm -rf .venv/ .resume-classifier/ env/ venv/ ENV/ env.bak/ venv.bak/

# Remove build and dist folders
rm -rf build/ dist/ wheelhouse/ pip-wheel-metadata/

# Remove outputs, data, models, resumes (if not versioned)
rm -rf outputs/ data/ models/ resumes/

# Remove editor/project files
rm -rf .vscode/ .idea/

# Remove temporary files
find . -name '*.log' -delete
find . -name '*.tmp' -delete
find . -name '*.bak' -delete

# Remove Jupyter/IPython checkpoints
find . -name '.ipynb_checkpoints' -type d -exec rm -rf {} +

# Done
echo "Project cleaned!"
