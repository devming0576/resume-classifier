# CI/CD Workflow Guide

## Overview

Continuous Integration and Continuous Deployment (CI/CD) help automate testing, quality checks, and deployment for your project. This guide provides a sample GitHub Actions workflow and best practices for setting up CI/CD for the Resume Classifier project.

## Sample GitHub Actions Workflow

Create a file at `.github/workflows/ci.yml` in your repository:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest flake8
          python -m spacy download en_core_web_sm

      - name: Lint with flake8
        run: |
          flake8 src/ tests/

      - name: Run tests
        run: |
          pytest tests/ --maxfail=1 --disable-warnings -v

      - name: Upload coverage report
        if: success()
        uses: actions/upload-artifact@v3
        with:
          name: coverage-report
          path: .pytest_cache

  # Optional: Add deployment job here
```

## Explanation
- **Checkout code**: Gets your code from GitHub.
- **Set up Python**: Installs the specified Python version.
- **Install dependencies**: Installs all required packages and the spaCy model.
- **Lint with flake8**: Checks code style and quality.
- **Run tests**: Runs all unit and integration tests.
- **Upload coverage report**: (Optional) Uploads test coverage artifacts.
- **Deployment**: You can add deployment steps (e.g., to PyPI, Docker, or a server) as needed.

## Best Practices
- Run CI on every push and pull request.
- Require passing CI for merging PRs.
- Add code coverage and static analysis.
- Use secrets for deployment credentials.
- Keep workflows fast and focused.

## References
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://docs.github.com/en/actions/guides/publishing-python-packages-to-pypi)
