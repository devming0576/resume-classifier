# API Reference

This document provides an overview of the main classes, methods, and usage examples for the Resume Classifier project.

## Contents
- [Core Classes](#core-classes)
- [Key Methods](#key-methods)
- [Usage Examples](#usage-examples)
- [Auto-Generated API Docs](#auto-generated-api-docs)

---

## Core Classes
- `ResumeJobMatcher`
- `ClassicModel`
- `TransformerModel`
- `DataLoader`
- `DataProcessor`

## Key Methods
- `train()`
- `predict()`
- `predict_proba()`
- `score()`
- `save()` / `load()`

## Usage Examples
```python
from src.resume_classifier import ResumeJobMatcher
matcher = ResumeJobMatcher(model_type="classic")
matcher.train(resumes, jobs, labels)
predictions = matcher.predict(resumes, jobs)
```

## Auto-Generated API Docs
For detailed API documentation, consider using [Sphinx](https://www.sphinx-doc.org/) with autodoc to generate HTML docs from your codebase.
