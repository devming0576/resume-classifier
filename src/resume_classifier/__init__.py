"""
Resume Classifier Package

A machine learning package for matching resumes to job descriptions.
"""

__version__ = "0.1.0"
__author__ = "Resume Classifier Team"

from .matching.matcher import ResumeJobMatcher
from .models.base import BaseModel
from .models.classic import ClassicModel
from .models.transformer import TransformerModel
from .pipeline.predictor import ModelPredictor
from .pipeline.trainer import ModelTrainer

__all__ = [
    "BaseModel",
    "ClassicModel",
    "TransformerModel",
    "ResumeJobMatcher",
    "ModelTrainer",
    "ModelPredictor",
]
