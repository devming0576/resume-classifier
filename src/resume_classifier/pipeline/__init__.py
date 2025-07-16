"""
Pipeline module for resume classifier.
"""

from .predictor import ModelPredictor
from .trainer import ModelTrainer

__all__ = ["ModelTrainer", "ModelPredictor"]
