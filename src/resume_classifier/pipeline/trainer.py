"""
Model training pipeline for resume classifier.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.loader import DataLoader
from ..data.processor import DataProcessor
from ..models.base import BaseModel
from ..models.classic import ClassicModel, ClassicModelFactory
from ..models.transformer import TransformerModel
from ..utils.config import config


class ModelTrainer:
    """Model trainer for resume classification models."""

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        data_processor: Optional[DataProcessor] = None,
    ):
        """
        Initialize the model trainer.

        Args:
            data_loader: Data loader instance
            data_processor: Data processor instance
        """
        self.data_loader = data_loader or DataLoader()
        self.data_processor = data_processor or DataProcessor()
        self.training_history = []

    def load_training_data(
        self, data_path: str
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Load training data from CSV file.

        Args:
            data_path: Path to the training data CSV

        Returns:
            Tuple of (resumes, job_descriptions, labels)
        """
        return self.data_loader.load_csv_data(data_path)

    def prepare_training_data(
        self, resumes: List[str], job_descriptions: List[str], labels: List[int]
    ) -> Tuple[List[tuple], List[int]]:
        """
        Prepare training data for model training.

        Args:
            resumes: List of resume texts
            job_descriptions: List of job description texts
            labels: List of labels

        Returns:
            Tuple of (resume-job pairs, labels)
        """
        # Clean and preprocess texts
        cleaned_resumes = self.data_processor.process_batch(resumes)
        cleaned_jobs = self.data_processor.process_batch(job_descriptions)

        # Create resume-job pairs
        pairs = list(zip(cleaned_resumes, cleaned_jobs))

        return pairs, labels

    def split_data(
        self,
        X: List[tuple],
        y: List[int],
        test_size: float = None,
        random_state: int = None,
    ) -> Tuple:
        """
        Split data into training and testing sets.

        Args:
            X: Input features (resume-job pairs)
            y: Target labels
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or config.model.TEST_SIZE
        random_state = random_state or config.model.RANDOM_STATE

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if config.model.STRATIFY else None,
        )

    def train_classic_model(
        self,
        model_type: str = "LogisticRegression",
        data_path: str = None,
        X: List[tuple] = None,
        y: List[int] = None,
        **kwargs,
    ) -> ClassicModel:
        """
        Train a classic machine learning model.

        Args:
            model_type: Type of classic model to train
            data_path: Path to training data CSV (if X and y not provided)
            X: Input features (resume-job pairs)
            y: Target labels
            **kwargs: Additional training parameters

        Returns:
            Trained ClassicModel instance
        """
        # Load data if not provided
        if X is None or y is None:
            if data_path is None:
                raise ValueError("Either data_path or X and y must be provided")
            resumes, job_descriptions, y = self.load_training_data(data_path)
            X, y = self.prepare_training_data(resumes, job_descriptions, y)

        # Create model
        model = ClassicModel(model_type=model_type)

        # Train model
        model.train(X, y, **kwargs)

        # Store training info
        self.training_history.append(
            {
                "model_type": "classic",
                "model_name": model_type,
                "training_samples": len(X),
                "parameters": model.get_params(),
            }
        )

        return model

    def train_transformer_model(
        self,
        model_name: str = None,
        data_path: str = None,
        X: List[tuple] = None,
        y: List[int] = None,
        **kwargs,
    ) -> TransformerModel:
        """
        Train a transformer model.

        Args:
            model_name: Name of the pre-trained model
            data_path: Path to training data CSV (if X and y not provided)
            X: Input features (resume-job pairs)
            y: Target labels
            **kwargs: Additional training parameters

        Returns:
            Trained TransformerModel instance
        """
        # Load data if not provided
        if X is None or y is None:
            if data_path is None:
                raise ValueError("Either data_path or X and y must be provided")
            resumes, job_descriptions, y = self.load_training_data(data_path)
            X, y = self.prepare_training_data(resumes, job_descriptions, y)

        # Create model
        model = TransformerModel(model_name=model_name)

        # Train model
        model.train(X, y, **kwargs)

        # Store training info
        self.training_history.append(
            {
                "model_type": "transformer",
                "model_name": model_name or config.model.TRANSFORMER_MODEL_NAME,
                "training_samples": len(X),
                "parameters": model.get_params(),
            }
        )

        return model

    def train_all_classic_models(
        self,
        data_path: str = None,
        X: List[tuple] = None,
        y: List[int] = None,
        **kwargs,
    ) -> Dict[str, ClassicModel]:
        """
        Train all available classic models.

        Args:
            data_path: Path to training data CSV (if X and y not provided)
            X: Input features (resume-job pairs)
            y: Target labels
            **kwargs: Additional training parameters

        Returns:
            Dictionary of trained models
        """
        # Load data if not provided
        if X is None or y is None:
            if data_path is None:
                raise ValueError("Either data_path or X and y must be provided")
            resumes, job_descriptions, y = self.load_training_data(data_path)
            X, y = self.prepare_training_data(resumes, job_descriptions, y)

        # Create all models
        models = ClassicModelFactory.create_all_models()

        # Train each model
        trained_models = {}
        for model_name, model in models.items():
            try:
                print(f"Training {model_name}...")
                model.train(X, y, **kwargs)
                trained_models[model_name] = model

                # Store training info
                self.training_history.append(
                    {
                        "model_type": "classic",
                        "model_name": model_name,
                        "training_samples": len(X),
                        "parameters": model.get_params(),
                    }
                )

            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")

        return trained_models

    def evaluate_model(
        self, model: BaseModel, X_test: List[tuple], y_test: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, roc_auc_score)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC AUC (use probability of positive class)
        y_pred_proba_positive = [proba[1] for proba in y_pred_proba]
        roc_auc = roc_auc_score(y_test, y_pred_proba_positive)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

    def cross_validate_model(
        self, model: BaseModel, X: List[tuple], y: List[int], cv: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to validate
            X: Input features
            y: Target labels
            cv: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import cross_val_score

        # Convert resume-job pairs to concatenated text for classic models
        if isinstance(model, ClassicModel):
            X_processed = [f"{resume.strip()} [SEP] {job.strip()}" for resume, job in X]
        else:
            X_processed = X

        # Perform cross-validation
        cv_scores = cross_val_score(
            model.pipeline if isinstance(model, ClassicModel) else model.model,
            X_processed,
            y,
            cv=cv,
            scoring="accuracy",
        )

        return {
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": cv_scores.mean(),
            "std_cv_score": cv_scores.std(),
        }

    def save_model(self, model: BaseModel, save_path: str) -> None:
        """
        Save a trained model to disk.

        Args:
            model: Trained model to save
            save_path: Path where to save the model
        """
        model.save(save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, model_path: str, model_type: str) -> BaseModel:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model
            model_type: Type of model ("classic" or "transformer")

        Returns:
            Loaded model instance
        """
        if model_type == "classic":
            model = ClassicModel()
        elif model_type == "transformer":
            model = TransformerModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.load(model_path)
        return model

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all training sessions.

        Returns:
            Dictionary with training summary
        """
        if not self.training_history:
            return {"message": "No training history available"}

        summary = {
            "total_training_sessions": len(self.training_history),
            "classic_models_trained": len(
                [h for h in self.training_history if h["model_type"] == "classic"]
            ),
            "transformer_models_trained": len(
                [h for h in self.training_history if h["model_type"] == "transformer"]
            ),
            "total_samples_processed": sum(
                h["training_samples"] for h in self.training_history
            ),
            "recent_training": self.training_history[-1]
            if self.training_history
            else None,
        }

        return summary
