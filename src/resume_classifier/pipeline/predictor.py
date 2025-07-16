"""
Model prediction pipeline for resume classifier.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..data.processor import DataProcessor
from ..matching.matcher import ResumeJobMatcher
from ..models.base import BaseModel
from ..models.classic import ClassicModel
from ..models.transformer import TransformerModel
from ..utils.config import config


class ModelPredictor:
    """Model predictor for resume classification."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        data_processor: Optional[DataProcessor] = None,
        matcher: Optional[ResumeJobMatcher] = None,
    ):
        """
        Initialize the model predictor.

        Args:
            model: Trained model for prediction
            data_processor: Data processor instance
            matcher: Resume-job matcher instance
        """
        self.model = model
        self.data_processor = data_processor or DataProcessor()
        self.matcher = matcher or ResumeJobMatcher(
            model=model, data_processor=data_processor
        )
        self.prediction_history = []

    def load_model(self, model_path: str, model_type: str = "classic") -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model
            model_type: Type of model ("classic" or "transformer")
        """
        if model_type == "classic":
            self.model = ClassicModel()
        elif model_type == "transformer":
            self.model = TransformerModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.load(model_path)
        self.matcher.model = self.model

    def predict_single(
        self, resume_text: str, job_description: str, include_features: bool = True
    ) -> Dict[str, Any]:
        """
        Predict match score for a single resume-job pair.

        Args:
            resume_text: Resume text
            job_description: Job description text
            include_features: Whether to include extracted features

        Returns:
            Dictionary with prediction results
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be loaded and trained before prediction")

        result = self.matcher.match_single(
            resume_text, job_description, include_features
        )

        # Store prediction history
        self.prediction_history.append(
            {
                "resume_length": len(resume_text),
                "job_length": len(job_description),
                "match_score": result["match_score"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            }
        )

        return result

    def predict_batch(
        self,
        resumes: List[str],
        job_descriptions: List[str],
        include_features: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict match scores for multiple resume-job pairs.

        Args:
            resumes: List of resume texts
            job_descriptions: List of job description texts
            include_features: Whether to include extracted features

        Returns:
            List of prediction results
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be loaded and trained before prediction")

        results = self.matcher.match_batch(resumes, job_descriptions, include_features)

        # Store prediction history
        for result in results:
            self.prediction_history.append(
                {
                    "resume_length": len(result.get("resume_text", "")),
                    "job_length": len(result.get("job_description", "")),
                    "match_score": result["match_score"],
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                }
            )

        return results

    def predict_from_csv(
        self,
        csv_path: str,
        resume_col: str = "resume_text",
        job_col: str = "job_description",
        include_features: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict match scores from a CSV file.

        Args:
            csv_path: Path to the CSV file
            resume_col: Name of the resume text column
            job_col: Name of the job description column
            include_features: Whether to include extracted features

        Returns:
            List of prediction results
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load data
        df = pd.read_csv(csv_path)

        if resume_col not in df.columns:
            raise ValueError(f"Resume column '{resume_col}' not found in CSV")

        if job_col not in df.columns:
            raise ValueError(f"Job column '{job_col}' not found in CSV")

        resumes = df[resume_col].fillna("").tolist()
        job_descriptions = df[job_col].fillna("").tolist()

        return self.predict_batch(resumes, job_descriptions, include_features)

    def get_job_recommendations(
        self,
        resume_text: str,
        job_descriptions: List[str],
        top_k: int = 5,
        threshold: float = 0.7,
        include_features: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get job recommendations for a resume.

        Args:
            resume_text: Resume text
            job_descriptions: List of job description texts
            top_k: Number of top recommendations to return
            threshold: Minimum match score for recommendations
            include_features: Whether to include extracted features

        Returns:
            List of job recommendations
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be loaded and trained before prediction")

        recommendations = self.matcher.get_recommendations(
            resume_text, job_descriptions, threshold
        )

        # Return top k recommendations
        return recommendations[:top_k]

    def get_candidate_rankings(
        self,
        job_description: str,
        resumes: List[str],
        top_k: int = 5,
        threshold: float = 0.7,
        include_features: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get candidate rankings for a job.

        Args:
            job_description: Job description text
            resumes: List of resume texts
            top_k: Number of top candidates to return
            threshold: Minimum match score for candidates
            include_features: Whether to include extracted features

        Returns:
            List of candidate rankings
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be loaded and trained before prediction")

        candidates = self.matcher.get_candidate_rankings(
            job_description, resumes, threshold
        )

        # Return top k candidates
        return candidates[:top_k]

    def analyze_predictions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze prediction results.

        Args:
            results: List of prediction results

        Returns:
            Dictionary with analysis results
        """
        if not results:
            return {"message": "No results to analyze"}

        # Extract scores
        match_scores = [r["match_score"] for r in results]
        confidence_scores = [r["confidence"] for r in results]
        predictions = [r["prediction"] for r in results]

        # Calculate statistics
        analysis = {
            "total_predictions": len(results),
            "positive_predictions": sum(predictions),
            "negative_predictions": len(predictions) - sum(predictions),
            "avg_match_score": sum(match_scores) / len(match_scores),
            "avg_confidence": sum(confidence_scores) / len(confidence_scores),
            "min_match_score": min(match_scores),
            "max_match_score": max(match_scores),
            "high_confidence_predictions": len(
                [c for c in confidence_scores if c > 0.8]
            ),
            "low_confidence_predictions": len(
                [c for c in confidence_scores if c < 0.5]
            ),
        }

        # Analyze feature patterns if available
        if "skill_similarity" in results[0]:
            skill_similarities = [r["skill_similarity"] for r in results]
            keyword_similarities = [r["keyword_similarity"] for r in results]

            analysis.update(
                {
                    "avg_skill_similarity": sum(skill_similarities)
                    / len(skill_similarities),
                    "avg_keyword_similarity": sum(keyword_similarities)
                    / len(keyword_similarities),
                }
            )

        return analysis

    def save_predictions(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save prediction results to CSV file.

        Args:
            results: List of prediction results
            output_path: Path to save the results
        """
        if not results:
            print("No results to save")
            return

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

    def get_prediction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all predictions made.

        Returns:
            Dictionary with prediction summary
        """
        if not self.prediction_history:
            return {"message": "No prediction history available"}

        # Calculate summary statistics
        match_scores = [h["match_score"] for h in self.prediction_history]
        confidence_scores = [h["confidence"] for h in self.prediction_history]
        predictions = [h["prediction"] for h in self.prediction_history]

        summary = {
            "total_predictions": len(self.prediction_history),
            "positive_predictions": sum(predictions),
            "negative_predictions": len(predictions) - sum(predictions),
            "avg_match_score": sum(match_scores) / len(match_scores),
            "avg_confidence": sum(confidence_scores) / len(confidence_scores),
            "min_match_score": min(match_scores),
            "max_match_score": max(match_scores),
            "high_confidence_predictions": len(
                [c for c in confidence_scores if c > 0.8]
            ),
            "low_confidence_predictions": len(
                [c for c in confidence_scores if c < 0.5]
            ),
        }

        return summary

    def batch_predict_from_files(
        self,
        resume_folder: str,
        job_folder: str,
        output_path: str,
        include_features: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict match scores from files in folders.

        Args:
            resume_folder: Path to folder containing resume files
            job_folder: Path to folder containing job description files
            output_path: Path to save the results
            include_features: Whether to include extracted features

        Returns:
            List of prediction results
        """
        from ..data.loader import DataLoader

        data_loader = DataLoader()

        # Load files
        resume_paths = data_loader.load_resumes_from_folder(resume_folder)
        job_paths = data_loader.load_job_descriptions_from_folder(job_folder)

        if not resume_paths:
            raise ValueError(f"No resume files found in {resume_folder}")

        if not job_paths:
            raise ValueError(f"No job description files found in {job_folder}")

        # Extract text from files
        resumes = []
        job_descriptions = []

        for resume_path in resume_paths:
            try:
                resume_text = data_loader._extract_text_from_file(resume_path)
                if self.data_processor.validate_text(resume_text):
                    resumes.append(resume_text)
            except Exception as e:
                print(f"Warning: Could not load resume {resume_path}: {str(e)}")

        for job_path in job_paths:
            try:
                job_text = data_loader._extract_text_from_file(job_path)
                if self.data_processor.validate_text(job_text):
                    job_descriptions.append(job_text)
            except Exception as e:
                print(f"Warning: Could not load job description {job_path}: {str(e)}")

        # Make predictions
        results = self.predict_batch(resumes, job_descriptions, include_features)

        # Save results
        self.save_predictions(results, output_path)

        return results
