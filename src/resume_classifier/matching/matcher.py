"""
Resume-job matching utilities for resume classifier.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..data.processor import DataProcessor
from ..features.extractor import FeatureExtractor
from ..models.base import BaseModel
from ..models.classic import ClassicModel
from ..models.transformer import TransformerModel
from ..utils.config import config


class ResumeJobMatcher:
    """Main class for matching resumes to job descriptions."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        data_processor: Optional[DataProcessor] = None,
    ):
        """
        Initialize the resume-job matcher.

        Args:
            model: Trained model for prediction
            feature_extractor: Feature extractor instance
            data_processor: Data processor instance
        """
        self.model = model
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.data_processor = data_processor or DataProcessor()

        # Results storage
        self.matching_results = []
        self.feature_results = []

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

    def match_single(
        self, resume_text: str, job_description: str, include_features: bool = True
    ) -> Dict[str, Any]:
        """
        Match a single resume to a job description.

        Args:
            resume_text: Resume text
            job_description: Job description text
            include_features: Whether to include extracted features

        Returns:
            Dictionary with matching results
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be loaded and trained before matching")

        # Preprocess texts
        processed_resume = self.data_processor.clean_text(resume_text)
        processed_job = self.data_processor.clean_text(job_description)

        # Get model prediction
        match_score = self.model.score(processed_resume, processed_job)

        result = {
            "resume_text": resume_text[:200] + "..."
            if len(resume_text) > 200
            else resume_text,
            "job_description": job_description[:200] + "..."
            if len(job_description) > 200
            else job_description,
            "match_score": match_score,
            "prediction": 1 if match_score > 0.5 else 0,
            "confidence": abs(match_score - 0.5) * 2,  # Convert to 0-1 confidence
        }

        # Add feature-based analysis if requested
        if include_features:
            feature_scores = self.feature_extractor.calculate_similarity_score(
                processed_resume, processed_job
            )
            result.update(feature_scores)

            # Extract detailed features
            resume_features = self.feature_extractor.extract_all_features(
                processed_resume
            )
            job_features = self.feature_extractor.extract_all_features(processed_job)

            result["resume_features"] = {
                "skills": resume_features["skills"][:10],  # Top 10 skills
                "education": resume_features["education"],
                "experience_years": resume_features["experience_years"],
                "job_titles": resume_features["job_titles"],
            }

            result["job_features"] = {
                "skills": job_features["skills"][:10],  # Top 10 skills
                "education": job_features["education"],
                "experience_years": job_features["experience_years"],
                "job_titles": job_features["job_titles"],
            }

        return result

    def match_batch(
        self,
        resumes: List[str],
        job_descriptions: List[str],
        include_features: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Match multiple resumes to multiple job descriptions.

        Args:
            resumes: List of resume texts
            job_descriptions: List of job description texts
            include_features: Whether to include extracted features

        Returns:
            List of matching results
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be loaded and trained before matching")

        results = []

        for resume in resumes:
            for job_desc in job_descriptions:
                result = self.match_single(resume, job_desc, include_features)
                results.append(result)

        # Sort by match score (descending)
        results.sort(key=lambda x: x["match_score"], reverse=True)

        return results

    def match_resume_to_jobs(
        self,
        resume_text: str,
        job_descriptions: List[str],
        top_k: int = 5,
        include_features: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Match a single resume to multiple job descriptions and return top matches.

        Args:
            resume_text: Resume text
            job_descriptions: List of job description texts
            top_k: Number of top matches to return
            include_features: Whether to include extracted features

        Returns:
            List of top matching results
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be loaded and trained before matching")

        results = []

        for job_desc in job_descriptions:
            result = self.match_single(resume_text, job_desc, include_features)
            results.append(result)

        # Sort by match score and return top k
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:top_k]

    def match_job_to_resumes(
        self,
        job_description: str,
        resumes: List[str],
        top_k: int = 5,
        include_features: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Match a single job description to multiple resumes and return top matches.

        Args:
            job_description: Job description text
            resumes: List of resume texts
            top_k: Number of top matches to return
            include_features: Whether to include extracted features

        Returns:
            List of top matching results
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be loaded and trained before matching")

        results = []

        for resume in resumes:
            result = self.match_single(resume, job_description, include_features)
            results.append(result)

        # Sort by match score and return top k
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:top_k]

    def analyze_matching_patterns(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in matching results.

        Args:
            results: List of matching results

        Returns:
            Dictionary with analysis results
        """
        if not results:
            return {}

        # Extract scores
        match_scores = [r["match_score"] for r in results]
        confidence_scores = [r["confidence"] for r in results]

        # Calculate statistics
        analysis = {
            "total_matches": len(results),
            "high_matches": len([s for s in match_scores if s > 0.8]),
            "medium_matches": len([s for s in match_scores if 0.5 <= s <= 0.8]),
            "low_matches": len([s for s in match_scores if s < 0.5]),
            "avg_match_score": sum(match_scores) / len(match_scores),
            "avg_confidence": sum(confidence_scores) / len(confidence_scores),
            "min_match_score": min(match_scores),
            "max_match_score": max(match_scores),
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

    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save matching results to CSV file.

        Args:
            results: List of matching results
            output_path: Path to save the results
        """
        if not results:
            print("No results to save")
            return

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    def get_recommendations(
        self, resume_text: str, job_descriptions: List[str], threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Get job recommendations for a resume based on a threshold.

        Args:
            resume_text: Resume text
            job_descriptions: List of job description texts
            threshold: Minimum match score for recommendations

        Returns:
            List of recommended jobs
        """
        results = self.match_resume_to_jobs(
            resume_text, job_descriptions, top_k=len(job_descriptions)
        )

        # Filter by threshold
        recommendations = [r for r in results if r["match_score"] >= threshold]

        return recommendations

    def get_candidate_rankings(
        self, job_description: str, resumes: List[str], threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Get candidate rankings for a job based on a threshold.

        Args:
            job_description: Job description text
            resumes: List of resume texts
            threshold: Minimum match score for candidates

        Returns:
            List of ranked candidates
        """
        results = self.match_job_to_resumes(
            job_description, resumes, top_k=len(resumes)
        )

        # Filter by threshold
        candidates = [r for r in results if r["match_score"] >= threshold]

        return candidates
