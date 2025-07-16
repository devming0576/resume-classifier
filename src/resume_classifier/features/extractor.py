"""Feature extraction module for resume classification."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from ..utils.config import config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from resume and job description text."""

    def __init__(self, method: str = "tfidf"):
        """Initialize feature extractor.

        Args:
            method: Feature extraction method ('tfidf' or 'transformer')
        """
        self.method = method
        self.vectorizer = None
        self.tokenizer = None
        self._setup_extractor()

    def _setup_extractor(self):
        """Setup the appropriate feature extractor based on method."""
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2,
                max_df=0.95,
            )
        elif self.method == "transformer":
            model_name = config.model.TRANSFORMER_MODEL_NAME
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def extract_tfidf_features(
        self, texts: List[str]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Extract TF-IDF features from text.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF feature matrix
        """
        if self.vectorizer is None:
            self._setup_extractor()

        features = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        return pd.DataFrame(features.toarray(), columns=feature_names)

    def extract_transformer_features(
        self, texts: List[str], max_length: int = 256
    ) -> Dict[str, np.ndarray]:
        """Extract transformer-based features from text.

        Args:
            texts: List of text documents
            max_length: Maximum sequence length

        Returns:
            Dictionary containing input_ids and attention_mask
        """
        if self.tokenizer is None:
            self._setup_extractor()

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
        }

    def extract_features(
        self, texts: List[str], **kwargs
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
        """Extract features from text using the specified method.

        Args:
            texts: List of text documents
            **kwargs: Additional arguments for feature extraction

        Returns:
            Extracted features
        """
        if self.method == "tfidf":
            return self.extract_tfidf_features(texts)
        elif self.method == "transformer":
            return self.extract_transformer_features(texts, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def get_feature_names(self) -> Optional[List[str]]:
        """Get feature names if available.

        Returns:
            List of feature names or None
        """
        if self.method == "tfidf" and self.vectorizer is not None:
            return list(self.vectorizer.get_feature_names_out())
        return None

    def fit_transform(
        self, texts: List[str], **kwargs
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
        """Fit the extractor and transform the data.

        Args:
            texts: List of text documents
            **kwargs: Additional arguments

        Returns:
            Transformed features
        """
        return self.extract_features(texts, **kwargs)

    def transform(
        self, texts: List[str], **kwargs
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray]]:
        """Transform new data using fitted extractor.

        Args:
            texts: List of text documents
            **kwargs: Additional arguments

        Returns:
            Transformed features
        """
        if self.method == "tfidf" and self.vectorizer is not None:
            features = self.vectorizer.transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            return pd.DataFrame(features.toarray(), columns=feature_names)
        elif self.method == "transformer" and self.tokenizer is not None:
            return self.extract_transformer_features(texts, **kwargs)
        else:
            raise ValueError("Extractor not fitted. Call fit_transform first.")

    def get_feature_importance(self, model=None) -> Optional[Dict[str, float]]:
        """Get feature importance if available.

        Args:
            model: Trained model with feature_importances_ attribute

        Returns:
            Dictionary of feature names and their importance scores
        """
        if model is None or not hasattr(model, "feature_importances_"):
            return None

        feature_names = self.get_feature_names()
        if feature_names is None:
            return None

        importance_dict = dict(zip(feature_names, model.feature_importances_))
        return dict(
            sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save(self, filepath: str):
        """Save the feature extractor to disk.

        Args:
            filepath: Path to save the extractor
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "FeatureExtractor":
        """Load a feature extractor from disk.

        Args:
            filepath: Path to the saved extractor

        Returns:
            Loaded feature extractor
        """
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)
