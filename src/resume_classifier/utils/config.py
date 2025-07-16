"""
Configuration management for resume classifier.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""

    # Model types
    MODEL_TYPES = ["classic", "transformer"]

    # Classic model configurations
    CLASSIC_MODELS = {
        "KNN": {"n_neighbors": 5},
        "DecisionTree": {"max_depth": 10},
        "RandomForest": {"n_estimators": 100, "max_depth": 10},
        "SVM": {"probability": True},
        "LogisticRegression": {"max_iter": 1000},
        "Bagging": {"n_estimators": 10},
        "AdaBoost": {"n_estimators": 50},
        "GradientBoosting": {"n_estimators": 100},
        "NaiveBayes": {},
    }

    # Transformer model configurations
    TRANSFORMER_MODEL_NAME: str = "bert-base-uncased"
    TRANSFORMER_MAX_LENGTH: int = 256
    TRANSFORMER_NUM_LABELS: int = 2
    TRANSFORMER_BATCH_SIZE: int = 8
    TRANSFORMER_EPOCHS: int = 2

    # Training configurations
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    STRATIFY: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # File extensions
    SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

    # Text processing
    MIN_TEXT_LENGTH: int = 10
    MAX_TEXT_LENGTH: int = 10000

    # Preprocessing
    REMOVE_STOPWORDS: bool = True
    LEMMATIZE: bool = True
    LOWERCASE: bool = True


@dataclass
class PathConfig:
    """Configuration for file paths."""

    # Base directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    SRC_DIR: Path = BASE_DIR / "src"
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    RESUMES_DIR: Path = BASE_DIR / "resumes"
    TESTS_DIR: Path = BASE_DIR / "tests"

    # Model paths
    CLASSIC_MODEL_PATH: Path = MODELS_DIR / "classic_model.pkl"
    TRANSFORMER_MODEL_PATH: Path = MODELS_DIR / "transformer_model"

    # Data paths
    TRAINING_DATA_PATH: Optional[Path] = None
    TEST_DATA_PATH: Optional[Path] = None

    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.OUTPUTS_DIR,
            self.RESUMES_DIR,
            self.TESTS_DIR,
        ]:
            path.mkdir(exist_ok=True)


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Environment variables
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "paths": {k: str(v) for k, v in self.paths.__dict__.items()},
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()

        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        if "paths" in config_dict:
            for key, value in config_dict["paths"].items():
                if hasattr(config.paths, key):
                    setattr(config.paths, key, Path(value))

        if "debug" in config_dict:
            config.DEBUG = config_dict["debug"]

        if "log_level" in config_dict:
            config.LOG_LEVEL = config_dict["log_level"]

        return config


# Global configuration instance
config = Config()
