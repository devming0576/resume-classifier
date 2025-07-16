"""
Unit tests for models module.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from resume_classifier.models.base import BaseModel
from resume_classifier.models.classic import ClassicModel, ClassicModelFactory
from resume_classifier.models.transformer import TransformerModel


class TestBaseModel(unittest.TestCase):
    """Test cases for BaseModel abstract class."""

    def test_base_model_initialization(self):
        """Test BaseModel initialization."""
        # This should raise an error since BaseModel is abstract
        with self.assertRaises(TypeError):
            BaseModel()

    def test_base_model_abstract_methods(self):
        """Test that BaseModel has required abstract methods."""
        # Check that required methods exist
        required_methods = [
            "train",
            "predict",
            "predict_proba",
            "_save_model",
            "_load_model",
        ]
        for method in required_methods:
            self.assertTrue(hasattr(BaseModel, method))


class TestClassicModel(unittest.TestCase):
    """Test cases for ClassicModel."""

    def setUp(self):
        """Set up test data."""
        self.sample_resumes = [
            "Experienced software engineer with Python and machine learning skills.",
            "Data scientist with 5 years of experience in analytics and statistics.",
            "Frontend developer specializing in React and JavaScript.",
        ]

        self.sample_jobs = [
            "We are looking for a Python developer with ML experience.",
            "Senior data scientist position requiring statistical analysis skills.",
            "React developer needed for frontend development.",
        ]

        # Create sample training data
        self.X = list(zip(self.sample_resumes, self.sample_jobs))
        self.y = [1, 1, 0]  # Match, Match, No Match

    def test_classic_model_initialization(self):
        """Test ClassicModel initialization."""
        model = ClassicModel(model_type="LogisticRegression")
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, "LogisticRegression")
        self.assertFalse(model.is_trained)

    def test_classic_model_invalid_type(self):
        """Test ClassicModel with invalid model type."""
        with self.assertRaises(ValueError):
            ClassicModel(model_type="InvalidModel")

    def test_classic_model_training(self):
        """Test ClassicModel training."""
        model = ClassicModel(model_type="LogisticRegression")
        model.train(self.X, self.y)
        self.assertTrue(model.is_trained)

    def test_classic_model_prediction(self):
        """Test ClassicModel prediction."""
        model = ClassicModel(model_type="LogisticRegression")
        model.train(self.X, self.y)

        # Test prediction
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(all(isinstance(p, int) for p in predictions))

    def test_classic_model_prediction_proba(self):
        """Test ClassicModel probability prediction."""
        model = ClassicModel(model_type="LogisticRegression")
        model.train(self.X, self.y)

        # Test probability prediction
        probas = model.predict_proba(self.X)
        self.assertEqual(len(probas), len(self.X))
        self.assertTrue(all(len(p) == 2 for p in probas))  # Binary classification
        self.assertTrue(all(sum(p) > 0.99 for p in probas))  # Probabilities sum to ~1

    def test_classic_model_scoring(self):
        """Test ClassicModel scoring."""
        model = ClassicModel(model_type="LogisticRegression")
        model.train(self.X, self.y)

        # Test scoring
        score = model.score(self.sample_resumes[0], self.sample_jobs[0])
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)

    def test_classic_model_save_load(self):
        """Test ClassicModel save and load functionality."""
        model = ClassicModel(model_type="LogisticRegression")
        model.train(self.X, self.y)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            save_path = tmp_file.name

        try:
            model.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            # Load model
            loaded_model = ClassicModel(model_type="LogisticRegression")
            loaded_model.load(save_path)

            self.assertTrue(loaded_model.is_trained)
            self.assertEqual(model.model_type, loaded_model.model_type)

            # Test that predictions are the same
            original_pred = model.predict(self.X)
            loaded_pred = loaded_model.predict(self.X)
            self.assertEqual(original_pred, loaded_pred)

        finally:
            # Clean up
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_classic_model_factory(self):
        """Test ClassicModelFactory."""
        # Test creating all models
        models = ClassicModelFactory.create_all_models()
        self.assertIsInstance(models, dict)
        self.assertGreater(len(models), 0)

        # Test creating specific model
        model = ClassicModelFactory.create_model("LogisticRegression")
        self.assertIsInstance(model, ClassicModel)
        self.assertEqual(model.model_type, "LogisticRegression")

    def test_classic_model_feature_importance(self):
        """Test ClassicModel feature importance."""
        model = ClassicModel(model_type="LogisticRegression")
        model.train(self.X, self.y)

        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)


class TestTransformerModel(unittest.TestCase):
    """Test cases for TransformerModel."""

    def setUp(self):
        """Set up test data."""
        self.sample_resumes = [
            "Experienced software engineer with Python and machine learning skills.",
            "Data scientist with 5 years of experience in analytics and statistics.",
        ]

        self.sample_jobs = [
            "We are looking for a Python developer with ML experience.",
            "Senior data scientist position requiring statistical analysis skills.",
        ]

        # Create sample training data
        self.X = list(zip(self.sample_resumes, self.sample_jobs))
        self.y = [1, 1]  # Both matches

    def test_transformer_model_initialization(self):
        """Test TransformerModel initialization."""
        model = TransformerModel()
        self.assertIsNotNone(model)
        self.assertFalse(model.is_trained)

    def test_transformer_model_training(self):
        """Test TransformerModel training."""
        # Skip if transformers not available
        try:
            model = TransformerModel()
            model.train(self.X, self.y, epochs=1)  # Use 1 epoch for faster testing
            self.assertTrue(model.is_trained)
        except ImportError:
            self.skipTest("Transformers library not available")

    def test_transformer_model_prediction(self):
        """Test TransformerModel prediction."""
        try:
            model = TransformerModel()
            model.train(self.X, self.y, epochs=1)

            # Test prediction
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.X))
            self.assertTrue(all(isinstance(p, int) for p in predictions))
        except ImportError:
            self.skipTest("Transformers library not available")

    def test_transformer_model_prediction_proba(self):
        """Test TransformerModel probability prediction."""
        try:
            model = TransformerModel()
            model.train(self.X, self.y, epochs=1)

            # Test probability prediction
            probas = model.predict_proba(self.X)
            self.assertEqual(len(probas), len(self.X))
            self.assertTrue(all(len(p) == 2 for p in probas))  # Binary classification
        except ImportError:
            self.skipTest("Transformers library not available")

    def test_transformer_model_scoring(self):
        """Test TransformerModel scoring."""
        try:
            model = TransformerModel()
            model.train(self.X, self.y, epochs=1)

            # Test scoring
            score = model.score(self.sample_resumes[0], self.sample_jobs[0])
            self.assertIsInstance(score, float)
            self.assertTrue(0 <= score <= 1)
        except ImportError:
            self.skipTest("Transformers library not available")


class TestResumeJobDataset(unittest.TestCase):
    """Test cases for ResumeJobDataset."""

    def test_dataset_initialization(self):
        """Test ResumeJobDataset initialization."""
        try:
            from transformers import BertTokenizer

            from resume_classifier.models.transformer import ResumeJobDataset

            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            resumes = ["Resume 1", "Resume 2"]
            jobs = ["Job 1", "Job 2"]
            labels = [1, 0]

            dataset = ResumeJobDataset(resumes, jobs, labels, tokenizer)
            self.assertEqual(len(dataset), 2)

            # Test getting an item
            item = dataset[0]
            self.assertIn("input_ids", item)
            self.assertIn("attention_mask", item)
            self.assertIn("labels", item)
        except ImportError:
            self.skipTest("Transformers library not available")


if __name__ == "__main__":
    unittest.main()
