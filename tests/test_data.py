"""
Unit tests for data module.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from resume_classifier.data.loader import DataLoader
from resume_classifier.data.processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor."""

    def setUp(self):
        """Set up test data."""
        self.processor = DataProcessor()
        self.sample_texts = [
            "This is a sample resume with Python and machine learning skills.",
            "Data scientist with 5+ years of experience in analytics.",
            "Frontend developer specializing in React, JavaScript, and CSS.",
            "",  # Empty text
            "   ",  # Whitespace only
        ]

    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test normal text
        cleaned = self.processor.clean_text(self.sample_texts[0])
        self.assertIsInstance(cleaned, str)
        self.assertGreater(len(cleaned), 0)

        # Test empty text
        cleaned_empty = self.processor.clean_text("")
        self.assertEqual(cleaned_empty, "")

        # Test whitespace normalization
        text_with_extra_spaces = "  hello   world  "
        cleaned_spaces = self.processor.clean_text(text_with_extra_spaces)
        self.assertEqual(cleaned_spaces, "hello world")

    def test_preprocess(self):
        """Test text preprocessing functionality."""
        # Test basic preprocessing
        processed = self.processor.preprocess(self.sample_texts[0])
        self.assertIsInstance(processed, str)

        # Test with custom parameters
        processed_custom = self.processor.preprocess(
            self.sample_texts[0],
            remove_stopwords=False,
            lemmatize=False,
            lowercase=False,
        )
        self.assertIsInstance(processed_custom, str)

    def test_validate_text(self):
        """Test text validation functionality."""
        # Test valid text
        self.assertTrue(self.processor.validate_text(self.sample_texts[0]))

        # Test empty text
        self.assertFalse(self.processor.validate_text(""))

        # Test very short text
        self.assertFalse(self.processor.validate_text("hi"))

        # Test very long text
        long_text = "a" * 15000  # Exceeds max length
        self.assertFalse(self.processor.validate_text(long_text))

    def test_process_batch(self):
        """Test batch processing functionality."""
        processed_texts = self.processor.process_batch(self.sample_texts)
        self.assertEqual(len(processed_texts), len(self.sample_texts))
        self.assertIsInstance(processed_texts, list)

    def test_get_text_statistics(self):
        """Test text statistics functionality."""
        stats = self.processor.get_text_statistics(self.sample_texts[0])
        self.assertIsInstance(stats, dict)
        self.assertIn("length", stats)
        self.assertIn("word_count", stats)
        self.assertIn("sentence_count", stats)
        self.assertIn("avg_word_length", stats)

        # Test empty text
        empty_stats = self.processor.get_text_statistics("")
        self.assertEqual(empty_stats["length"], 0)
        self.assertEqual(empty_stats["word_count"], 0)


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader."""

    def setUp(self):
        """Set up test data."""
        self.loader = DataLoader()
        self.temp_dir = tempfile.mkdtemp()

        # Create sample CSV data
        self.sample_data = {
            "resume_text": [
                "Experienced software engineer with Python skills.",
                "Data scientist with ML experience.",
            ],
            "job_description": [
                "Looking for Python developer.",
                "Senior data scientist position.",
            ],
            "label": [1, 1],
        }

        # Create sample CSV file
        self.csv_path = os.path.join(self.temp_dir, "sample_data.csv")
        df = pd.DataFrame(self.sample_data)
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_csv_data(self):
        """Test CSV data loading functionality."""
        resumes, jobs, labels = self.loader.load_csv_data(self.csv_path)

        self.assertEqual(len(resumes), 2)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels, [1, 1])

    def test_load_csv_data_missing_file(self):
        """Test CSV loading with missing file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_csv_data("nonexistent_file.csv")

    def test_load_csv_data_missing_columns(self):
        """Test CSV loading with missing columns."""
        # Create CSV with missing columns
        bad_data = {"resume_text": ["test"], "label": [1]}  # Missing job_description
        bad_csv_path = os.path.join(self.temp_dir, "bad_data.csv")
        pd.DataFrame(bad_data).to_csv(bad_csv_path, index=False)

        with self.assertRaises(ValueError):
            self.loader.load_csv_data(bad_csv_path)

    def test_load_resumes_from_folder(self):
        """Test loading resumes from folder."""
        # Create test files
        test_files = [
            "resume1.pdf",
            "resume2.docx",
            "resume3.txt",
            "document.txt",
            "image.jpg",  # Should be ignored
        ]

        for filename in test_files:
            file_path = os.path.join(self.temp_dir, filename)
            with open(file_path, "w") as f:
                f.write("Sample content")

        resume_paths = self.loader.load_resumes_from_folder(self.temp_dir)

        # Should find 3 resume files (pdf, docx, txt)
        self.assertEqual(len(resume_paths), 3)
        self.assertTrue(
            all(
                Path(p).suffix.lower() in [".pdf", ".docx", ".txt"]
                for p in resume_paths
            )
        )

    def test_load_job_descriptions_from_folder(self):
        """Test loading job descriptions from folder."""
        # Create test files
        test_files = [
            "job1.pdf",
            "job2.docx",
            "job3.txt",
            "document.txt",
            "image.jpg",  # Should be ignored
        ]

        for filename in test_files:
            file_path = os.path.join(self.temp_dir, filename)
            with open(file_path, "w") as f:
                f.write("Sample content")

        job_paths = self.loader.load_job_descriptions_from_folder(self.temp_dir)

        # Should find 3 job files (pdf, docx, txt)
        self.assertEqual(len(job_paths), 3)
        self.assertTrue(
            all(Path(p).suffix.lower() in [".pdf", ".docx", ".txt"] for p in job_paths)
        )

    def test_create_training_data(self):
        """Test training data creation."""
        # Create resume and job folders
        resume_folder = os.path.join(self.temp_dir, "resumes")
        job_folder = os.path.join(self.temp_dir, "jobs")
        os.makedirs(resume_folder)
        os.makedirs(job_folder)

        # Create sample files
        resume_files = ["resume1.txt", "resume2.txt"]
        job_files = ["job1.txt", "job2.txt"]

        for filename in resume_files:
            file_path = os.path.join(resume_folder, filename)
            with open(file_path, "w") as f:
                f.write("Experienced software engineer with Python skills.")

        for filename in job_files:
            file_path = os.path.join(job_folder, filename)
            with open(file_path, "w") as f:
                f.write("Looking for Python developer.")

        output_path = os.path.join(self.temp_dir, "training_data.csv")

        # Test with synthetic labels
        self.loader.create_training_data(
            resume_folder, job_folder, output_path, create_labels=True
        )

        # Check that output file was created
        self.assertTrue(os.path.exists(output_path))

        # Load and verify the data
        df = pd.read_csv(output_path)
        self.assertIn("resume_text", df.columns)
        self.assertIn("job_description", df.columns)
        self.assertIn("label", df.columns)
        self.assertGreater(len(df), 0)

    def test_get_data_statistics(self):
        """Test data statistics functionality."""
        resumes = [
            "Experienced software engineer with Python and machine learning skills.",
            "Data scientist with 5 years of experience in analytics and statistics.",
        ]

        job_descriptions = [
            "We are looking for a Python developer with ML experience.",
            "Senior data scientist position requiring statistical analysis skills.",
        ]

        stats = self.loader.get_data_statistics(resumes, job_descriptions)

        self.assertIsInstance(stats, dict)
        self.assertIn("num_resumes", stats)
        self.assertIn("num_job_descriptions", stats)
        self.assertIn("resume_stats", stats)
        self.assertIn("job_stats", stats)

        self.assertEqual(stats["num_resumes"], 2)
        self.assertEqual(stats["num_job_descriptions"], 2)


if __name__ == "__main__":
    unittest.main()
