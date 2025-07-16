"""Main CLI interface for resume classifier."""

import argparse
import logging
import sys

from .matching.matcher import ResumeJobMatcher
from .pipeline.predictor import Predictor
from .pipeline.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Resume Classifier - Match resumes with job descriptions"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model-type",
        choices=["classic", "transformer"],
        default="classic",
        help="Type of model to train",
    )
    train_parser.add_argument(
        "--classic-model",
        choices=[
            "LogisticRegression",
            "RandomForest",
            "SVM",
            "KNN",
            "DecisionTree",
            "GradientBoosting",
        ],
        default="LogisticRegression",
        help="Classic ML model type",
    )
    train_parser.add_argument(
        "--data-path", required=True, help="Path to training data CSV"
    )
    train_parser.add_argument(
        "--output-path", required=True, help="Path to save trained model"
    )
    train_parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size"
    )
    train_parser.add_argument(
        "--random-state", type=int, default=42, help="Random state"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--model-path", required=True, help="Path to trained model"
    )
    predict_parser.add_argument(
        "--resume-path", help="Path to resume file"
    )
    predict_parser.add_argument(
        "--job-path", help="Path to job description file"
    )
    predict_parser.add_argument(
        "--resumes-dir", help="Directory containing resume files"
    )
    predict_parser.add_argument(
        "--jobs-dir", help="Directory containing job description files"
    )
    predict_parser.add_argument(
        "--output-path", help="Path to save predictions"
    )

    # Match command
    match_parser = subparsers.add_parser("match", help="Match resumes with jobs")
    match_parser.add_argument(
        "--model-path", required=True, help="Path to trained model"
    )
    match_parser.add_argument(
        "--resumes-dir", required=True, help="Directory containing resume files"
    )
    match_parser.add_argument(
        "--jobs-dir", required=True, help="Directory containing job description files"
    )
    match_parser.add_argument(
        "--output-path", required=True, help="Path to save matches"
    )
    match_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Matching threshold"
    )

    return parser


def train_model(args: argparse.Namespace) -> None:
    """Train a model using the provided arguments.

    Args:
        args: Command line arguments
    """
    logger.info("Starting model training...")

    trainer = Trainer(
        model_type=args.model_type,
        classic_model_type=args.classic_model,
    )

    trainer.train(
        data_path=args.data_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    logger.info(f"Model saved to {args.output_path}")


def predict(args: argparse.Namespace) -> None:
    """Make predictions using the provided arguments.

    Args:
        args: Command line arguments
    """
    logger.info("Starting prediction...")

    predictor = Predictor()
    predictor.load_model(args.model_path)

    if args.resume_path and args.job_path:
        # Single prediction
        prediction = predictor.predict_single(
            resume_path=args.resume_path, job_path=args.job_path
        )
        print(f"Prediction: {prediction}")
    elif args.resumes_dir and args.jobs_dir:
        # Batch prediction
        predictor.predict_batch(
            resumes_dir=args.resumes_dir,
            jobs_dir=args.jobs_dir,
            output_path=args.output_path,
        )
        print(f"Predictions saved to {args.output_path}")
    else:
        logger.error("Please provide either single files or directories")
        sys.exit(1)


def match_resumes(args: argparse.Namespace) -> None:
    """Match resumes with jobs using the provided arguments.

    Args:
        args: Command line arguments
    """
    logger.info("Starting resume-job matching...")

    matcher = ResumeJobMatcher()
    matcher.load_model(args.model_path)

    matcher.match_resumes_with_jobs(
        resumes_dir=args.resumes_dir,
        jobs_dir=args.jobs_dir,
        output_path=args.output_path,
        threshold=args.threshold,
    )

    print(f"Matches saved to {args.output_path}")


def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "train":
            train_model(args)
        elif args.command == "predict":
            predict(args)
        elif args.command == "match":
            match_resumes(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
