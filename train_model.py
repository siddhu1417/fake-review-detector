"""Command-line entry point for training the Fake Review Detection System."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.model_training import train_and_evaluate


def parse_arguments() -> argparse.Namespace:
    """Parse a dataset path without adding unnecessary training switches."""

    parser = argparse.ArgumentParser(
        description="Train and compare fake-review screening models."
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="CSV with required review_text and label columns; rating is optional.",
    )
    return parser.parse_args()


def main() -> None:
    """Load CSV data, train models, and print a clear evaluation table."""

    arguments = parse_arguments()
    if not arguments.data.exists():
        raise SystemExit(f"Dataset not found: {arguments.data}")

    try:
        dataset = pd.read_csv(arguments.data)
        result = train_and_evaluate(dataset)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise SystemExit(f"Training stopped: {error}") from error

    comparison = pd.DataFrame(result.metrics["model_comparison"]).T
    display_columns = ["accuracy", "precision", "recall", "macro_f1"]
    print("\nModel comparison (holdout set)")
    print(comparison[display_columns].to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved best model: {result.metrics['best_model']}")
    print("Artifacts: models/review_model.joblib, models/tfidf_vectorizer.joblib, models/model_metrics.json")


if __name__ == "__main__":
    main()
