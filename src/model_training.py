"""Training and evaluation functions for the review classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src.preprocessing import validate_training_dataframe
from src.utils import save_artifacts

RANDOM_STATE = 42
DEFAULT_MODEL_NAME = "Logistic Regression"


@dataclass
class TrainingResult:
    """Objects and metadata produced by a successful training run."""

    model: Any
    vectorizer: FeatureUnion
    metrics: dict[str, Any]


def build_vectorizer() -> FeatureUnion:
    """Build a word + character TF-IDF representation.

    Word n-grams capture phrases such as ``not worth``; character n-grams add
    resilience to elongated spelling and small misspellings.
    """

    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=12_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=16_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    return FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])


def _build_models(train_labels: pd.Series) -> dict[str, Any]:
    """Create the three required candidate estimators with safe CV settings."""

    smallest_class = int(train_labels.value_counts().min())
    calibration_folds = min(3, smallest_class)
    if calibration_folds < 2:
        raise ValueError("Each class needs at least two training reviews for calibrated SVM.")

    return {
        "Naive Bayes": MultinomialNB(alpha=0.4),
        DEFAULT_MODEL_NAME: LogisticRegression(
            max_iter=3_000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Calibrated Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            cv=calibration_folds,
        ),
    }


def _evaluate_model(model: Any, features: Any, labels: pd.Series) -> dict[str, Any]:
    """Calculate human-readable metrics for binary fake-review screening."""

    predictions = model.predict(features)
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def train_and_evaluate(dataframe: pd.DataFrame, test_size: float = 0.25) -> TrainingResult:
    """Clean data, compare classifiers, choose the highest macro-F1 model, and save it.

    Logistic Regression wins exact macro-F1 ties, making it the project's stable
    default model while still allowing a clearly better candidate to be selected.
    """

    prepared = validate_training_dataframe(dataframe)
    label_counts = prepared["normalized_label"].value_counts()
    if len(prepared) < 8 or int(label_counts.min()) < 3:
        raise ValueError(
            "Use at least 8 non-empty reviews and at least 3 examples of each label."
        )

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        prepared["clean_review_text"],
        prepared["normalized_label"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=prepared["normalized_label"],
    )
    vectorizer = build_vectorizer()
    train_features = vectorizer.fit_transform(texts_train)
    test_features = vectorizer.transform(texts_test)

    model_metrics: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}
    for model_name, model in _build_models(labels_train).items():
        model.fit(train_features, labels_train)
        trained_models[model_name] = model
        model_metrics[model_name] = _evaluate_model(model, test_features, labels_test)

    # Sort by score, then prefer the stated default to resolve ties predictably.
    best_name = max(
        trained_models,
        key=lambda name: (
            model_metrics[name]["macro_f1"],
            name == DEFAULT_MODEL_NAME,
        ),
    )
    metrics: dict[str, Any] = {
        "best_model": best_name,
        "selection_metric": "macro_f1",
        "training_rows": int(len(prepared)),
        "test_rows": int(len(labels_test)),
        "label_mapping": {"0": "Likely Genuine", "1": "Suspicious Review"},
        "model_comparison": model_metrics,
    }
    save_artifacts(trained_models[best_name], vectorizer, metrics)
    return TrainingResult(trained_models[best_name], vectorizer, metrics)
