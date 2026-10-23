"""Prediction helpers used by both the single-review and bulk interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.explainability import InfluentialFeature, influential_features, is_logistic_regression
from src.preprocessing import MAX_REVIEW_LENGTH, clean_text


@dataclass(frozen=True)
class PredictionResult:
    """A user-facing review assessment."""

    label: str
    confidence: float
    risk_category: str
    explanation: str
    influential_features: list[InfluentialFeature]
    was_truncated: bool


def _risk_category(probability_fake: float) -> str:
    if probability_fake >= 0.75:
        return "High risk"
    if probability_fake >= 0.50:
        return "Elevated risk"
    if probability_fake >= 0.25:
        return "Moderate confidence"
    return "Low risk"


def predict_review(model: Any, vectorizer: Any, review_text: Any) -> PredictionResult:
    """Assess one review without overstating what a model can establish."""

    raw_text = "" if review_text is None else str(review_text)
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        raise ValueError("Please enter a review with some readable text.")

    feature_row = vectorizer.transform([cleaned_text])
    predicted_class = int(model.predict(feature_row)[0])
    if hasattr(model, "predict_proba"):
        fake_probability = float(model.predict_proba(feature_row)[0][1])
    else:
        # Every bundled candidate is calibrated/probabilistic. This is a safe
        # fallback for an externally supplied compatible estimator.
        fake_probability = float(predicted_class)

    label = "Suspicious Review" if predicted_class == 1 else "Likely Genuine"
    confidence = fake_probability if predicted_class == 1 else 1 - fake_probability
    risk_category = _risk_category(fake_probability)
    if predicted_class == 1:
        explanation = (
            "The wording resembles patterns that the model saw more often in "
            "suspicious examples. Treat this as a prompt for human review, not proof."
        )
    else:
        explanation = (
            "The wording is more similar to genuine examples in the training data. "
            "A human review is still the final decision-maker."
        )

    features = influential_features(model, vectorizer, cleaned_text, predicted_class)
    return PredictionResult(
        label=label,
        confidence=confidence,
        risk_category=risk_category,
        explanation=explanation,
        influential_features=features,
        was_truncated=len(raw_text) > MAX_REVIEW_LENGTH,
    )


def predict_dataframe(model: Any, vectorizer: Any, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add assessment columns to a CSV dataframe while handling blank rows safely."""

    if "review_text" not in dataframe.columns:
        raise ValueError("The uploaded CSV needs a review_text column.")

    processed = dataframe.copy()
    labels: list[str] = []
    confidence_scores: list[float | None] = []
    risk_categories: list[str] = []
    for review in processed["review_text"]:
        try:
            result = predict_review(model, vectorizer, review)
            labels.append(result.label)
            confidence_scores.append(round(result.confidence * 100, 2))
            risk_categories.append(result.risk_category)
        except ValueError:
            labels.append("Needs Review")
            confidence_scores.append(None)
            risk_categories.append("Insufficient text")

    processed["predicted_label"] = labels
    processed["confidence_score"] = confidence_scores
    processed["risk_category"] = risk_categories
    return processed


def supports_explainability(model: Any) -> bool:
    """Expose explainability availability without leaking model internals to the UI."""

    return is_logistic_regression(model)
