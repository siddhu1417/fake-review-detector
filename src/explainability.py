"""Small, transparent feature-attribution helpers for Logistic Regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InfluentialFeature:
    """A feature that contributed toward the displayed prediction."""

    feature: str
    contribution: float


def is_logistic_regression(model: Any) -> bool:
    """Return whether a fitted estimator exposes linear logistic coefficients."""

    return model.__class__.__name__ == "LogisticRegression" and hasattr(model, "coef_")


def influential_features(
    model: Any, vectorizer: Any, cleaned_text: str, predicted_label: int, limit: int = 6
) -> list[InfluentialFeature]:
    """Return the strongest active word features for a Logistic Regression decision.

    Positive coefficients point toward suspicious reviews.  For a genuine result,
    the direction is inverted so every returned feature supports the shown result.
    """

    if not is_logistic_regression(model) or not cleaned_text:
        return []

    feature_row = vectorizer.transform([cleaned_text])
    names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    direction = 1 if predicted_label == 1 else -1
    contributions = feature_row.multiply(coefficients * direction).tocoo()

    candidates: list[InfluentialFeature] = []
    for column, value in zip(contributions.col, contributions.data):
        name = str(names[column])
        # Prefer readable word tokens. Character n-grams remain available if no
        # word token contributes strongly enough.
        if value > 0 and name.startswith("word__"):
            candidates.append(InfluentialFeature(name.removeprefix("word__"), float(value)))

    candidates.sort(key=lambda item: item.contribution, reverse=True)
    return candidates[:limit]
