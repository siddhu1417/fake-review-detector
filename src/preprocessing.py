"""Text cleaning and dataset validation utilities.

The cleaner intentionally does not remove stop words.  In particular, words such
as ``not`` and ``never`` often change a review's meaning and are useful signals.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from nltk.tokenize import TweetTokenizer

MAX_REVIEW_LENGTH = 10_000
REQUIRED_COLUMNS = {"review_text", "label"}
TEXT_COLUMN = "review_text"
LABEL_COLUMN = "label"
RATING_COLUMN = "rating"

_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")
# Keep word characters, whitespace, and apostrophes in contractions. Numbers are
# preserved because a repeated order number or rating can still carry context.
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s']", flags=re.UNICODE)
_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)


def _is_missing(value: Any) -> bool:
    """Return whether a scalar value is missing without failing on unusual input."""

    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: Any, max_length: int = MAX_REVIEW_LENGTH) -> str:
    """Return a safe, compact representation of a review.

    Missing values and whitespace-only input become an empty string.  Very long
    input is capped before vectorisation to keep the application responsive.
    """

    if _is_missing(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = text[:max_length]
    text = _URL_PATTERN.sub(" ", text)
    text = _PUNCTUATION_PATTERN.sub(" ", text.lower())
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    # TweetTokenizer is a lightweight NLTK tokenizer and needs no downloaded
    # corpora. It keeps useful tokens such as "not" intact.
    return " ".join(_TOKENIZER.tokenize(text))


def normalize_label(value: Any) -> int:
    """Map common review labels to 0 (genuine) or 1 (suspicious/fake).

    Raises:
        ValueError: If the value cannot be interpreted safely.
    """

    if _is_missing(value):
        raise ValueError("A review label is missing.")

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if float(value) in (0.0, 1.0):
            return int(value)

    normalised = str(value).strip().lower()
    fake_labels = {"1", "1.0", "fake", "fraud", "suspicious", "spam", "false"}
    genuine_labels = {"0", "0.0", "genuine", "real", "authentic", "legit", "true"}
    if normalised in fake_labels:
        return 1
    if normalised in genuine_labels:
        return 0

    raise ValueError(
        f"Unsupported label {value!r}. Use fake/genuine or 1/0 (case-insensitive)."
    )


def validate_training_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and label-normalise a training dataframe.

    The returned copy includes ``clean_review_text`` and ``normalized_label``.
    Rows with an empty review are removed because they cannot train a text model.
    """

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        required = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {required}.")

    prepared = dataframe.copy()
    prepared["clean_review_text"] = prepared[TEXT_COLUMN].map(clean_text)
    prepared = prepared.loc[prepared["clean_review_text"].ne("")].copy()
    if prepared.empty:
        raise ValueError("No usable reviews remain after removing empty review_text values.")

    labels: list[int] = []
    invalid_rows: list[str] = []
    for index, value in prepared[LABEL_COLUMN].items():
        try:
            labels.append(normalize_label(value))
        except ValueError:
            invalid_rows.append(str(index))

    if invalid_rows:
        example_rows = ", ".join(invalid_rows[:5])
        raise ValueError(f"Unsupported label values at row(s): {example_rows}.")

    prepared["normalized_label"] = labels
    if prepared["normalized_label"].nunique() < 2:
        raise ValueError("Training needs at least one genuine and one fake/suspicious review.")
    return prepared
