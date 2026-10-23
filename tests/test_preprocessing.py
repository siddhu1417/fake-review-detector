"""Unit tests for safe text cleaning and accepted training labels."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import clean_text, normalize_label, validate_training_dataframe


def test_clean_text_removes_urls_and_preserves_negation() -> None:
    """URLs and noisy punctuation should go, while meaning-changing 'not' remains."""

    cleaned = clean_text("This is NOT good!!! See https://example.com/item now.")
    assert "not" in cleaned
    assert "https" not in cleaned
    assert "!" not in cleaned


def test_clean_text_handles_missing_and_long_input() -> None:
    """Missing and excessive text should not crash preprocessing."""

    assert clean_text(None) == ""
    # Use non-repeated characters: TweetTokenizer deliberately shortens highly
    # repeated sequences such as "aaaaa".
    assert clean_text("abcdefghijklmnopqrst", max_length=10) == "abcdefghij"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("fake", 1), ("GENUINE", 0), (1, 1), (0, 0), ("1.0", 1), ("real", 0)],
)
def test_normalize_label_accepts_common_forms(value: object, expected: int) -> None:
    """Training labels may use text or binary values."""

    assert normalize_label(value) == expected


def test_normalize_label_rejects_unknown_value() -> None:
    """Unexpected labels must produce a helpful validation failure."""

    with pytest.raises(ValueError, match="Unsupported label"):
        normalize_label("maybe")


def test_validate_training_dataframe_creates_expected_columns() -> None:
    """A valid dataset receives clean text and binary labels."""

    data = pd.DataFrame(
        {"review_text": ["Great product", "Not useful"], "label": ["genuine", "fake"]}
    )
    prepared = validate_training_dataframe(data)
    assert prepared["normalized_label"].tolist() == [0, 1]
    assert prepared["clean_review_text"].tolist() == ["great product", "not useful"]
