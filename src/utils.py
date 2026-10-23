"""Small file-system helpers shared by training and the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "review_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def save_artifacts(model: Any, vectorizer: Any, metrics: dict[str, Any]) -> None:
    """Persist a trained estimator, feature extractor, and serialisable metrics."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_artifacts() -> tuple[Any, Any] | None:
    """Load the saved model bundle, or return ``None`` before first training."""

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def load_metrics() -> dict[str, Any] | None:
    """Load saved evaluation metadata when it is available and valid JSON."""

    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
