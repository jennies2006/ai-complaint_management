"""
predict.py — production predict_complaint()
Owner: Member 2 (AI module)

Contract (Doc 1):
    §2  raw text input, max 1000 chars (truncated). Preprocessing internal.
    §3  exact JSON shape. AI never returns department.
    §4  never raises. On error: status="error", category=None, priority=None,
        sentiment_score=0.0, urgency_signals=[], plus a message field.
    §6  urgency vocabulary is the fixed 5-value set (imported from add_features).
    §8  confidence fields are floats in [0, 1].
    §10e sentiment + urgency computed on RAW text, before cleaning.
    §11 models loaded once at module import.

Architecture:
    - Category:  sentence-transformer embedding -> LogisticRegression
    - Priority:  [sentiment_score, 5 urgency flags] -> LogisticRegression
                 (no embedding; see train.py v3 notes)
"""

from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from ai_module.add_features import (compute_sentiment_score,
                                    compute_urgency_signals)
from ai_module.preprocess import clean_text


MODEL_VERSION = "1.0"
MAX_INPUT_CHARS = 1000
MODELS_DIR = Path(__file__).parent / "models"

# ── Loaded ONCE at import (§11) ──────────────────────────────
_cat_model   = joblib.load(MODELS_DIR / "category_model.joblib")
_pri_model   = joblib.load(MODELS_DIR / "priority_model.joblib")
_urg_cols    = joblib.load(MODELS_DIR / "urgency_flags_order.joblib")
_encoder_name = (MODELS_DIR / "encoder_name.txt").read_text().strip()
_encoder     = SentenceTransformer(_encoder_name)


def _empty_result(message: str) -> dict:
    """§4 error shape — never raise, always return this."""
    return {
        "category": None,
        "category_confidence": 0.0,
        "priority": None,
        "priority_confidence": 0.0,
        "sentiment_score": 0.0,
        "urgency_signals": [],
        "model_version": MODEL_VERSION,
        "status": "error",
        "message": message,
    }


def _urgency_vector(urgency_list: list) -> np.ndarray:
    """Build the 5-dim binary vector in the SAME order train.py used."""
    present = set(urgency_list) if isinstance(urgency_list, list) else set()
    return np.array([1.0 if col.replace("urg_", "").replace("_", " ") in present
                     else 0.0 for col in _urg_cols],
                    dtype=float).reshape(1, -1)


def predict_complaint(text: str) -> dict:
    """
    Predict category and priority for a raw complaint string.

    Never raises. On any error, returns the §4 error shape.
    """
    try:
        # ── §2 input validation ──────────────────────────────
        if not isinstance(text, str):
            return _empty_result("input must be a string")
        text = text[:MAX_INPUT_CHARS]
        if not text.strip():
            return _empty_result("empty input")

        # ── §10e sentiment + urgency on RAW text ─────────────
        sentiment = float(compute_sentiment_score(text))
        urgency   = compute_urgency_signals(text)
        if not isinstance(urgency, list):
            urgency = []

        # ── CATEGORY: clean -> embed -> predict ──────────────
        cleaned = clean_text(text)
        if not cleaned.strip():
            return _empty_result("text empty after cleaning")

        emb = _encoder.encode([cleaned], convert_to_numpy=True)

        cat_pred = _cat_model.predict(emb)[0]
        cat_proba = _cat_model.predict_proba(emb)[0]
        cat_conf = float(np.max(cat_proba))

        # ── PRIORITY: signals only -> predict ────────────────
        urg_vec = _urgency_vector(urgency)
        pri_features = np.hstack([np.array([[sentiment]], dtype=float), urg_vec])

        pri_pred = _pri_model.predict(pri_features)[0]
        pri_proba = _pri_model.predict_proba(pri_features)[0]
        pri_conf = float(np.max(pri_proba))

        # ── §3 output shape ──────────────────────────────────
        return {
            "category": str(cat_pred),
            "category_confidence": round(cat_conf, 4),
            "priority": str(pri_pred),
            "priority_confidence": round(pri_conf, 4),
            "sentiment_score": round(sentiment, 4),
            "urgency_signals": urgency,
            "model_version": MODEL_VERSION,
            "status": "ok",
        }

    except Exception as e:
        # §4 — never raise, always return the error shape
        return _empty_result(f"{type(e).__name__}: {e}")
