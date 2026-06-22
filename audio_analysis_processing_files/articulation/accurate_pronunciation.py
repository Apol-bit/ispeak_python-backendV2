"""Estimate pronunciation accuracy from genuine ASR confidence values."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

MINIMUM_CONFIDENCE = 0.60


def _confidence(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return max(0.0, min(float(value), 1.0))


def analyze_pronunciation_accuracy(
    word_segments: List[Dict[str, Any]],
    minimum_confidence: float = MINIMUM_CONFIDENCE,
) -> Dict[str, Any]:
    """Score pronunciation only when the recognizer provides real confidence.

    This is an ASR-confidence proxy, not a phoneme-level clinical assessment.
    Returning ``None`` is more honest than inventing a perfect score when the
    model does not expose confidence.
    """
    analyzed = []
    low_confidence_words = []
    for word in word_segments:
        text = str(word.get("text", "")).strip()
        confidence = _confidence(word.get("confidence"))
        if not text or confidence is None:
            continue
        analyzed.append(confidence)
        if confidence < minimum_confidence:
            low_confidence_words.append({
                "word": text,
                "confidence": round(confidence, 3),
            })

    if not analyzed:
        return {
            "pronunciation_accuracy_score": None,
            "available": False,
            "low_confidence_words": [],
            "message": "Pronunciation confidence unavailable from the speech model",
        }

    score = round(mean(analyzed) * 100, 1)
    return {
        "pronunciation_accuracy_score": score,
        "available": True,
        "low_confidence_words": low_confidence_words,
        "message": "Accurate pronunciation" if score >= 85 else (
            "Some pronunciation uncertainty" if score >= 70 else "Pronunciation needs improvement"
        ),
    }
