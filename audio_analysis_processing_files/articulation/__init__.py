"""Articulation measurements: pronunciation accuracy and clear enunciation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .accurate_pronunciation import analyze_pronunciation_accuracy
from .clear_enunciation import analyze_clear_enunciation, compare_reference_enunciation


def analyze_articulation(
    word_segments: List[Dict[str, Any]],
    y: np.ndarray | None = None,
    sr: int | None = None,
) -> Dict[str, Any]:
    """Combine available pronunciation evidence with enunciation timing."""
    accuracy = analyze_pronunciation_accuracy(word_segments)
    enunciation = analyze_clear_enunciation(word_segments, y=y, sr=sr)
    accuracy_score = accuracy["pronunciation_accuracy_score"]
    enunciation_score = float(enunciation["enunciation_score"])

    if accuracy_score is None:
        score = enunciation_score
        message = f"{enunciation['message']}; pronunciation confidence unavailable"
    else:
        score = round(float(accuracy_score) * 0.60 + enunciation_score * 0.40, 1)
        message = "Strong articulation" if score >= 85 else (
            "Acceptable articulation" if score >= 70 else "Articulation needs improvement"
        )

    return {
        "articulation_score": score,
        "message": message,
        "accurate_pronunciation": accuracy,
        "clear_enunciation": enunciation,
    }


__all__ = [
    "analyze_articulation",
    "analyze_clear_enunciation",
    "analyze_pronunciation_accuracy",
    "compare_reference_enunciation",
]
