"""Measure pauses and estimate whether their placement looks intentional."""

from __future__ import annotations

import re
from typing import Any, Dict, List

MIN_PAUSE_SECONDS = 0.35
LONG_PAUSE_SECONDS = 1.50
_BOUNDARY_RE = re.compile(r"[.!?;:,][\"')\]]*$")


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def analyze_temporal_pauses(
    word_segments: List[Dict[str, Any]],
    minimum_pause: float = MIN_PAUSE_SECONDS,
) -> Dict[str, Any]:
    """Analyze gaps between words.

    Intent cannot be known from audio alone. A pause after punctuation is
    therefore labelled ``likely_intentional``; other gaps are labelled
    ``possibly_unintentional`` and should be interpreted as a heuristic.
    """
    valid = []
    for word in word_segments:
        start = _number(word.get("start"))
        end = _number(word.get("end"))
        if start is None or end is None or end < start:
            continue
        valid.append({"text": str(word.get("text", "")).strip(), "start": start, "end": end})
    valid.sort(key=lambda item: (item["start"], item["end"]))

    pauses = []
    for previous, current in zip(valid, valid[1:]):
        duration = current["start"] - previous["end"]
        if duration < minimum_pause:
            continue
        follows_boundary = bool(_BOUNDARY_RE.search(previous["text"]))
        classification = (
            "likely_intentional" if follows_boundary else "possibly_unintentional"
        )
        pauses.append({
            "start": round(previous["end"], 3),
            "end": round(current["start"], 3),
            "duration": round(duration, 3),
            "after_word": previous["text"],
            "before_word": current["text"],
            "classification": classification,
            "is_long": duration >= LONG_PAUSE_SECONDS,
        })

    intentional = sum(p["classification"] == "likely_intentional" for p in pauses)
    possibly_unintentional = len(pauses) - intentional
    total = round(sum(p["duration"] for p in pauses), 3)
    penalty = sum(12 + max(0.0, p["duration"] - LONG_PAUSE_SECONDS) * 8 for p in pauses
                  if p["classification"] == "possibly_unintentional")
    pause_score = round(max(0.0, 100.0 - penalty), 1)

    return {
        "pause_count": len(pauses),
        "likely_intentional_count": intentional,
        "possibly_unintentional_count": possibly_unintentional,
        "total_pause_seconds": total,
        "pause_score": pause_score,
        "pauses": pauses,
        "method": "Heuristic: punctuation boundary and word-timestamp gaps",
    }
