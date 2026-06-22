"""Measure overall speaking rate and active articulation rate."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _merged_speaking_seconds(segments: List[Dict[str, Any]]) -> float:
    intervals = []
    for segment in segments:
        start, end = segment.get("start"), segment.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
            intervals.append((float(start), float(end)))
    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def analyze_speaking_rate(
    word_segments: List[Dict[str, Any]],
    audio_duration_seconds: float,
    slow_wpm: int = 120,
    fast_wpm: int = 150,
) -> Dict[str, Any]:
    """Calculate words per total minute and words per active speaking minute."""
    if audio_duration_seconds <= 0:
        raise ValueError("Audio duration must be greater than zero")
    if audio_duration_seconds < 2.0:
        return {
            "wpm": 0.0,
            "articulation_rate": 0.0,
            "pacing_status": "Insufficient speech",
        }

    total_words = sum(
        len(re.findall(r"[\w'-]+", str(segment.get("text", "")), re.UNICODE))
        for segment in word_segments
    )
    if total_words == 0:
        return {
            "wpm": 0.0,
            "articulation_rate": 0.0,
            "pacing_status": "No speech detected",
        }

    speaking_seconds = _merged_speaking_seconds(word_segments)
    wpm = min(300.0, total_words / (audio_duration_seconds / 60.0))
    articulation_rate = (
        min(300.0, total_words / (speaking_seconds / 60.0))
        if speaking_seconds > 0 else 0.0
    )
    if wpm < slow_wpm:
        status = "Slow pacing"
    elif wpm > fast_wpm:
        status = "Fast pacing"
    else:
        status = "Excellent pacing"

    return {
        "wpm": round(wpm, 1),
        "articulation_rate": round(articulation_rate, 1),
        "pacing_status": status,
    }
