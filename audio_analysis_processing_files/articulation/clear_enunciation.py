"""Estimate enunciation from normalized word timing and local audio evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Dict, List

import librosa
import numpy as np

_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)


@dataclass(frozen=True)
class EnunciationConfig:
    minimum_word_duration: float = 0.03
    minimum_seconds_per_syllable: float = 0.12
    maximum_seconds_per_syllable: float = 0.35
    rushed_ratio: float = 0.55
    prolonged_ratio: float = 1.80
    minimum_relative_rms: float = 0.45
    timing_weight: float = 0.70
    audibility_weight: float = 0.30
    issue_threshold: float = 0.78


DEFAULT_CONFIG = EnunciationConfig()


def _syllable_count(text: str) -> int:
    """Estimate English/Filipino syllables from vowel groups."""
    cleaned = re.sub(r"[^A-Za-z]", "", text).lower()
    if not cleaned:
        return 1
    count = len(_VOWEL_GROUPS.findall(cleaned))
    # A final silent-e adjustment helps common English words while leaving
    # Filipino words, which are usually phonetic, mostly unchanged.
    if cleaned.endswith("e") and not cleaned.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(1, count)


def _timing_score(ratio: float, config: EnunciationConfig) -> float:
    if ratio <= 0:
        return 0.0
    if ratio < config.rushed_ratio:
        return max(0.0, ratio / config.rushed_ratio)
    if ratio <= config.prolonged_ratio:
        return 1.0
    return max(0.30, 1.0 - (ratio - config.prolonged_ratio) / 2.0)


def _segment_rms(y: np.ndarray, sr: int, start: float, end: float) -> float | None:
    first = max(0, int(round(start * sr)))
    last = min(y.size, int(round(end * sr)))
    if last <= first:
        return None
    segment = y[first:last]
    if segment.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))


def analyze_clear_enunciation(
    word_segments: List[Dict[str, Any]],
    y: np.ndarray | None = None,
    sr: int | None = None,
    config: EnunciationConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Score word timing relative to syllable length and local audibility.

    The score is speaker-normalized. It remains an acoustic proxy, not a
    phoneme-level diagnosis; ``evidence_level`` makes that limitation explicit.
    """
    records = []
    for word in word_segments:
        text = str(word.get("text", "")).strip()
        start, end = word.get("start"), word.get("end")
        if (
            not text
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or isinstance(start, bool)
            or isinstance(end, bool)
        ):
            continue
        start, end = float(start), float(end)
        duration = end - start
        syllables = _syllable_count(text)
        records.append({
            "word": text,
            "start": start,
            "end": end,
            "duration": duration,
            "syllables": syllables,
            "seconds_per_syllable": duration / syllables if duration > 0 else 0.0,
        })

    if not records:
        return {
            "enunciation_score": 0.0,
            "unclear_words": [],
            "message": "No valid word timing available",
            "evidence_level": "unavailable",
            "method": "speaker-normalized syllable timing and relative word audibility",
        }

    positive_rates = [
        record["seconds_per_syllable"]
        for record in records
        if record["duration"] >= config.minimum_word_duration
    ]
    observed_baseline = median(positive_rates) if positive_rates else 0.20
    timing_baseline = min(
        config.maximum_seconds_per_syllable,
        max(config.minimum_seconds_per_syllable, observed_baseline),
    )

    signal = None
    audio_available = y is not None and isinstance(sr, int) and sr > 0
    if audio_available:
        signal = np.asarray(y, dtype=np.float32).reshape(-1)
        audio_available = signal.size > 0 and np.all(np.isfinite(signal))

    rms_values = []
    if audio_available and signal is not None and sr is not None:
        for record in records:
            record["rms"] = _segment_rms(
                signal, sr, record["start"], record["end"]
            )
            if record["rms"] is not None and record["rms"] > 0:
                rms_values.append(record["rms"])
    rms_baseline = median(rms_values) if rms_values else None

    scores = []
    unclear_words = []
    for record in records:
        expected_duration = timing_baseline * record["syllables"]
        ratio = record["duration"] / expected_duration if expected_duration > 0 else 0.0
        timing_score = _timing_score(ratio, config)
        reasons = []
        if record["duration"] <= 0:
            reasons.append("invalid timestamps")
        elif ratio < config.rushed_ratio:
            reasons.append("rushed for estimated syllable count")
        elif ratio > config.prolonged_ratio:
            reasons.append("prolonged for estimated syllable count")

        audibility_score = None
        relative_rms = None
        if rms_baseline and record.get("rms") is not None:
            relative_rms = record["rms"] / rms_baseline
            audibility_score = min(1.0, max(0.0, relative_rms / config.minimum_relative_rms))
            if audibility_score < 1.0:
                reasons.append("word is weak relative to nearby speech")

        if audibility_score is None:
            word_score = timing_score
        else:
            word_score = (
                timing_score * config.timing_weight
                + audibility_score * config.audibility_weight
            )
        scores.append(word_score)

        if word_score < config.issue_threshold:
            unclear_words.append({
                "word": record["word"],
                "duration": round(record["duration"], 3),
                "estimated_syllables": record["syllables"],
                "timing_ratio": round(ratio, 3),
                "relative_rms": round(relative_rms, 3) if relative_rms is not None else None,
                "score": round(word_score * 100, 1),
                "issue": ", ".join(reasons) or "weak timing/acoustic evidence",
            })

    score = round(mean(scores) * 100, 1)
    return {
        "enunciation_score": score,
        "unclear_words": unclear_words,
        "baseline_seconds_per_syllable": round(timing_baseline, 3),
        "message": "Clear enunciation" if score >= 85 else (
            "Mostly clear enunciation" if score >= 70 else "Enunciation needs improvement"
        ),
        "evidence_level": "timing_and_audio" if rms_baseline else "timing_only",
        "method": "speaker-normalized syllable timing and relative word audibility",
    }


def compare_reference_enunciation(
    user_audio: np.ndarray,
    reference_audio: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """Compare enunciation contours locally using normalized MFCC DTW."""
    user = np.asarray(user_audio, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference_audio, dtype=np.float32).reshape(-1)
    if user.size < sr // 2 or reference.size < sr // 2:
        return {
            "available": False,
            "score": None,
            "message": "Audio is too short for reference enunciation comparison",
        }

    user, _ = librosa.effects.trim(user, top_db=35)
    reference, _ = librosa.effects.trim(reference, top_db=35)
    if user.size < sr // 2 or reference.size < sr // 2:
        return {
            "available": False,
            "score": None,
            "message": "Not enough active speech for reference comparison",
        }

    def features(signal: np.ndarray) -> np.ndarray:
        normalized = librosa.util.normalize(signal)
        mfcc = librosa.feature.mfcc(
            y=normalized,
            sr=sr,
            n_mfcc=13,
            n_fft=512,
            hop_length=256,
        )[1:]
        delta = librosa.feature.delta(mfcc, mode="nearest")
        combined = np.vstack((mfcc, delta))
        return (combined - np.mean(combined, axis=1, keepdims=True)) / (
            np.std(combined, axis=1, keepdims=True) + 1e-6
        )

    try:
        reference_features = features(reference)
        user_features = features(user)
        accumulated_cost, path = librosa.sequence.dtw(
            X=reference_features,
            Y=user_features,
            metric="cosine",
        )
        mean_cost = float(accumulated_cost[-1, -1] / max(len(path), 1))
        score = round(float(100.0 * np.exp(-1.5 * max(0.0, mean_cost))), 1)
    except (ValueError, FloatingPointError) as exc:
        return {
            "available": False,
            "score": None,
            "message": f"Reference comparison failed: {exc}",
        }

    return {
        "available": True,
        "score": score,
        "normalized_dtw_cost": round(mean_cost, 4),
        "message": "Local acoustic similarity to reference delivery",
        "method": "cepstral-mean-normalized MFCC and delta-MFCC DTW",
    }
