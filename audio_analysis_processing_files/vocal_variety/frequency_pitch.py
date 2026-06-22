"""Robust, voice-range-independent pitch-variety analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import librosa
import numpy as np


@dataclass(frozen=True)
class PitchConfig:
    frame_length: int = 2048
    hop_length: int = 512
    minimum_voiced_frames: int = 8
    minimum_voiced_ratio: float = 0.08
    minimum_voiced_probability: float = 0.30
    contour_interval_seconds: float = 0.25
    maximum_tracker_jump_semitones: float = 7.0
    monotone_range_semitones: float = 2.5
    monotone_robust_std_semitones: float = 0.85
    monotone_contour_movement_semitones: float = 0.70


DEFAULT_CONFIG = PitchConfig()


def _unavailable_result(message: str) -> Dict[str, Any]:
    return {
        "analysis_available": False,
        "median_pitch_hz": None,
        "pitch_variation_hz": None,
        "pitch_variation_semitones": None,
        "pitch_range_semitones": None,
        "contour_movement_semitones": None,
        "pitch_variety_score": None,
        "voiced_frame_ratio": 0.0,
        "median_voiced_probability": 0.0,
        "analysis_quality": "unavailable",
        "is_monotone": None,
        "pitch_status": message,
        "method": "pYIN with robust semitone contour statistics",
    }


def _analysis_quality(voiced_ratio: float, probability: float) -> str:
    if voiced_ratio >= 0.40 and probability >= 0.80:
        return "high"
    if voiced_ratio >= 0.20 and probability >= 0.60:
        return "moderate"
    return "low"


def _contour_movement(
    midi_contour: np.ndarray,
    sr: int,
    config: PitchConfig,
) -> float:
    lag = max(
        1,
        int(round(config.contour_interval_seconds * sr / config.hop_length)),
    )
    if midi_contour.size <= lag:
        return 0.0
    left, right = midi_contour[:-lag], midi_contour[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    differences = np.abs(right[valid] - left[valid])
    # Abrupt octave-like jumps are commonly tracking errors, not expressive
    # intonation. Exclude them only from the contour-movement measurement.
    differences = differences[
        differences <= config.maximum_tracker_jump_semitones
    ]
    return float(np.median(differences)) if differences.size else 0.0


def analyze_frequency_pitch(
    y: np.ndarray,
    sr: int,
    config: PitchConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Measure pitch variety without bias toward naturally high/low voices.

    Hertz is retained for display compatibility, but monotone classification is
    based on semitones, robust spread, and movement across the pitch contour.
    """
    signal = np.asarray(y, dtype=np.float32).reshape(-1)
    if signal.size == 0 or sr <= 0 or not np.all(np.isfinite(signal)):
        return _unavailable_result("No valid audio for pitch analysis")

    try:
        f0, voiced_flag, voiced_probability = librosa.pyin(
            signal,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
        )
    except (ValueError, FloatingPointError):
        return _unavailable_result("Pitch tracking failed")

    f0 = np.atleast_1d(np.asarray(f0, dtype=float))
    voiced_flag = np.atleast_1d(np.asarray(voiced_flag, dtype=bool))
    voiced_probability = np.atleast_1d(np.asarray(voiced_probability, dtype=float))
    frame_count = f0.size
    valid = (
        np.isfinite(f0)
        & voiced_flag
        & np.isfinite(voiced_probability)
        & (voiced_probability >= config.minimum_voiced_probability)
    )
    voiced_ratio = float(np.mean(valid)) if frame_count else 0.0
    if int(np.sum(valid)) < config.minimum_voiced_frames or voiced_ratio < config.minimum_voiced_ratio:
        result = _unavailable_result("Insufficient reliable voiced pitch")
        result["voiced_frame_ratio"] = round(voiced_ratio, 3)
        return result

    voiced_hz = f0[valid]
    probabilities = voiced_probability[valid]
    midi_contour = np.full(f0.shape, np.nan, dtype=float)
    midi_contour[valid] = librosa.hz_to_midi(voiced_hz)
    voiced_midi = midi_contour[valid]

    # Percentiles and MAD resist brief pYIN octave errors better than raw min,
    # max, or standard deviation.
    low_midi, high_midi = np.percentile(voiced_midi, [10, 90])
    pitch_range = float(high_midi - low_midi)
    midi_median = float(np.median(voiced_midi))
    robust_std = float(
        1.4826 * np.median(np.abs(voiced_midi - midi_median))
    )
    winsorized_midi = np.clip(voiced_midi, low_midi, high_midi)
    variation_semitones = float(np.std(winsorized_midi))
    movement = _contour_movement(midi_contour, sr, config)

    is_monotone = (
        pitch_range < config.monotone_range_semitones
        and robust_std < config.monotone_robust_std_semitones
        and movement < config.monotone_contour_movement_semitones
    )
    pitch_variety_score = round(
        min(1.0, pitch_range / 6.0) * 45.0
        + min(1.0, robust_std / 2.0) * 30.0
        + min(1.0, movement / 1.5) * 25.0,
        1,
    )
    median_probability = float(np.median(probabilities))
    quality = _analysis_quality(voiced_ratio, median_probability)
    if is_monotone:
        status = "Monotone/robotic"
    elif pitch_variety_score < 50:
        status = "Limited pitch variation"
    else:
        status = "Varied/expressive"

    return {
        "analysis_available": True,
        "median_pitch_hz": round(float(np.median(voiced_hz)), 2),
        # Legacy display field; no longer used to classify monotone delivery.
        "pitch_variation_hz": round(float(np.std(voiced_hz)), 2),
        "pitch_variation_semitones": round(variation_semitones, 3),
        "robust_pitch_std_semitones": round(robust_std, 3),
        "pitch_range_semitones": round(pitch_range, 3),
        "pitch_range_hz": round(
            float(np.percentile(voiced_hz, 90) - np.percentile(voiced_hz, 10)),
            2,
        ),
        "contour_movement_semitones": round(movement, 3),
        "pitch_variety_score": pitch_variety_score,
        "voiced_frame_ratio": round(voiced_ratio, 3),
        "median_voiced_probability": round(median_probability, 3),
        "analysis_quality": quality,
        "is_monotone": is_monotone,
        "pitch_status": status,
        "method": "pYIN with robust semitone contour statistics",
    }
