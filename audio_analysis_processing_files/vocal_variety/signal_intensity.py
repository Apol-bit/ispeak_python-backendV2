"""Measure voice loudness and signal-intensity variation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

WHISPER_THRESHOLD_DBFS = -35.0
SHOUT_THRESHOLD_DBFS = -10.0
LOW_VARIATION_DB = 6.0
# A frame must clear both this absolute floor and the relative gate below.  The
# absolute floor prevents digital silence/quantisation noise from being treated
# as speech while still allowing genuinely quiet recordings to be measured.
ABSOLUTE_GATE_DBFS = -50.0
RELATIVE_GATE_DB = 30.0
MIN_ACTIVE_DURATION_SECONDS = 0.05
DEFAULT_FRAME_LENGTH = 1024
DEFAULT_HOP_LENGTH = 256


def _silence_result() -> Dict[str, Any]:
    return {
        "average_volume_db": -80.0,
        "dynamic_range_db": 0.0,
        "loudness_status": "Silence",
        "is_low_intensity_variation": True,
        "active_frame_ratio": 0.0,
        "peak_volume_db": -80.0,
        "clipping_ratio": 0.0,
        "method": "gated RMS dBFS with robust active-frame percentiles",
    }


def analyze_signal_intensity(
    y: np.ndarray,
    sr: int,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> Dict[str, Any]:
    """Estimate recording level and vocal intensity variation in dBFS.

    The loudness labels describe the *recorded* level, not calibrated acoustic
    sound-pressure level.  Real-world whispering or shouting cannot be inferred
    reliably without microphone calibration and a known microphone distance.
    """

    signal = np.asarray(y, dtype=np.float32).reshape(-1)
    if (
        signal.size == 0
        or sr <= 0
        or frame_length <= 0
        or hop_length <= 0
        or not np.all(np.isfinite(signal))
    ):
        return _silence_result()

    original_peak = float(np.max(np.abs(signal)))
    clipping_ratio = float(np.mean(np.abs(signal) >= 0.999))

    # Microphone/interface DC offset adds energy that is not audible loudness.
    signal = signal - np.mean(signal, dtype=np.float64)

    # Pad only clips shorter than one frame; do not pad ordinary frame boundaries,
    # because that creates artificial intensity variation at the clip edges.
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))

    # Cumulative energy avoids materialising a large, overlapping frame matrix
    # for long recordings.
    squared = np.square(signal, dtype=np.float64)
    cumulative_energy = np.concatenate(([0.0], np.cumsum(squared)))
    frame_starts = np.arange(
        0, signal.size - frame_length + 1, hop_length, dtype=np.int64
    )
    frame_energy = (
        cumulative_energy[frame_starts + frame_length]
        - cumulative_energy[frame_starts]
    )
    rms_frames = np.sqrt(frame_energy / frame_length)
    frame_db = 20.0 * np.log10(np.maximum(rms_frames, 1e-10))

    # A relative gate removes pauses even when the recording has a raised noise
    # floor.  Basing it on P90 rather than the maximum makes it resistant to a
    # click or other one-frame transient.
    reference_db = float(np.percentile(frame_db, 90))
    gate_db = max(ABSOLUTE_GATE_DBFS, reference_db - RELATIVE_GATE_DB)
    active_mask = frame_db >= gate_db
    active_ratio = float(np.mean(active_mask))
    minimum_active_frames = max(
        1,
        int(np.ceil(MIN_ACTIVE_DURATION_SECONDS * sr / hop_length)),
    )
    required_active_frames = min(minimum_active_frames, rms_frames.size)
    if int(np.count_nonzero(active_mask)) < required_active_frames:
        return _silence_result()

    # Loudness is calculated from active voice frames so pauses are handled by
    # the temporal-pause component instead of making the voice look quieter.
    active_rms = np.maximum(rms_frames[active_mask], 1e-10)
    db_levels = frame_db[active_mask]

    # Cap only the highest 5% before energy averaging so an isolated handling
    # noise does not turn an otherwise normal recording into "shouting".
    rms_cap = float(np.percentile(active_rms, 95))
    robust_rms = np.minimum(active_rms, rms_cap)
    average_db = float(10.0 * np.log10(np.mean(np.square(robust_rms))))
    if db_levels.size >= 4:
        low, high = np.percentile(db_levels, [10, 90])
    elif db_levels.size >= 2:
        low, high = np.min(db_levels), np.max(db_levels)
    else:
        low = high = float(db_levels[0])
    dynamic_range = max(0.0, float(high - low))

    if average_db < WHISPER_THRESHOLD_DBFS:
        status = "Too quiet (whispering)"
    elif average_db > SHOUT_THRESHOLD_DBFS:
        status = "Too loud (shouting)"
    else:
        status = "Normal volume"

    return {
        "average_volume_db": round(average_db, 2),
        "dynamic_range_db": round(dynamic_range, 2),
        "loudness_status": status,
        "is_low_intensity_variation": dynamic_range < LOW_VARIATION_DB,
        "active_frame_ratio": round(active_ratio, 3),
        "peak_volume_db": round(
            float(20.0 * np.log10(max(original_peak, 1e-10))),
            2,
        ),
        "clipping_ratio": round(clipping_ratio, 6),
        "method": "gated RMS dBFS with robust active-frame percentiles",
    }
