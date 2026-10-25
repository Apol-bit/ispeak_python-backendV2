#whisper_service.py
from __future__ import annotations

import numpy as np
import logging
import os
from contextlib import contextmanager
import librosa
from memory_diagnostics import memory_stage, memory_metadata
from audio_validation import (MIN_RMS, SpeechValidationError, load_valid_audio,
                              timestamp_validity_counts, validate_audio, validate_transcript)
from difflib import SequenceMatcher
from typing import Any, Dict, List

from audio_analysis_processing_files.articulation import (
    analyze_articulation,
    compare_reference_enunciation,
)
from audio_analysis_processing_files.filler_words import analyze_fillers
from assessment_details import assessment_details
from audio_analysis_processing_files.speaking_rate import analyze_speaking_rate
from audio_analysis_processing_files.vocal_variety import (
    analyze_frequency_pitch,
    analyze_signal_intensity,
    analyze_temporal_pauses,
)

logger = logging.getLogger(__name__)


@contextmanager
def _validation_stage(role, stage):
    """Log only fixed processing labels and validation codes, never audio/text."""
    try:
        yield
    except SpeechValidationError as exc:
        logger.warning('audio_validation_failed role=%s stage=%s code=%s',
                       role, getattr(exc, 'validation_stage', stage), exc.code)
        raise


def _log_transcription(text, words):
    logger.info("Transcription complete words=%d", len(words))
    if os.getenv("ISPEAK_ENV", "development").lower() != "production" and os.getenv("ISPEAK_DEBUG_SPEECH_LOGS") == "true":
        logger.debug("Full text: %s", text)
        for i, word in enumerate(words):
            logger.debug("Word %d: %s", i, word)


def _log_decode_diagnostics(diagnostics, words, duration, audio_role):
    """Emit fixed aggregate fields only; diagnostics must not affect analysis."""
    try:
        safe_role = audio_role if audio_role in ('user', 'reference') else 'unknown'
        segments = diagnostics.get('segments', []) if isinstance(diagnostics, dict) else []
        retries = 0
        for index, segment in enumerate(segments):
            initial = segment.get('initial_compression_ratio')
            final = segment.get('final_compression_ratio')
            initial = float(initial) if isinstance(initial, (int, float)) else -1.0
            final = float(final) if isinstance(final, (int, float)) else -1.0
            retry_count = int(segment.get('retry_count', 0))
            elapsed_value = segment.get('elapsed_seconds')
            elapsed_available = isinstance(elapsed_value, (int, float))
            elapsed = float(elapsed_value) if elapsed_available else -1.0
            retries += retry_count
            logger.info(
                'whisper_decode_segment role=%s segment_index=%d '
                'compression_ratio_initial=%.6f compression_ratio_final=%.6f '
                'retry_count=%d elapsed_seconds=%.6f elapsed_available=%s',
                safe_role, index, initial, final, retry_count, elapsed,
                str(elapsed_available).lower(),
            )
        timestamp_total, timestamp_usable = timestamp_validity_counts(words, duration)
        decode_elapsed = (float(diagnostics.get('elapsed_seconds', 0.0))
                          if isinstance(diagnostics, dict) else 0.0)
        long_form = bool(diagnostics.get('long_form', False)) if isinstance(diagnostics, dict) else False
        logger.info(
            'whisper_decode_summary role=%s long_form=%s segment_count=%d '
            'retry_count=%d elapsed_seconds=%.6f timestamp_total=%d timestamp_usable=%d',
            safe_role, str(long_form).lower(), len(segments), retries, decode_elapsed,
            timestamp_total, timestamp_usable,
        )
    except Exception:
        # Telemetry failures must not change transcription or scoring.
        pass

# Compatibility exports used by the standalone evaluation utility.
# Validation is enforced by audio_validation before ASR and scoring.
RMS_SILENCE_THRESHOLD = MIN_RMS
MIN_WORD_COUNT = 3


def _extract_word_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten and remap Whisper word segments for pronunciation analysis."""
    return [
        {
            "text": word.get("word", ""),
            "start": word.get("start"),
            "end": word.get("end"),
            "confidence": word.get("probability"),
        }
        for seg in segments
        for word in seg.get("words", [])
    ]


def _extract_word_timestamps(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract clean word-level timestamps for the teleprompter UI."""
    timestamps = []
    for seg in segments:
        for word in seg.get("words", []):
            w_text = word.get("word", "").strip()
            if w_text:
                timestamps.append({
                    "word": w_text,
                    "start": round(word.get("start", 0.0), 3),
                    "end": round(word.get("end", 0.0), 3),
                })
    return timestamps


def _compute_pacing_score(pacing_stats: Dict[str, Any]) -> float:
    """
    Convert pacing status into a 0-100 score.

    Excellent pacing â†’ 100
    Slow or fast     â†’ scaled by how far WPM is from the ideal range
    """
    status = pacing_stats.get("pacing_status", "")
    wpm = pacing_stats.get("wpm", 0.0)

    if status in ("No speech detected", "Insufficient speech", "Analysis failed"):
        return 0.0

    if status == "Excellent pacing":
        return 100.0

    # Ideal range is 120-150 WPM (from calculate_pacing defaults)
    if status == "Slow pacing":
        # 0 WPM â†’ 0, approaching 120 WPM â†’ 100
        return round(min(100.0, max(0.0, (wpm / 120.0) * 100)), 1)

    if status == "Fast pacing":
        # 150 WPM â†’ 100, penalty grows beyond that
        excess = wpm - 150.0
        return round(min(100.0, max(0.0, 100.0 - (excess / 150.0) * 100)), 1)

    # fallback (e.g. "Analysis failed", "No speech detected")
    return 0.0


def _compute_energy_score(energy_stats: Dict[str, Any]) -> float:
    """
    Convert energy stats into a 0-100 score.

    Factors:
    - loudness_status: Normal volume â†’ full points
    - is_low_variation: penalize monotone energy
    - is_monotone: penalize monotone pitch
    """
    score = 100.0

    loudness_status = energy_stats.get("loudness_status", "")
    if loudness_status == "Too quiet (whispering)":
        score -= 40.0
    elif loudness_status == "Too loud (shouting)":
        score -= 30.0
    elif loudness_status == "Silence":
        return 0.0

    if energy_stats.get("is_low_variation"):
        score -= 20.0

    if energy_stats.get("is_monotone"):
        score -= 20.0
    elif isinstance(energy_stats.get("pitch_variety_score"), (int, float)):
        if energy_stats["pitch_variety_score"] < 50:
            score -= 10.0

    score = max(0.0, score)
    pause_score = energy_stats.get("temporal_pauses", {}).get("pause_score")
    if isinstance(pause_score, (int, float)):
        score = score * 0.80 + float(pause_score) * 0.20

    return round(max(0.0, score), 1)


def _compute_overall_score(
    pacing_score: float,
    clarity_score: float,
    energy_score: float,
) -> float:
    """
    Weighted average of all three scores.

    Weights:
    - Clarity     40%  (most important for speech quality)
    - Pacing      35%
    - Energy      25%
    """
    overall = (
        clarity_score * 0.40 +
        pacing_score  * 0.35 +
        energy_score  * 0.25
    )
    return round(overall, 1)


def _run_analysis(
    file_path: str,
    y: np.ndarray,
    sr: int,
    model,
    language: str = "English",
    audio_role: str = "user",
) -> Dict[str, Any]:
    """
    Core analysis logic shared by both standard and reference-based analysis.

    Returns analysis for usable speech; invalid input raises SpeechValidationError.
    """

    with _validation_stage(audio_role, 'signal_validation'), memory_stage('signal_validation', audio_role):
        audio_duration = validate_audio(y, sr)
    logger.info(
        'whisper_input_metadata role=%s duration_seconds=%.6f sample_rate=%d '
        'channels=1 samples=%d',
        audio_role if audio_role in ('user', 'reference') else 'unknown',
        audio_duration, sr, len(y),
    )
    y_original = y.copy()

    # ---------- TRANSCRIBE ----------
    # Passing an initial_prompt with filler words strongly biases Whisper
    # against filtering out hesitation sounds like "um" and "uh" from the transcript.
    # condition_on_previous_text=False prevents Whisper from using its own
    # previous output to suppress filler words in later segments.
    with memory_stage('whisper_word_timestamps', audio_role):
        transcription = model.transcribe(
            file_path,
            audio_data=y_original,
            word_timestamps=True,
            initial_prompt="Umm, uh, hmm, like, you know, ah, er, um, ano, parang, yung, basically, actually.",
            condition_on_previous_text=False,
            language=language,
            audio_duration_seconds=audio_duration,
        )

    text: str = transcription.get("text", "").strip()
    segments: List[Dict[str, Any]] = transcription.get("segments", [])

    word_segments = _extract_word_segments(segments)
    _log_decode_diagnostics(
        transcription.get('_decoding_diagnostics', {}), word_segments,
        audio_duration, audio_role,
    )
    with _validation_stage(audio_role, 'transcript_validation'):
        validate_transcript(text, word_segments, audio_duration,
                            language=language, audio_role=audio_role)

    _log_transcription(text, word_segments)

    # ---------- WORD TIMESTAMPS (for Teleprompter UI) ----------
    word_timestamps = _extract_word_timestamps(segments)

    # ---------- AUDIO DURATION ----------
    audio_duration = librosa.get_duration(y=y, sr=sr)

    # ---------- ENERGY ----------
    # Use ORIGINAL audio â€” normalization flattens loudness, making score always 100
    with memory_stage('intensity', audio_role):
        intensity_stats = analyze_signal_intensity(y_original, sr)
    with memory_stage('pitch', audio_role):
        frequency_stats = analyze_frequency_pitch(y_original, sr)
    with memory_stage('pauses', audio_role):
        pause_stats = analyze_temporal_pauses(word_segments)
    energy_stats = {
        "average_volume_db": intensity_stats.get("average_volume_db", -80.0),
        "dynamic_range": intensity_stats.get("dynamic_range_db", 0.0),
        "loudness_status": intensity_stats.get("loudness_status", "Silence"),
        "is_low_variation": intensity_stats.get("is_low_intensity_variation", True),
        "pitch_variation_hz": frequency_stats.get("pitch_variation_hz", 0.0),
        "pitch_variety_score": frequency_stats.get("pitch_variety_score"),
        "is_monotone": frequency_stats.get("is_monotone", True),
        "signal_intensity": intensity_stats,
        "frequency_pitch": frequency_stats,
        "temporal_pauses": pause_stats,
    }
    energy_score = _compute_energy_score(energy_stats)

    # ---------- PACING ----------
    try:
        pacing_stats = analyze_speaking_rate(word_segments, audio_duration, full_transcript=text)
        pacing_stats["total_pause_seconds"] = pause_stats["total_pause_seconds"]
        pacing_score = _compute_pacing_score(pacing_stats)
    except ValueError as e:
        logger.error("Pacing analysis failed category=%s", type(e).__name__)
        pacing_stats = {}
        pacing_score = 0.0

    # ---------- PRONUNCIATION ----------
    with memory_stage('articulation', audio_role):
        articulation_stats = analyze_articulation(word_segments, y=y_original, sr=sr)
    pronunciation_score = float(articulation_stats.get("articulation_score", 0))

    # ---------- FILLER ----------
    with memory_stage('filler_classification', audio_role):
        filler_stats = analyze_fillers(word_segments, full_transcript=text)
    raw_filler_score = filler_stats.get("filler_score")
    filler_score = (
        float(raw_filler_score)
        if isinstance(raw_filler_score, (int, float)) else None
    )

    # ---------- CLARITY ----------
    if filler_score is None:
        # Do not reward or punish the speaker when the local classifier is absent.
        clarity_score = round(pronunciation_score)
    else:
        clarity_score = round(
            pronunciation_score * 0.60 +
            filler_score        * 0.40
        )

    # ---------- OVERALL ----------
    overall_score = _compute_overall_score(pacing_score, clarity_score, energy_score)

    return {
        "analysis_valid": True,
        "transcription": text,
        "language": language,
        "duration_seconds": round(float(audio_duration), 3),
        "word_timestamps": word_timestamps,
        "scores": {
            "overall":    overall_score,
            "clarity":    clarity_score,
            "pacing":     pacing_score,
            "energy":     energy_score,
            "vocal_variety": energy_score if frequency_stats.get("analysis_available") else None,
            "articulation": pronunciation_score,
            "speaking_rate": pacing_score,
            "filler_words": filler_score,
        },
        "pacing": {
            "wpm": pacing_stats.get("wpm", 0.0) if isinstance(pacing_stats, dict) else 0.0,
            "message": pacing_stats.get("pacing_status", "") if isinstance(pacing_stats, dict) else "Analysis failed"
        },
        "speaking_rate": {
            "score": pacing_score,
            "wpm": pacing_stats.get("wpm", 0.0) if isinstance(pacing_stats, dict) else 0.0,
            "articulation_rate": pacing_stats.get("articulation_rate", 0.0) if isinstance(pacing_stats, dict) else 0.0,
            "message": pacing_stats.get("pacing_status", "") if isinstance(pacing_stats, dict) else "Analysis failed",
        },
        "pronunciation": {
            "score":             pronunciation_score,
            "message":           articulation_stats.get("message", ""),
            "problematic_words": articulation_stats.get("clear_enunciation", {}).get(
                "unclear_words", []
            ),
        },
        "articulation": {
            "score": pronunciation_score,
            "message": articulation_stats.get("message", ""),
            "accurate_pronunciation": articulation_stats.get("accurate_pronunciation", {}),
            "clear_enunciation": articulation_stats.get("clear_enunciation", {}),
        },
        "vocal_variety": {
            "score": energy_score if frequency_stats.get("analysis_available") else None,
            "analysis_available": frequency_stats.get("analysis_available", False),
            "energy_score_is_partial": not frequency_stats.get("analysis_available", False),
            "signal_intensity": intensity_stats,
            "frequency_pitch": frequency_stats,
            "temporal_pauses": pause_stats,
        },
        "fillers": {
            "score":   filler_score,
            "analysis_available": filler_stats.get("analysis_available", False),
            "count":   filler_stats.get("filler_count", 0),
            "rate":    filler_stats.get("filler_rate", 0.0),
            "words":   [
                {
                    "word":  fw.get("word", ""),
                    "confidence": fw.get("confidence"),
                    "start": round(fw.get("start", 0.0), 3),
                    "end":   round(fw.get("end", 0.0), 3),
                }
                for fw in filler_stats.get("filler_words", [])
            ],
            "candidates": filler_stats.get("filler_candidates", []),
            "candidate_count": filler_stats.get("candidate_count", 0),
            "count_unit": filler_stats.get("count_unit", "grouped_spans"),
            "threshold": filler_stats.get("threshold", 0.60),
            "failure": filler_stats.get("failure"),
            "coverage": filler_stats.get("coverage"),
            "message": filler_stats.get("message", ""),
            "method": filler_stats.get("method", ""),
        },
        # Internal â€” used by reference comparison
        "_internal": {
            "energy_stats": energy_stats,
            "energy_rule_base": _compute_energy_score({**energy_stats, "temporal_pauses": {}}),
            "pacing_stats": pacing_stats,
            "pacing_score": pacing_score,
            "clarity_score": clarity_score,
            "energy_score": energy_score,
            "articulation_stats": articulation_stats,
            "filler_score": filler_score,
        },
    }


def generate_full_analysis(
    file_path: str,
    model,
    language: str = "English",
    max_duration_seconds: float = 300,
) -> Dict[str, Any]:
    """
    Run full speech analysis on an audio file (standard mode, no reference).

    Returns
    -------
    dict with keys:
        transcription, word_timestamps, scores, pacing, pronunciation, fillers
    """

    # ---------- LOAD AUDIO ----------
    with _validation_stage('user', 'decode'), memory_stage('decode_validate', 'user'):
        y, sr = load_valid_audio(file_path, max_duration_seconds)
        memory_metadata(audio=y, sample_rate=sr)
    sr = int(sr)

    result = _run_analysis(file_path, y, sr, model, language)
    result['assessment_details'] = assessment_details(result)

    # Remove internal data before returning
    result.pop("_internal", None)

    return result


def generate_reference_analysis(
    user_path: str,
    reference_path: str,
    model,
    language: str = "English",
    max_duration_seconds: float = 300,
) -> Dict[str, Any]:
    """
    Run reference-based speech analysis.

    Compares the user's recorded audio against the Validator's reference audio.
    The reference audio's metrics serve as the "perfect score" baseline.
    Scores are adjusted based on how closely the user matches the reference.

    Returns
    -------
    dict â€” same structure as generate_full_analysis, with scores adjusted
           relative to the reference baseline.
    """

    logger.info("=== REFERENCE-BASED ANALYSIS ===")

    # ---------- LOAD BOTH AUDIO FILES ----------
    with _validation_stage('user', 'decode'), memory_stage('decode_validate', 'user'):
        y_user, sr = load_valid_audio(user_path, max_duration_seconds)
        memory_metadata(audio=y_user, sample_rate=sr)
    sr = int(sr)
    # Reference comparison is optional: scripts without reference audio already
    # use the standard assessment. Only validation failures permit this fallback;
    # model, storage and unexpected runtime failures must still propagate.
    ref_result = None
    try:
        with _validation_stage('reference', 'decode'), memory_stage('decode_validate', 'reference'):
            y_ref, sr_ref = load_valid_audio(reference_path)
            memory_metadata(role='reference', audio=y_ref, sample_rate=sr_ref)
        sr_ref = int(sr_ref)
        logger.info("Analyzing reference audio...")
        ref_result = _run_analysis(reference_path, y_ref, sr_ref, model, language, audio_role='reference')
    except SpeechValidationError:
        logger.warning('reference_comparison_unavailable reason=reference_validation_failed')

    logger.info("Analyzing user audio...")
    user_result = _run_analysis(user_path, y_user, sr, model, language)

    # If either analysis returned silence, return the user result as-is
    if ref_result is None or not user_result.get("_internal") or not ref_result.get("_internal"):
        user_result['articulation']['clear_enunciation']['reference_comparison'] = {
            'available': False, 'applied': False,
            'message': 'Reference comparison unavailable; standard user-speech assessment used.',
        }
        user_result['assessment_details'] = assessment_details(user_result)
        user_result['assessment_details']['fallback']['referencePacing'] = True
        user_result['assessment_details']['completeness'] = 'partial'
        user_result.pop("_internal", None)
        return user_result

    ref_internal = ref_result["_internal"]
    user_internal = user_result["_internal"]

    # ---------- REFERENCE-BASED PACING SCORE ----------
    # Compare user WPM against reference WPM (instead of the fixed 120-150 range)
    ref_wpm = ref_internal["pacing_stats"].get("wpm", 130.0) if isinstance(ref_internal["pacing_stats"], dict) else 130.0
    user_wpm = user_internal["pacing_stats"].get("wpm", 0.0) if isinstance(user_internal["pacing_stats"], dict) else 0.0

    if ref_wpm > 0 and user_wpm > 0:
        wpm_ratio = user_wpm / ref_wpm
        # Perfect ratio = 1.0 â†’ score 100. Penalty for deviation.
        deviation = abs(1.0 - wpm_ratio)
        ref_pacing_score = round(max(0.0, 100.0 - (deviation * 100.0)), 1)
    else:
        ref_pacing_score = user_internal["pacing_score"]

    # ---------- REFERENCE-BASED ENERGY SCORE ----------
    # Compare energy profiles using RMS correlation
    ref_energy = ref_internal["energy_stats"]
    user_energy = user_internal["energy_stats"]

    # Use the user's energy score but adjust based on reference comparison
    ref_energy_score = user_internal["energy_score"]
    ref_is_monotone = ref_energy.get("is_monotone", False)
    user_is_monotone = user_energy.get("is_monotone", False)

    # If reference is monotone but user isn't, that's good â€” bonus
    if ref_is_monotone and not user_is_monotone:
        ref_energy_score = min(100.0, ref_energy_score + 10.0)
    # If user is monotone but reference isn't, that's bad â€” penalty already applied
    ref_energy_score = round(ref_energy_score, 1)

    # ---------- REFERENCE-BASED ENUNCIATION ----------
    with memory_stage('reference_dtw'):
        enunciation_comparison = compare_reference_enunciation(y_user, y_ref, sr)
    transcript_similarity = SequenceMatcher(
        None,
        ref_result.get("transcription", "").lower().split(),
        user_result.get("transcription", "").lower().split(),
    ).ratio()
    enunciation_comparison["transcript_similarity"] = round(transcript_similarity, 3)

    user_articulation = user_internal["articulation_stats"]
    base_enunciation = float(
        user_articulation.get("clear_enunciation", {}).get("enunciation_score", 0)
    )
    enunciation_comparison['base_enunciation_score'] = base_enunciation
    if enunciation_comparison.get("available") and transcript_similarity >= 0.60:
        reference_score = float(enunciation_comparison["score"])
        adjusted_enunciation = round(base_enunciation * 0.40 + reference_score * 0.60, 1)
        enunciation_comparison["applied"] = True
    else:
        adjusted_enunciation = base_enunciation
        enunciation_comparison["applied"] = False
        if transcript_similarity < 0.60:
            enunciation_comparison["message"] = (
                "Reference comparison not applied because transcripts differ"
            )

    accuracy_score = user_articulation.get("accurate_pronunciation", {}).get(
        "pronunciation_accuracy_score"
    )
    if isinstance(accuracy_score, (int, float)):
        adjusted_articulation = round(float(accuracy_score) * 0.60 + adjusted_enunciation * 0.40, 1)
    else:
        adjusted_articulation = adjusted_enunciation

    filler_score = user_internal.get("filler_score")
    if isinstance(filler_score, (int, float)):
        ref_clarity_score = round(adjusted_articulation * 0.60 + float(filler_score) * 0.40)
    else:
        ref_clarity_score = round(adjusted_articulation)

    user_result["articulation"]["score"] = adjusted_articulation
    user_result["articulation"]["clear_enunciation"][
        "enunciation_score"
    ] = adjusted_enunciation
    user_result["articulation"]["clear_enunciation"][
        "reference_comparison"
    ] = enunciation_comparison
    user_result["pronunciation"]["score"] = adjusted_articulation
    user_result["scores"]["articulation"] = adjusted_articulation
    user_result["scores"]["clarity"] = ref_clarity_score

    # ---------- RECALCULATE OVERALL ----------
    ref_overall = _compute_overall_score(ref_pacing_score, ref_clarity_score, ref_energy_score)

    # ---------- UPDATE USER RESULT ----------
    user_result["scores"]["pacing"] = ref_pacing_score
    user_result["scores"]["energy"] = ref_energy_score
    user_result["scores"]["speaking_rate"] = ref_pacing_score
    pitch_available = user_result["vocal_variety"].get("analysis_available", False)
    user_result["scores"]["vocal_variety"] = ref_energy_score if pitch_available else None
    user_result["speaking_rate"]["score"] = ref_pacing_score
    user_result["vocal_variety"]["score"] = ref_energy_score if pitch_available else None
    user_result["speaking_rate"]["reference_wpm"] = ref_wpm
    user_result["scores"]["overall"] = ref_overall
    user_result['assessment_details'] = assessment_details(user_result)

    # Clean up internal data
    user_result.pop("_internal", None)

    logger.info(
        "Reference comparison â€” Ref WPM: %.1f, User WPM: %.1f, "
        "Adj Pacing: %.1f, Adj Energy: %.1f, Adj Overall: %.1f",
        ref_wpm, user_wpm, ref_pacing_score, ref_energy_score, ref_overall,
    )

    return user_result
