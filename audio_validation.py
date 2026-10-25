"""Conservative input checks; these reject obvious invalid audio, not diagnose speech."""
from collections import Counter
import re
import logging
import subprocess
import numpy as np
import librosa
import soundfile
import audioread

MIN_DURATION_SECONDS = 1.0
MIN_ACTIVE_SECONDS = 0.25
MIN_RMS = 1e-4  # -80 dBFS: a digital-silence floor, not a microphone loudness target.
MAX_DURATION_SECONDS = 300.0
DURATION_TOLERANCE_SECONDS = 0.5


def duration_limit(value=MAX_DURATION_SECONDS):
    try:
        if isinstance(value, bool):
            raise ValueError()
        limit = float(value)
        if not np.isfinite(limit) or not 1 <= limit <= MAX_DURATION_SECONDS:
            raise ValueError()
        return limit
    except (TypeError, ValueError):
        raise SpeechValidationError('invalid_duration', 'Invalid recording duration metadata.')


def check_duration(seconds, maximum=MAX_DURATION_SECONDS):
    if seconds > duration_limit(maximum) + DURATION_TOLERANCE_SECONDS:
        raise SpeechValidationError('recording_too_long', 'The recording exceeds the permitted duration. Please record a shorter speech.')


def declared_duration(value, maximum):
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 16 or not re.fullmatch(r'\d+(?:\.\d{1,3})?', value):
        raise SpeechValidationError('invalid_duration', 'Invalid recording duration metadata.')
    seconds = float(value)
    if not np.isfinite(seconds) or seconds > maximum:
        raise SpeechValidationError('recording_too_long', 'The recording exceeds the permitted duration.')
    return seconds

class SpeechValidationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

def insufficient():
    return SpeechValidationError("insufficient_speech", "The recording does not contain enough usable speech. Please record a short sentence in a quiet place and try again.")

def validate_audio(y, sr):
    signal = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr <= 0 or signal.size == 0:
        raise insufficient()
    check_duration(signal.size / sr)
    if not np.all(np.isfinite(signal)):
        raise insufficient()
    if signal.size / sr < MIN_DURATION_SECONDS:
        raise SpeechValidationError("recording_too_short", "The recording is too short. Please record at least one second of speech, preferably a complete sentence.")
    centered = signal - np.mean(signal, dtype=np.float64)
    frame_length = max(1, round(sr * 0.064))
    frames = centered[:signal.size // frame_length * frame_length].reshape(-1, frame_length)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    active = rms >= max(MIN_RMS, float(np.max(rms)) * 0.05)
    if np.count_nonzero(active) * frame_length / sr < MIN_ACTIVE_SECONDS:
        raise insufficient()
    # Only reject near-pure stationary tones: nearly all spectral energy in
    # three neighbouring bins, in at least 90% of active frames. Ordinary
    # voiced speech has harmonics and is not rejected for being monotone.
    selected = frames[active]
    selected = selected[::max(1, len(selected) // 256)]
    power = np.abs(np.fft.rfft(selected * np.hanning(frame_length), axis=1)) ** 2
    peak = np.argmax(power, axis=1)
    rows = np.arange(len(power))
    concentrated = sum(power[rows, np.clip(peak + offset, 0, power.shape[1]-1)] for offset in (-1, 0, 1))
    ratio = concentrated / np.maximum(power.sum(axis=1), 1e-20)
    if np.mean(ratio > 0.98) >= 0.90:
        raise insufficient()
    return signal.size / sr

def load_valid_audio(file_path, max_duration_seconds=MAX_DURATION_SECONDS):
    # Fixed output format and a bounded decode window prevent small compressed
    # uploads from expanding into unbounded arrays. One extra sample detects
    # over-limit audio rather than silently truncating it into a valid recording.
    maximum = duration_limit(max_duration_seconds)
    sample_rate = 16000
    frames = int((maximum + DURATION_TOLERANCE_SECONDS) * sample_rate)
    try:
        decoded = subprocess.run([
            'ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-filter_threads', '1',
            '-protocol_whitelist', 'file,pipe', '-i', str(file_path),
            '-map', '0:a:0', '-vn', '-sn', '-dn', '-t', str((frames + 1) / sample_rate),
            '-af', f'aresample={sample_rate},atrim=end_sample={frames + 1}',
            '-ac', '1', '-ar', str(sample_rate), '-f', 'f32le', 'pipe:1',
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Audio decoder unavailable') from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SpeechValidationError('invalid_audio', 'This audio could not be decoded safely. Please record again.') from exc
    try:
        y = np.frombuffer(decoded.stdout, dtype='<f4')
    except ValueError as exc:
        raise SpeechValidationError('invalid_audio', 'The decoded audio is invalid.') from exc
    try:
        check_duration(y.size / sample_rate, maximum)
        validate_audio(y, sample_rate)
    except SpeechValidationError as exc:
        # Internal diagnostic metadata only; public error codes/messages stay stable.
        exc.validation_stage = 'signal_validation'
        raise
    return y, sample_rate


def timestamp_validity_counts(words, duration):
    """Return the raw word count and the exact count accepted by validation."""
    tolerance = min(1.0, max(0.25, duration * 0.05))
    usable = 0
    for word in words:
        start, end = word.get('start'), word.get('end')
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= duration + tolerance:
                usable += 1
    return len(words), usable


def validate_transcript(text, words, duration, *, language=None, audio_role='user'):
    tokens = re.findall(r"[\w'-]+", text.lower(), re.UNICODE)
    def reject(rule, phrase_size=0, phrase_repeats=0, phrase_fraction=0.0):
        # Only fixed labels and aggregate numbers are emitted. Never include
        # tokens, word objects, exception text, paths or arbitrary caller input.
        _, usable_count = timestamp_validity_counts(words, duration)
        most_common_count = Counter(tokens).most_common(1)[0][1] if tokens else 0
        safe_language = language if language in ('English', 'Filipino', 'Taglish') else 'unknown'
        safe_role = audio_role if audio_role in ('user', 'reference') else 'unknown'
        logging.getLogger(__name__).warning(
            'transcript_validation_rejected code=insufficient_speech role=%s rule=%s '
            'language=%s duration_seconds=%.6f recognized_tokens=%d '
            'timestamp_total=%d timestamp_usable=%d tokens_per_second=%.6f '
            'most_common_token_count=%d most_common_token_fraction=%.6f '
            'phrase_size=%d phrase_repeats=%d phrase_fraction=%.6f',
            safe_role, rule, safe_language, duration, len(tokens), len(words), usable_count,
            len(tokens) / duration if duration > 0 else 0.0,
            most_common_count, most_common_count / len(tokens) if tokens else 0.0,
            phrase_size, phrase_repeats, phrase_fraction,
        )
        raise insufficient()

    if len(tokens) < 3 or not words:
        reject('minimum_recognized_tokens' if len(tokens) < 3 else 'missing_word_timestamps')
    # Conservative guards for runaway output, not ordinary hesitation.
    if len(tokens) >= 20 and len(tokens) / duration > 10:
        reject('excessive_token_rate')
    if len(tokens) >= 12:
        if Counter(tokens).most_common(1)[0][1] / len(tokens) >= 0.80:
            reject('single_token_dominance')
        for size in (2, 3, 4):
            for start in range(len(tokens) - size * 6 + 1):
                phrase = tokens[start:start+size]
                end = start + size
                while tokens[end:end+size] == phrase:
                    end += size
                if end-start >= size*6 and (end-start)/len(tokens) >= 0.80:
                    reject('repeated_phrase_dominance', size, (end-start)//size, (end-start)/len(tokens))
    _, usable = timestamp_validity_counts(words, duration)
    if usable < max(1, len(words) * 0.5):
        reject('insufficient_usable_timestamps')
