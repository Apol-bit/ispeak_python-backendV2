"""Conservative input checks; these reject obvious invalid audio, not diagnose speech."""
from collections import Counter
import re
import numpy as np
import librosa
import soundfile
import audioread

MIN_DURATION_SECONDS = 1.0
MIN_ACTIVE_SECONDS = 0.25
MIN_RMS = 1e-4  # -80 dBFS: a digital-silence floor, not a microphone loudness target.

class SpeechValidationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

def insufficient():
    return SpeechValidationError("insufficient_speech", "The recording does not contain enough usable speech. Please record a short sentence in a quiet place and try again.")

def validate_audio(y, sr):
    signal = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr <= 0 or signal.size == 0 or not np.all(np.isfinite(signal)):
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

def load_valid_audio(file_path):
    # PCM/container decoding does not need Librosa's lazy Numba initialization.
    # Validate at the original rate before resampling or loading ASR features.
    try:
        y, sr = soundfile.read(file_path, dtype="float32")
    except soundfile.LibsndfileError:
        # Preserve the existing decoder fallback for formats such as AAC.
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
        except (soundfile.LibsndfileError, audioread.NoBackendError, EOFError, ValueError) as exc:
            raise SpeechValidationError("invalid_audio", "This audio could not be read. Please upload a valid recording in a supported audio format.") from exc
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    validate_audio(y, int(sr))
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y, 16000


def validate_transcript(text, words, duration):
    tokens = re.findall(r"[\w'-]+", text.lower(), re.UNICODE)
    if len(tokens) < 3 or not words:
        raise insufficient()
    # Conservative guards for runaway output, not ordinary hesitation.
    if len(tokens) >= 20 and len(tokens) / duration > 10:
        raise insufficient()
    if len(tokens) >= 12:
        if Counter(tokens).most_common(1)[0][1] / len(tokens) >= 0.80:
            raise insufficient()
        for size in (2, 3, 4):
            for start in range(len(tokens) - size * 6 + 1):
                phrase = tokens[start:start+size]
                end = start + size
                while tokens[end:end+size] == phrase:
                    end += size
                if end-start >= size*6 and (end-start)/len(tokens) >= 0.80:
                    raise insufficient()
    usable = 0
    tolerance = min(1.0, max(0.25, duration * 0.05))
    for word in words:
        start, end = word.get("start"), word.get("end")
        if isinstance(start, (int,float)) and isinstance(end, (int,float)):
            if np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= duration+tolerance:
                usable += 1
    if usable < max(1, len(words) * 0.5):
        raise insufficient()
