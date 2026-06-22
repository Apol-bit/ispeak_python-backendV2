"""Project-local ONNX Whisper model loading."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPEECH_MODEL_PATH = PROJECT_ROOT / "models" / "iSpeak_v3" / "model_files"


class ModelUnavailableError(RuntimeError):
    """Raised when required local model files or runtime packages are absent."""


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    path = Path(model_path).expanduser() if model_path else DEFAULT_SPEECH_MODEL_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelUnavailableError(f"Speech model directory not found: {path}") from exc
    if not resolved.is_dir():
        raise ModelUnavailableError(f"Speech model path is not a directory: {resolved}")
    if not (resolved / "config.json").is_file() or not any(resolved.glob("*.onnx")):
        raise ModelUnavailableError(
            f"Speech model directory is incomplete: {resolved} "
            "(config.json and ONNX files are required)"
        )
    return resolved


class OptimizedONNXWhisper:
    def __init__(self, model_path: str | Path):
        self.model_path = resolve_model_path(model_path)

        try:
            import onnxruntime as ort
            from transformers import WhisperProcessor, pipeline

            ort_model_class = importlib.import_module(
                "optimum.onnxruntime"
            ).ORTModelForSpeechSeq2Seq
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise ModelUnavailableError(
                "Missing ONNX speech dependencies. Run setup_backend.ps1 first."
            ) from exc

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            provider = "CUDAExecutionProvider"
            device_label = "GPU (CUDA)"
        else:
            provider = "CPUExecutionProvider"
            device_label = "CPU"

        print(f"Loading local ONNX model from {self.model_path} on {device_label}...")
        try:
            self.processor = WhisperProcessor.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self.model = ort_model_class.from_pretrained(
                str(self.model_path),
                provider=provider,
                use_merged=False,
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load local speech model from {self.model_path}: {exc}"
            ) from exc

        self.model.model.config = self.model.config
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            return_timestamps=True,
        )
        print("Local ONNX model loaded and ready.")

    def transcribe(self, file_path: str, **_: Any) -> dict[str, Any]:
        output = self.pipe(file_path)
        formatted_words = []
        formatted_segments = []
        full_text = output.get("text", "").strip()

        for chunk in output.get("chunks", []):
            start, end = chunk.get("timestamp", (0.0, 0.0))
            start = 0.0 if start is None else float(start)
            end = start + 1.0 if end is None else float(end)
            phrase = chunk.get("text", "").strip()
            words = phrase.split()
            if not words:
                continue

            word_duration = max(0.0, end - start) / len(words)
            current_start = start
            segment_words = []
            confidence = chunk.get("score")
            if not isinstance(confidence, (int, float)):
                confidence = None

            for word in words:
                word_result = {
                    "word": word,
                    "start": round(current_start, 2),
                    "end": round(current_start + word_duration, 2),
                    "probability": confidence,
                }
                formatted_words.append(word_result)
                segment_words.append(word_result)
                current_start += word_duration

            formatted_segments.append({
                "text": phrase,
                "start": round(start, 2),
                "end": round(end, 2),
                "words": segment_words,
            })

        total_end = formatted_words[-1]["end"] if formatted_words else 0.0
        return {
            "text": full_text,
            "segments": formatted_segments,
            "duration": total_end,
        }


def load_model(model_name: str | Path | None = None) -> OptimizedONNXWhisper:
    return OptimizedONNXWhisper(model_name or DEFAULT_SPEECH_MODEL_PATH)
