"""Project-local iSpeak Whisper PEFT adapter loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "iSpeak_v5"
DEFAULT_SPEECH_MODEL_PATH = PROJECT_ROOT / "models" / DEFAULT_MODEL_NAME
DEFAULT_BASE_MODEL_PATH = DEFAULT_SPEECH_MODEL_PATH / "base_model"
MODEL_PATH_ENV = "ISPEAK_MODEL_PATH"
BASE_MODEL_PATH_ENV = "ISPEAK_BASE_MODEL_PATH"

_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


class ModelUnavailableError(RuntimeError):
    """Raised when required local model files or runtime packages are absent."""


def configured_adapter_path(model_path: str | Path | None = None) -> Path:
    """Return the configured adapter path without requiring it to exist."""
    configured = model_path
    if configured is None:
        configured = os.getenv(MODEL_PATH_ENV, "").strip() or DEFAULT_SPEECH_MODEL_PATH
    candidate = Path(configured).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def configured_base_model_path(
    base_model_path: str | Path | None = None,
    *,
    adapter_path: str | Path | None = None,
) -> Path:
    """Return the configured base path, preferring a base bundled with the adapter."""
    configured = base_model_path
    if configured is None:
        configured = os.getenv(BASE_MODEL_PATH_ENV, "").strip() or None
    if configured is not None:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    if adapter_path is not None:
        bundled_base = Path(adapter_path) / "base_model"
        if bundled_base.is_dir():
            return bundled_base
    return DEFAULT_BASE_MODEL_PATH


def _resolve_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelUnavailableError(f"{label} directory not found: {candidate}") from exc
    if not resolved.is_dir():
        raise ModelUnavailableError(f"{label} path is not a directory: {resolved}")
    return resolved


def resolve_adapter_path(model_path: str | Path | None = None) -> Path:
    """Resolve and validate the configured local iSpeak PEFT adapter directory."""
    resolved = _resolve_directory(
        configured_adapter_path(model_path),
        "iSpeak adapter",
    )
    missing = [name for name in _ADAPTER_FILES if not (resolved / name).is_file()]
    if missing:
        raise ModelUnavailableError(
            f"{resolved.name} adapter directory is incomplete: {resolved} "
            f"(missing: {', '.join(missing)})"
        )
    return resolved


def _has_base_weights(path: Path) -> bool:
    return any(
        (
            (path / "model.safetensors").is_file(),
            (path / "model.safetensors.index.json").is_file(),
            (path / "pytorch_model.bin").is_file(),
            (path / "pytorch_model.bin.index.json").is_file(),
            any(path.glob("model-*.safetensors")),
            any(path.glob("pytorch_model-*.bin")),
        )
    )


def resolve_base_model_path(
    base_model_path: str | Path | None = None,
    *,
    adapter_path: str | Path | None = None,
) -> Path:
    """Resolve the local Whisper base model required by the selected adapter."""
    resolved = _resolve_directory(
        configured_base_model_path(base_model_path, adapter_path=adapter_path),
        "Whisper base model",
    )
    if not (resolved / "config.json").is_file() or not _has_base_weights(resolved):
        raise ModelUnavailableError(
            f"Whisper base model directory is incomplete: {resolved} "
            "(config.json and local PyTorch/Safetensors weights are required)"
        )
    return resolved


def declared_base_model_id(model_path: str | Path | None = None) -> str:
    """Read the base-model identifier declared by the adapter metadata."""
    adapter_path = resolve_adapter_path(model_path)
    try:
        config = json.loads(
            (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(
            f"Could not read {adapter_path.name} adapter metadata: {exc}"
        ) from exc
    base_model_id = config.get("base_model_name_or_path")
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ModelUnavailableError(
            f"{adapter_path.name} adapter_config.json does not declare "
            "base_model_name_or_path"
        )
    return base_model_id.strip()


def resolve_model_paths(
    model_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
) -> tuple[Path, Path]:
    adapter_path = resolve_adapter_path(model_path)
    base_path = resolve_base_model_path(base_model_path, adapter_path=adapter_path)
    declared_base_model_id(adapter_path)
    return adapter_path, base_path


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Backward-compatible alias that resolves the selected adapter path."""
    return resolve_adapter_path(model_path)


class ISpeakWhisper:
    """Whisper with the selected local iSpeak LoRA adapter merged for inference."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        base_model_path: str | Path | None = None,
    ):
        self.adapter_path, self.base_model_path = resolve_model_paths(
            model_path,
            base_model_path,
        )
        self.model_name = self.adapter_path.name
        self.base_model_id = declared_base_model_id(self.adapter_path)

        try:
            import torch
            from peft import PeftModel
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
                pipeline,
            )
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise ModelUnavailableError(
                f"Missing {self.model_name} runtime dependencies. "
                "Run setup_backend.ps1 first."
            ) from exc

        use_cuda = torch.cuda.is_available()
        device = 0 if use_cuda else -1
        device_label = "GPU (CUDA)" if use_cuda else "CPU"
        print(
            f"Loading {self.model_name} adapter from {self.adapter_path} "
            f"with base model {self.base_model_path} on {device_label}..."
        )

        try:
            # Transformers and PEFT expose these through dynamic factory APIs;
            # keep the boundary explicit so static analysis does not select an
            # unrelated pipeline/model overload.
            self.processor: Any = WhisperProcessor.from_pretrained(
                str(self.adapter_path),
                local_files_only=True,
            )
            base_model = WhisperForConditionalGeneration.from_pretrained(
                str(self.base_model_path),
                local_files_only=True,
            )
            adapted_model: Any = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
                is_trainable=False,
                local_files_only=True,
            )
            self.model: Any = adapted_model.merge_and_unload(safe_merge=True)
            self.model.eval()
            self.pipe: Any = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=device,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load local {self.model_name} model: {exc}"
            ) from exc

        print(f"{self.model_name} loaded and ready.")

    def transcribe(
        self,
        file_path: str,
        *,
        audio_data: Any | None = None,
        initial_prompt: str | None = None,
        language: str | None = None,
        condition_on_previous_text: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        pipeline_input: Any = audio_data if audio_data is not None else file_path
        generate_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "condition_on_prev_tokens": condition_on_previous_text,
        }
        if initial_prompt:
            prompt_ids = self.processor.get_prompt_ids(
                initial_prompt,
                return_tensors="pt",
            )
            generate_kwargs["prompt_ids"] = prompt_ids.to(self.model.device)
        language_codes = {"English": "en", "Filipino": "tl"}
        if language in language_codes:
            generate_kwargs["language"] = language_codes[language]

        output: Any = self.pipe(
            pipeline_input,
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
        )
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
                    "start": round(current_start, 3),
                    "end": round(current_start + word_duration, 3),
                    "probability": confidence,
                }
                formatted_words.append(word_result)
                segment_words.append(word_result)
                current_start += word_duration

            formatted_segments.append(
                {
                    "text": phrase,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "words": segment_words,
                }
            )

        total_end = formatted_words[-1]["end"] if formatted_words else 0.0
        return {
            "text": full_text,
            "segments": formatted_segments,
            "duration": total_end,
        }


# Preserve the generic historical import name used by local scripts.
ISpeakV5Whisper = ISpeakWhisper
OptimizedONNXWhisper = ISpeakWhisper


def load_model(
    model_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
) -> ISpeakWhisper:
    return ISpeakWhisper(model_path, base_model_path)
