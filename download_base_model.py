"""Explicitly download the iSpeak_v4 Whisper base model for offline runtime use."""

from __future__ import annotations

from model import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_SPEECH_MODEL_PATH,
    ModelUnavailableError,
    declared_base_model_id,
    resolve_base_model_path,
)


def main() -> int:
    model_id = declared_base_model_id(DEFAULT_SPEECH_MODEL_PATH)

    try:
        existing = resolve_base_model_path(DEFAULT_BASE_MODEL_PATH)
        print(f"Base model is already available: {existing}")
        return 0
    except ModelUnavailableError:
        pass

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is unavailable. Run setup_backend.ps1 first."
        ) from exc

    DEFAULT_BASE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} to {DEFAULT_BASE_MODEL_PATH}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=DEFAULT_BASE_MODEL_PATH,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ],
    )
    resolved = resolve_base_model_path(DEFAULT_BASE_MODEL_PATH)
    print(f"Base model ready: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
