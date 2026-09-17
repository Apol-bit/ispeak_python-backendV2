"""Explicitly download the selected iSpeak adapter's base model for offline use."""

from __future__ import annotations

from model import (
    ModelUnavailableError,
    configured_base_model_path,
    declared_base_model_id,
    resolve_adapter_path,
    resolve_base_model_path,
)


def main() -> int:
    adapter_path = resolve_adapter_path()
    base_model_path = configured_base_model_path(adapter_path=adapter_path)
    model_id = declared_base_model_id(adapter_path)

    try:
        existing = resolve_base_model_path(base_model_path)
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

    base_model_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} to {base_model_path}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=base_model_path,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ],
    )
    resolved = resolve_base_model_path(base_model_path)
    print(f"Base model ready: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
