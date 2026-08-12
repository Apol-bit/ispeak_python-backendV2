"""Offline project readiness check for teammates."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from model import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_SPEECH_MODEL_PATH,
    ModelUnavailableError,
    declared_base_model_id,
    resolve_model_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_IMPORTS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "multipart": "python-multipart",
    "librosa": "librosa",
    "numpy": "numpy",
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
}


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local iSpeak backend readiness")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when anything is missing")
    args = parser.parse_args()

    missing_packages = [
        package for module, package in REQUIRED_IMPORTS.items()
        if not _module_available(module)
    ]
    try:
        adapter_path, base_path = resolve_model_paths()
        speech = {
            "available": True,
            "adapter_path": str(adapter_path),
            "base_path": str(base_path),
            "base_model_id": declared_base_model_id(adapter_path),
            "error": None,
        }
    except ModelUnavailableError as exc:
        speech = {
            "available": False,
            "adapter_path": str(DEFAULT_SPEECH_MODEL_PATH),
            "base_path": str(DEFAULT_BASE_MODEL_PATH),
            "base_model_id": None,
            "error": str(exc),
        }
    try:
        from audio_analysis_processing_files.filler_words import filler_classifier_status

        filler = filler_classifier_status()
    except (ImportError, ModuleNotFoundError) as exc:
        filler = {
            "available": False,
            "path": str(PROJECT_ROOT / "models" / "filler_classifier"),
            "error": f"Dependency unavailable: {exc}",
        }

    print("iSpeak backend readiness")
    print(f"  Project folder     : {PROJECT_ROOT}")
    print(f"  Python dependencies: {'ready' if not missing_packages else 'missing ' + ', '.join(missing_packages)}")
    print(f"  iSpeak_v4 model    : {'ready' if speech['available'] else 'missing/incomplete'}")
    print(f"    Adapter: {speech['adapter_path']}")
    print(f"    Base   : {speech['base_path']}")
    if speech["base_model_id"]:
        print(f"    Declared base model: {speech['base_model_id']}")
    if speech["error"]:
        print(f"    {speech['error']}")
    print(f"  Filler classifier  : {'ready' if filler['available'] else 'missing/incomplete (optional)'}")
    print(f"    {filler['path']}")
    print("  Runtime API/network: disabled for model inference")

    ready = not missing_packages and speech["available"]
    return 1 if args.strict and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
