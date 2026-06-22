"""Offline project readiness check for teammates."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from audio_analysis_processing_files.filler_words import filler_classifier_status
from model import DEFAULT_SPEECH_MODEL_PATH, ModelUnavailableError, resolve_model_path

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_IMPORTS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "multipart": "python-multipart",
    "librosa": "librosa",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "transformers": "transformers",
    "optimum.onnxruntime": "optimum-onnx",
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
        speech_path = resolve_model_path(DEFAULT_SPEECH_MODEL_PATH)
        speech = {"available": True, "path": str(speech_path), "error": None}
    except ModelUnavailableError as exc:
        speech = {
            "available": False,
            "path": str(DEFAULT_SPEECH_MODEL_PATH),
            "error": str(exc),
        }
    filler = filler_classifier_status()

    print("iSpeak backend readiness")
    print(f"  Project folder     : {PROJECT_ROOT}")
    print(f"  Python dependencies: {'ready' if not missing_packages else 'missing ' + ', '.join(missing_packages)}")
    print(f"  Speech model       : {'ready' if speech['available'] else 'missing/incomplete'}")
    print(f"    {speech['path']}")
    if speech["error"]:
        print(f"    {speech['error']}")
    print(f"  Filler classifier  : {'ready' if filler['available'] else 'missing/incomplete (optional)'}")
    print(f"    {filler['path']}")
    print("  Runtime API/network: disabled for model inference")

    ready = not missing_packages and speech["available"]
    return 1 if args.strict and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
