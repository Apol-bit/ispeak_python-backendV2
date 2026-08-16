"""FastAPI entry point with project-local, failure-safe model startup."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from audio_analysis_processing_files.filler_words import filler_classifier_status
from model import ModelUnavailableError, load_model
from whisper_service import generate_full_analysis, generate_reference_analysis

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
MAX_UPLOAD_MB = int(os.getenv("ISPEAK_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Environment-based configuration
HOST = os.getenv("ISPEAK_HOST", "127.0.0.1")
PORT = int(os.getenv("ISPEAK_PORT", "8000"))

# CORS: comma-separated origins, or "*" for development
_cors_env = os.getenv("ISPEAK_CORS_ORIGINS", "*").strip()
CORS_ORIGINS: list[str] = (
    ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting iSpeak backend")
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = load_model()
    except ModelUnavailableError as exc:
        app.state.model_error = str(exc)
        logger.warning("Speech analysis unavailable: %s", exc)
    except Exception as exc:  # Keep startup alive and expose the failure in /health.
        app.state.model_error = f"Unexpected model-loading failure: {exc}"
        logger.exception("Unexpected model-loading failure")
    yield


app = FastAPI(title="iSpeak Speech Analysis Backend", lifespan=lifespan)
app.state.model = None
app.state.model_error = "Application lifespan has not started"

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_extension(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    suffix = os.path.splitext(filename)[1].lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="File must have an extension")
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return suffix


def _require_upload(value: Any, field_name: str) -> Any:
    if value is None or not hasattr(value, "filename") or not hasattr(value, "file"):
        raise HTTPException(
            status_code=400,
            detail=f"Multipart field '{field_name}' must contain an uploaded audio file",
        )
    return value


def _copy_with_limit(source, dest, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Copy from source to dest in chunks, raising 413 if max_bytes is exceeded."""
    total = 0
    while True:
        chunk = source.read(64 * 1024)  # 64KB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB}MB.",
            )
        dest.write(chunk)
    return total


@app.get("/health")
async def health() -> dict[str, Any]:
    speech_ready = app.state.model is not None
    model = app.state.model
    filler_status = filler_classifier_status()
    return {
        "status": "ready" if speech_ready else "degraded",
        "speech_model": {
            "available": speech_ready,
            "error": app.state.model_error,
            "name": getattr(model, "model_name", "iSpeak_v4"),
            "adapter_path": str(getattr(model, "adapter_path", "")) or None,
            "base_model_path": str(getattr(model, "base_model_path", "")) or None,
        },
        "filler_classifier": filler_status,
    }


@app.post("/transcribe")
async def transcribe(request: Request):
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Local speech model is unavailable",
                "reason": app.state.model_error,
            },
        )

    try:
        form = await request.form()
    except (AssertionError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Multipart support is missing. Run setup_backend.ps1.",
        ) from exc

    file = _require_upload(form.get("file"), "file")
    reference_audio = form.get("reference_audio")
    if reference_audio is not None:
        reference_audio = _require_upload(reference_audio, "reference_audio")

    suffix = _validate_extension(file.filename)
    ref_suffix = _validate_extension(reference_audio.filename) if reference_audio else None
    temp_path = None
    ref_temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            _copy_with_limit(file.file, temp)

        if reference_audio is not None:
            with tempfile.NamedTemporaryFile(suffix=ref_suffix, delete=False) as ref_temp:
                ref_temp_path = ref_temp.name
                _copy_with_limit(reference_audio.file, ref_temp)

        if ref_temp_path:
            result = await run_in_threadpool(
                generate_reference_analysis,
                temp_path,
                ref_temp_path,
                app.state.model,
            )
        else:
            result = await run_in_threadpool(
                generate_full_analysis,
                temp_path,
                app.state.model,
            )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Processing failed for %s:\n%s", file.filename, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {exc}",
        ) from exc
    finally:
        for path in (temp_path, ref_temp_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Failed to remove temp file %s: %s", path, exc)
