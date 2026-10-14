"""FastAPI entry point with project-local, failure-safe model startup."""

from __future__ import annotations

import logging
import hmac
import os
import time
import tempfile
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from audio_analysis_processing_files.filler_words import filler_classifier_status, _default_classifier
from analysis_admission import AnalysisAdmission, AdmissionFull, AdmissionTimeout, finish_thread_on_cancel
from model import DEFAULT_MODEL_NAME, ModelUnavailableError, load_model
from whisper_service import generate_full_analysis, generate_reference_analysis
from audio_validation import SpeechValidationError

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
MAX_UPLOAD_MB = int(os.getenv("ISPEAK_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Environment-based configuration
HOST = os.getenv("ISPEAK_HOST", "127.0.0.1")
PORT = int(os.getenv("PORT") or os.getenv("ISPEAK_PORT", "8000"))
MAX_CONCURRENT_INFERENCES = max(1, int(os.getenv("ISPEAK_MAX_CONCURRENT_INFERENCES", "1")))
QUEUE_TIMEOUT_SECONDS = max(1, int(os.getenv("ISPEAK_QUEUE_TIMEOUT_SECONDS", "30")))
IS_PRODUCTION = os.getenv("ISPEAK_ENV", "development").lower() == "production"
SERVICE_TOKEN = os.getenv("ISPEAK_SERVICE_TOKEN", "")
if IS_PRODUCTION and len(SERVICE_TOKEN) < 32:
    raise RuntimeError("ISPEAK_SERVICE_TOKEN must contain at least 32 characters in production")

# CORS: comma-separated origins, or "*" for development
_cors_env = os.getenv("ISPEAK_CORS_ORIGINS", "*").strip()
CORS_ORIGINS: list[str] = (
    ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]
)
if IS_PRODUCTION and "*" in CORS_ORIGINS:
    raise RuntimeError("ISPEAK_CORS_ORIGINS must be explicit in production")
if IS_PRODUCTION:
    for origin in CORS_ORIGINS:
        parsed = urlsplit(origin)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise RuntimeError("Production ISPEAK_CORS_ORIGINS must contain exact HTTPS origins or be empty")

MAX_QUEUED_ANALYSES = max(0, int(os.getenv("ISPEAK_MAX_QUEUED_ANALYSES", "4")))
admission = AnalysisAdmission(MAX_CONCURRENT_INFERENCES, MAX_QUEUED_ANALYSES, QUEUE_TIMEOUT_SECONDS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _log_failure(event: str, exc: Exception) -> None:
    # Tracebacks may include speech, filenames and model configuration.
    if not IS_PRODUCTION and os.getenv("ISPEAK_DEBUG_ERRORS") == "true":
        logger.exception(event)
    else:
        logger.error("%s category=%s", event, type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting iSpeak backend")
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = await run_in_threadpool(load_model)
    except ModelUnavailableError as exc:
        app.state.model_error = str(exc)
        _log_failure("speech_model_unavailable", exc)
    except Exception as exc:  # Keep startup alive and expose the failure in /health.
        app.state.model_error = f"Unexpected model-loading failure: {exc}"
        _log_failure("model_startup_failed", exc)
    app.state.filler_ready = False
    app.state.filler_error = None
    try:
        # Exercise the existing classifier; no scoring or classification changes.
        await run_in_threadpool(lambda: _default_classifier()("Readiness check."))
        app.state.filler_ready = True
    except Exception as exc:
        app.state.filler_error = str(exc)
        _log_failure("filler_model_unavailable", exc)
    yield


app = FastAPI(title="iSpeak Speech Analysis Backend", lifespan=lifespan,
              docs_url=None if IS_PRODUCTION else "/docs",
              redoc_url=None if IS_PRODUCTION else "/redoc",
              openapi_url=None if IS_PRODUCTION else "/openapi.json")
app.state.model = None
app.state.model_error = "Application lifespan has not started"
app.state.filler_ready = False
app.state.filler_error = "Application lifespan has not started"


@app.middleware("http")
async def service_authentication(request: Request, call_next):
    # Minimal probes remain available to the host; analysis/metrics require the
    # private service credential before parsing any upload or reserving work.
    if SERVICE_TOKEN and request.url.path not in {"/live", "/ready", "/health"}:
        supplied = request.headers.get("X-iSpeak-Service-Token", "")
        if not hmac.compare_digest(supplied.encode(), SERVICE_TOKEN.encode()):
            return JSONResponse(status_code=401, content={"detail": "Service authentication required"})
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ORIGINS != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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


def _validate_media_signature(file_path: str, suffix: str) -> None:
    with open(file_path, "rb") as media:
        header = media.read(16)
    valid = {
        ".wav": header.startswith(b"RIFF") and header[8:12] == b"WAVE",
        ".flac": header.startswith(b"fLaC"),
        ".ogg": header.startswith(b"OggS"),
        ".mp3": header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        ),
        ".m4a": len(header) >= 8 and header[4:8] == b"ftyp",
        ".aac": len(header) >= 2 and header[0] == 0xFF and header[1] & 0xF0 == 0xF0,
    }.get(suffix, False)
    if not valid:
        raise HTTPException(status_code=400, detail="File content does not match its audio extension")


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


def _save_upload(upload, suffix):
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            path = temp.name
            _copy_with_limit(upload.file, temp)
        _validate_media_signature(path, suffix)
        return path
    except BaseException:
        if path and os.path.exists(path):
            os.remove(path)
        raise


@app.get("/live")
async def live():
    return {"status": "alive"}


@app.get("/metrics")
async def metrics():
    # Private local service: process counters only; no recordings or transcripts.
    return admission.snapshot()


@app.get("/ready")
@app.get("/health")
async def health():
    speech_ready = app.state.model is not None
    model = app.state.model
    filler_status = filler_classifier_status()
    filler_status["files_available"] = filler_status["available"]
    filler_status["available"] = app.state.filler_ready
    filler_status["error"] = app.state.filler_error
    ready = speech_ready and app.state.filler_ready
    if IS_PRODUCTION:
        return JSONResponse(status_code=200 if ready else 503, content={
            "status": "ready" if ready else "degraded",
            "speech_model": {"available": speech_ready},
            "filler_classifier": {"available": app.state.filler_ready},
        })
    return JSONResponse(status_code=200 if ready else 503, content={
        "status": "ready" if ready else "degraded",
        "speech_model": {
            "available": speech_ready,
            "error": app.state.model_error,
            "name": getattr(model, "model_name", DEFAULT_MODEL_NAME),
            "adapter_path": str(getattr(model, "adapter_path", "")) or None,
            "base_model_path": str(getattr(model, "base_model_path", "")) or None,
        },
        "filler_classifier": filler_status,
        "admission": admission.snapshot(),
    })


@app.post("/transcribe")
async def transcribe(request: Request, response: Response):
    try:
        async with admission.reserve():
            return await _transcribe(request, response)
    except (AdmissionFull, AdmissionTimeout) as exc:
        raise HTTPException(
            status_code=503,
            detail="Speech analysis is busy. Please try again shortly.",
            headers={"Retry-After": "5"},
        ) from exc
    finally:
        # Starlette owns multipart spool files, independently of our copies.
        await request.close()


async def _transcribe(request: Request, response: Response):
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Local speech model is unavailable",
                "reason": "Speech model unavailable" if IS_PRODUCTION else app.state.model_error,
            },
        )

    try:
        form = await request.form(max_files=2, max_fields=4)
    except (AssertionError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Multipart support is missing. Run setup_backend.ps1.",
        ) from exc

    file = _require_upload(form.get("file"), "file")
    reference_audio = form.get("reference_audio")
    language = str(form.get("language") or "English")
    if language not in {"English", "Filipino", "Taglish"}:
        raise HTTPException(status_code=400, detail="Unsupported language")
    if reference_audio is not None:
        reference_audio = _require_upload(reference_audio, "reference_audio")

    suffix = _validate_extension(file.filename)
    ref_suffix = _validate_extension(reference_audio.filename) if reference_audio else None
    temp_path = None
    ref_temp_path = None

    try:
        temp_path = _save_upload(file, suffix)
        if reference_audio is not None:
            ref_temp_path = _save_upload(reference_audio, ref_suffix)

        async with admission.worker() as queue_wait_ms:
            if await request.is_disconnected():
                raise HTTPException(status_code=499, detail="Client disconnected")
            started = time.perf_counter()
            if ref_temp_path:
                work = run_in_threadpool(
                    generate_reference_analysis, temp_path, ref_temp_path,
                    app.state.model, language,
                )
            else:
                work = run_in_threadpool(
                    generate_full_analysis, temp_path, app.state.model, language,
                )
            result = await finish_thread_on_cancel(work)
            response.headers["X-Analysis-Queue-Wait-Ms"] = f"{queue_wait_ms:.3f}"
            response.headers["X-Analysis-Processing-Ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
            admission.completed += 1
            app.state.filler_ready = result.get("fillers", {}).get("analysis_available") is True
            app.state.filler_error = None if app.state.filler_ready else "Filler analysis unavailable on last request"
            return result
    except (AdmissionFull, AdmissionTimeout):
        raise
    except SpeechValidationError as exc:
        admission.failed += 1
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        admission.failed += 1
        _log_failure("audio_processing_failed", exc)
        raise HTTPException(
            status_code=500,
            detail="Audio processing failed",
        ) from exc
    finally:
        for path in (temp_path, ref_temp_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    _log_failure("temporary_audio_cleanup_failed", exc)
