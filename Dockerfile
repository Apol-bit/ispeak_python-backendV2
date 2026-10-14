FROM python:3.13.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/tmp/ispeak \
    NUMBA_CACHE_DIR=/tmp/ispeak/numba \
    HF_HOME=/tmp/ispeak/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ISPEAK_HOST=0.0.0.0 \
    ISPEAK_MAX_CONCURRENT_INFERENCES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 ispeak \
    && useradd --uid 10001 --gid ispeak --no-create-home ispeak \
    && mkdir -p /tmp/ispeak/numba /tmp/ispeak/huggingface \
    && chown -R ispeak:ispeak /tmp/ispeak

WORKDIR /app
COPY requirements.txt requirements-linux.txt ./
# Install the same Torch release from the CPU wheel index before resolving PEFT.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && python -m pip install -r requirements-linux.txt \
    && python -m pip check

COPY fastapi_backend.py model.py whisper_service.py audio_validation.py analysis_admission.py ./
COPY audio_analysis_processing_files/ ./audio_analysis_processing_files/
COPY models/iSpeak_v5/ ./models/iSpeak_v5/
COPY models/filler_classifier/ ./models/filler_classifier/
# Private Node-to-Python requests need no browser CORS origins by default.
ENV ISPEAK_ENV=production ISPEAK_CORS_ORIGINS=""
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+str(os.getenv('PORT') or os.getenv('ISPEAK_PORT','8000'))+'/health',timeout=4)"
CMD ["sh", "-c", "exec python -m uvicorn fastapi_backend:app --host 0.0.0.0 --port ${PORT:-${ISPEAK_PORT:-8000}} --workers 1"]
