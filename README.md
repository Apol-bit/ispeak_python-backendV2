# iSpeak Python Backend V2

This is the **only supported production speech-analysis service**. The sibling
`ispeak_python-backend` repository is legacy and must not be started in a
production environment.

FastAPI speech-analysis service using the local `iSpeak_v5` PEFT/LoRA adapter
on top of `openai/whisper-small`.

- Default adapter: `models/iSpeak_v5`
- Local base model: `models/iSpeak_v5/base_model`
- Optional overrides: `ISPEAK_MODEL_PATH` and `ISPEAK_BASE_MODEL_PATH`
- Setup: `powershell -ExecutionPolicy Bypass -File .\setup_backend.ps1 -DownloadBaseModel`
- Start: `.\start_backend.ps1`
- Readiness: `http://127.0.0.1:8000/health`
- Production: set `ISPEAK_ENV=production`, use an explicit
  `ISPEAK_CORS_ORIGINS`, bind to the private service interface, and configure
  concurrency for the available CPU/GPU memory. See `.env.example`.

Model inference is local-only at API runtime. See `SETUP.md` for details.
