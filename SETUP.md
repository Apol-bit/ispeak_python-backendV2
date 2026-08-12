# Team setup

Install Python 3.13. The `models/iSpeak_v4` directory contains the local PEFT
adapter. Its declared base model is `openai/whisper-small` and must be stored at
`models/iSpeak_v4/base_model` for offline runtime use.

From PowerShell in this repository, create the environment, install dependencies,
and explicitly download the base model once:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_backend.ps1 -DownloadBaseModel
```

Later setup runs can omit `-DownloadBaseModel`. Backend startup never downloads
model files. Start it with:

```powershell
.\start_backend.ps1
```

Check `http://127.0.0.1:8000/health`. The server starts in degraded mode when
model weights are missing; `/transcribe` returns HTTP 503 with the exact reason
instead of crashing. No model inference API or runtime model download is used.
