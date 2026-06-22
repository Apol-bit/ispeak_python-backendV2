# Team setup

Install Python 3.13 and Git LFS first. After cloning, retrieve any model weights
that the team has committed through LFS:

```powershell
git lfs install
git lfs pull
```

Then, from PowerShell in this repository:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_backend.ps1
```

Copy the local models into the folders documented under `models/`, then start:

```powershell
.\start_backend.ps1
```

Check `http://127.0.0.1:8000/health`. The server starts in degraded mode when
model weights are missing; `/transcribe` returns HTTP 503 with the exact reason
instead of crashing. No model inference API or runtime model download is used.
