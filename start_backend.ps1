$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Project .venv is missing. Run .\setup_backend.ps1 first.'
}

& $Python -m uvicorn fastapi_backend:app --host 127.0.0.1 --port 8000
