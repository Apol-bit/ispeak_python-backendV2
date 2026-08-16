$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Project .venv is missing. Run .\setup_backend.ps1 first.'
}

$Host_ = if ($env:ISPEAK_HOST) { $env:ISPEAK_HOST } else { '127.0.0.1' }
$Port_ = if ($env:ISPEAK_PORT) { $env:ISPEAK_PORT } else { '8000' }

& $Python -m uvicorn fastapi_backend:app --host $Host_ --port $Port_
