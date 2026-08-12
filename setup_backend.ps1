param(
    [switch]$SkipInstall,
    [switch]$DownloadBaseModel
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$PortableRequirements = Join-Path $env:TEMP 'ispeak-portable-requirements.txt'

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host 'Creating project-local .venv...'
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & $PyLauncher.Source -3.13 -m venv (Join-Path $ProjectRoot '.venv')
    }
    else {
        $SystemPython = Get-Command python -ErrorAction SilentlyContinue
        if (-not $SystemPython) {
            throw 'Python 3.13 was not found. Install Python before running setup.'
        }
        & $SystemPython.Source -m venv (Join-Path $ProjectRoot '.venv')
    }
}

if (-not $SkipInstall) {
    Write-Host 'Installing dependencies into project-local .venv...'
    Get-Content -LiteralPath (Join-Path $ProjectRoot 'requirements.txt') |
        Where-Object { $_ -notmatch '^onnxruntime-gpu==' } |
        Set-Content -LiteralPath $PortableRequirements -Encoding utf8
    try {
        & $Python -m pip install --upgrade pip
        & $Python -m pip install -r $PortableRequirements
    }
    finally {
        Remove-Item -LiteralPath $PortableRequirements -Force -ErrorAction SilentlyContinue
    }
}

if ($DownloadBaseModel) {
    Write-Host 'Downloading the openai/whisper-small base model for iSpeak_v4...'
    & $Python (Join-Path $ProjectRoot 'download_base_model.py')
}

& $Python (Join-Path $ProjectRoot 'check_setup.py')
Write-Host ''
Write-Host 'Setup finished. Missing model weights are reported above.'
Write-Host 'Start the backend with: .\start_backend.ps1'
