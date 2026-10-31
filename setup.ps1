param([switch]$UseSystemPackages)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($UseSystemPackages) { python -m venv --system-site-packages .venv }
    else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Pembuatan virtual environment gagal.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Instalasi Python gagal.' }
& '.\.venv\Scripts\python.exe' scripts/setup_model.py
if ($LASTEXITCODE -ne 0) { throw 'Unduhan model gagal.' }
npm.cmd install --prefix frontend
if ($LASTEXITCODE -ne 0) { throw 'Instalasi frontend gagal.' }
npm.cmd run build --prefix frontend
if ($LASTEXITCODE -ne 0) { throw 'Build frontend gagal.' }
Write-Host 'Setup selesai. Jalankan .\start.ps1 lalu buka http://localhost:8000'
