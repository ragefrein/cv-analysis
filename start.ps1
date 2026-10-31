param([switch]$Dev, [int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error 'Virtual environment belum tersedia. Jalankan .\setup.ps1 terlebih dahulu.'
}
if ($Dev) {
    Write-Host 'API: http://localhost:8000. Jalankan npm run dev --prefix frontend di terminal lain.'
} elseif (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\dist\index.html'))) {
    npm.cmd run build --prefix frontend
    if ($LASTEXITCODE -ne 0) { throw 'Build frontend gagal.' }
}
Write-Host "Futsal Vision: http://localhost:$Port"
& $pythonPath -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
