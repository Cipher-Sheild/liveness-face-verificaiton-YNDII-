# PowerShell launcher for YIDIE Unified eKYC
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   YIDIE Full Biometric eKYC Platform" -ForegroundColor Green
Write-Host "   Active Liveness + Passive Anti-Spoof + AdaFace Face Verification" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$PythonCandidates = @(
    (Join-Path $ScriptDir "venv\Scripts\python.exe"),
    (Join-Path $ScriptDir ".venv\Scripts\python.exe"),
    "E:\eKYC\liveness\venv\Scripts\python.exe",
    "python"
)

$PythonExe = $null
foreach ($cand in $PythonCandidates) {
    if ($cand -eq "python") {
        $cmd = Get-Command "python" -ErrorAction SilentlyContinue
        if ($cmd) {
            $PythonExe = "python"
            break
        }
    } elseif (Test-Path $cand) {
        $PythonExe = $cand
        break
    }
}

if (-not $PythonExe) {
    Write-Host "[ERROR] Python not found. Please activate your virtual environment or install Python 3.10+." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[INFO] Using Python: $PythonExe" -ForegroundColor Cyan
Write-Host "[INFO] Verifying model weights..." -ForegroundColor Cyan
& $PythonExe download_models.py

Write-Host ""
Write-Host "Starting unified server on http://localhost:8000 ..." -ForegroundColor Green
Start-Process "http://localhost:8000"
& $PythonExe server.py
