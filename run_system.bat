@echo off
title YIDIE Full Biometric eKYC Platform
cd /d "%~dp0"

echo ======================================================================
echo    YIDIE Full Biometric eKYC Platform
echo    Active Liveness + Passive Anti-Spoof + AdaFace Face Verification
echo ======================================================================
echo.

:: Detect Python executable (local venv -> parent/common venv -> system PATH)
set "PYTHON_EXE="
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "E:\eKYC\liveness\venv\Scripts\python.exe" (
    set "PYTHON_EXE=E:\eKYC\liveness\venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python not found. Please activate your virtual environment or install Python 3.10+.
    pause
    exit /b 1
)

echo [INFO] Using Python: %PYTHON_EXE%

:: Verify models before launching
echo [INFO] Verifying model weights...
"%PYTHON_EXE%" download_models.py

echo.
echo Starting unified server on http://localhost:8000 ...
start "" http://localhost:8000
"%PYTHON_EXE%" server.py
pause
