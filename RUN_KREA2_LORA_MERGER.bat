@echo off
title Krea2 LoRA Merger & SVD Compressor
cd /d "%~dp0"

echo.
echo ==========================================
echo   Krea2 LoRA Merger & SVD Compressor
echo ==========================================
echo.

if exist "venv\Scripts\python.exe" (
    set "PY=%CD%\venv\Scripts\python.exe"
) else if exist "int4_env\Scripts\python.exe" (
    set "PY=%CD%\int4_env\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" -c "import torch, safetensors; print('Python environment: OK'); print('Torch:', torch.__version__)"
if errorlevel 1 (
    echo.
    echo Missing dependencies.
    echo Run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

echo.
echo Starting application...
"%PY%" app.py

echo.
pause
