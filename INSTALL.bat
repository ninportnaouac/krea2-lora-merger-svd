@echo off
title Install Krea2 LoRA Merger
cd /d "%~dp0"

echo.
echo Creating isolated Python environment...
python -m venv venv
if errorlevel 1 (
    echo.
    echo Could not create the virtual environment.
    echo Make sure Python 3.11 or newer is installed and available as "python".
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"

echo.
echo Updating pip...
python -m pip install --upgrade pip

echo.
echo Installing requirements...
python -m pip install -r requirements.txt

echo.
echo Installation complete.
echo You can now double-click RUN_KREA2_LORA_MERGER.bat
echo.
pause
