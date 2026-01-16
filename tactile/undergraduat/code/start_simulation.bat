@echo off
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Installing/Updating dependencies...
.venv\Scripts\python -m pip install -r requirements.txt

echo.
echo Starting Simulator...
.venv\Scripts\python simulator.py

if %errorlevel% neq 0 (
    echo.
    echo Simulation exited with error code %errorlevel%
    pause
) else (
    echo.
    echo Simulation finished successfully.
    pause
)
