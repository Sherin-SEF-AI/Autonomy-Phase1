@echo off
REM Automated setup script for AV Perception System v1.1.0 (Windows)
REM This script sets up the environment and dependencies for the ADAS platform

setlocal enabledelayedexpansion

echo ========================================
echo AV Perception System Setup - v1.1.0
echo ========================================
echo.

REM Check Python version
echo [*] Checking Python version...
python --version > nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed or not in PATH
    echo [!] Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [+] Python version: %PYTHON_VERSION%

REM Check if virtual environment exists
echo [*] Checking for virtual environment...
if not exist "venv\" (
    echo [!] Virtual environment not found. Creating...
    python -m venv venv
    echo [+] Virtual environment created
) else (
    echo [+] Virtual environment found
)

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
echo [+] Virtual environment activated

REM Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [+] pip upgraded

REM Install dependencies
echo [*] Installing dependencies from requirements.txt...
if exist "requirements.txt" (
    pip install -r requirements.txt --quiet
    echo [+] Dependencies installed
) else (
    echo [X] requirements.txt not found!
    pause
    exit /b 1
)

REM Create necessary directories
echo [*] Creating data directories...
if not exist "data\recordings" mkdir data\recordings
if not exist "data\recordings\events" mkdir data\recordings\events
if not exist "data\logs" mkdir data\logs
if not exist "data\exports" mkdir data\exports
if not exist "data\analytics" mkdir data\analytics
if not exist "config\camera_profiles" mkdir config\camera_profiles
echo [+] Directories created

REM Test imports
echo [*] Testing critical imports...
python -c "import cv2; import PyQt6; import numpy; print('[+] All critical imports successful')"
if errorlevel 1 (
    echo [X] Import test failed
    pause
    exit /b 1
)

REM Download YOLOv8 model (optional)
echo [*] Checking for YOLOv8 model...
if not exist "yolov8n.pt" (
    echo [!] YOLOv8 model not found. It will be downloaded on first run.
    set /p DOWNLOAD="Download YOLOv8n model now? (y/n): "
    if /i "!DOWNLOAD!"=="y" (
        python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('[+] YOLOv8n model downloaded')"
    )
) else (
    echo [+] YOLOv8 model found
)

REM Display system information
echo.
echo ========================================
echo System Information
echo ========================================
echo.
python --version
for /f "tokens=*" %%i in ('python -c "import cv2; print('OpenCV:', cv2.__version__)"') do echo %%i
for /f "tokens=*" %%i in ('python -c "import numpy; print('NumPy:', numpy.__version__)"') do echo %%i

REM Setup complete
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo [+] The AV Perception System is ready to run!
echo.
echo To start the application:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Run the application: python main.py
echo.
echo For more information, see:
echo   - README.md - General documentation
echo   - QUICKSTART.md - Quick start guide
echo   - docs\ADVANCED_FEATURES.md - Advanced features documentation
echo.
echo To run tests:
echo   pytest tests\
echo.
echo To run advanced features demo:
echo   python examples\advanced_features_demo.py
echo.
echo [+] Setup completed successfully!
echo.
pause
