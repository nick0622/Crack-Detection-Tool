@echo off
chcp 65001 >nul
title Crack Detection Setup

echo ================================================================
echo                   ⚙️ Crack Detection Tool Setup ⚙️
echo ================================================================
echo.

:: Check Python
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not installed or not in PATH
    echo 💡 Please install Python 3.7+ from https://python.org
    echo    Make sure to check "Add Python to PATH" during installation
    pause
    exit
) else (
    echo ✅ Python found:
    python --version
)

echo.
echo 📦 Installing required packages...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo Installing core packages manually...
    pip install numpy>=1.21.0 opencv-python>=4.5.0 Pillow>=8.0.0 onnxruntime>=1.12.0
)

if errorlevel 1 (
    echo ❌ Package installation failed
    echo 💡 Try running as administrator or check your internet connection
    pause
    exit
)

echo ✅ Packages installed successfully!

echo.
echo 🔍 Testing package imports...
python -c "import cv2, numpy, onnxruntime, PIL" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Some packages may not be properly installed
    echo    You can still try running the tool, but you may encounter errors
) else (
    echo ✅ All core packages can be imported successfully
)

echo.
echo 📁 Creating necessary folders...
if not exist "model" mkdir model
if not exist "images" mkdir images
if not exist "results" mkdir results
if not exist "enhanced_images" mkdir enhanced_images
echo ✅ Folders created

echo.
echo ================================================================
echo                    ✅ Setup Complete!
echo ================================================================
echo.
echo 🚀 You can now use:
echo    • run_preprocessing.bat  - Preprocess the images
echo    • run_crack_detector.bat - Full featured interface
echo.
pause