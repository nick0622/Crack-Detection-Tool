@echo off
REM This batch file is a wrapper for the image preprocessing tool.
REM Usage: Drag and drop an image file or folder onto this .bat file, or double-click to enter a path.

REM Set console to UTF-8 to handle file paths with non-English characters.
chcp 65001 >nul
title Image Preprocessing Tool

setlocal EnableDelayedExpansion

REM Set the working directory to the location of the batch file to ensure the Python script is found.
cd /d "%~dp0"

REM Check if a file or folder was dragged and dropped onto the script.
if not "%~1"=="" (
    set "image_path=%~1"
) else (
    REM If not, prompt the user to manually enter a path.
    echo.
    echo Please drag and drop an image file or folder into this window, or type the full path:
    echo.
    set /p "image_path=> "
)

REM Check if the user provided an empty path.
if "%image_path%"=="" (
    echo.
    echo Error: No path was provided.
    echo.
    pause
    exit /b 1
)

REM Find the Python executable.
where /q python.exe
if %errorlevel% neq 0 (
    echo.
    echo Error: Python was not found in your system's PATH.
    echo Please ensure Python is installed and added to the PATH.
    echo.
    pause
    exit /b 1
)

REM Execute the Python script for image preprocessing.
python.exe preprocess.py "%image_path%"

echo.
echo Preprocessing complete. The enhanced images are in the 'enhanced_images' folder.
echo.
pause