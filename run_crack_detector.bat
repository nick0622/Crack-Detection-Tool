@echo off
chcp 65001 >nul
title Crack Detection Tool - Run

:: Enable delayed expansion for conditional variables
setlocal EnableDelayedExpansion

:: Change directory to the location of this script to ensure all relative paths work
cd /d "%~dp0"

:: Default settings
set "CONFIDENCE=0.2"
set "ENHANCE_FLAG=--enhance"
set "SAVE_FLAG=--save"
set "TTA_FLAG="
set "TILING_FLAG="
set "ENHANCED=Enabled"
set "TTA=Disabled"
set "TILING=Disabled"
set "MODEL_TYPE=yolov8_single"
set "MODEL_NAME=YOLOv8 Single Class"
set "INFERENCE_SCRIPT=inference_yolo.py"
set "CLASS_CONF_FLAG="
set "USE_CLASS_CONF=No"
set "CONF_0=0.2"
set "CONF_1=0.2"
set "CONF_2=0.2"
set "CONF_3=0.2"
set "MULTI_MODEL_FILE=yolov8_multi.onnx"
set "MULTI_MODEL_FLAG="

:MODEL_SELECTION
cls
echo ================================================================
echo                   🤖 Crack Detection Tool - Model Selection
echo ================================================================
echo.
echo Please select a model:
echo.
echo 1. 🎯 YOLOv8 Single Class
echo 2. 🎯 YOLOv8 4 Classes
echo 3. 🎯 Faster R-CNN
echo.
echo Current Model: %MODEL_NAME%
echo ----------------------------------------------------------------
set /p model_choice="Enter your choice (1-3): "

if "%model_choice%"=="1" (
    set "MODEL_TYPE=yolov8_single"
    set "MODEL_NAME=YOLOv8 Single Class"
    set "INFERENCE_SCRIPT=inference_yolo.py"
    set "TILING_FLAG="
    set "TILING=Disabled"
    set "CLASS_CONF_FLAG="
    set "USE_CLASS_CONF=No"
    set "CONF_0=0.2"
    set "CONF_1=0.2"
    set "CONF_2=0.2"
    set "CONF_3=0.2"
    set "MULTI_MODEL_FILE=yolov8_multi.onnx"
    set "MULTI_MODEL_FLAG="
    goto MENU
)
if "%model_choice%"=="2" (
    goto YOLO4_SUBMODEL
)
if "%model_choice%"=="3" (
    set "MODEL_TYPE=faster_rcnn"
    set "MODEL_NAME=Faster R-CNN"
    set "INFERENCE_SCRIPT=inference_frcnn.py"
    set "TILING_FLAG="
    set "TILING=Disabled"
    set "MULTI_MODEL_FLAG="
    goto MENU
)

echo.
echo ❌ Invalid choice. Please try again.
pause
goto MODEL_SELECTION

:YOLO4_SUBMODEL
cls
echo ================================================================
echo                   🎯 YOLOv8 4 Classes - Sub-Model Selection
echo ================================================================
echo.
echo Please select a sub-model:
echo.
echo 1. Model 7
echo 2. Model 10 (High res training)(not updated yet)
echo 3. Model 11 (Low res training)
echo.
echo B. ↩ Back to Model Selection
echo ----------------------------------------------------------------
set /p sub_choice="Enter your choice (1/2/3/B): "

if /i "%sub_choice%"=="B" goto MODEL_SELECTION

if "%sub_choice%"=="1" (
    set "MODEL_TYPE=yolov8_4classes"
    set "MULTI_MODEL_FILE=yolov8_multi.onnx"
    set "MODEL_NAME=YOLOv8 4 Classes (Standard)"
    set "INFERENCE_SCRIPT=inference_yolo.py"
    set "MULTI_MODEL_FLAG=--multi-model-file yolov8_multi.onnx"
    goto YOLO4_DEFAULTS
)
if "%sub_choice%"=="2" (
    set "MODEL_TYPE=yolov8_4classes"
    set "MULTI_MODEL_FILE=yolov8_multi_highres.onnx"
    set "MODEL_NAME=YOLOv8 4 Classes (High Res)"
    set "INFERENCE_SCRIPT=inference_yolo.py"
    set "MULTI_MODEL_FLAG=--multi-model-file yolov8_multi_highres.onnx"
    goto YOLO4_DEFAULTS
)
if "%sub_choice%"=="3" (
    set "MODEL_TYPE=yolov8_4classes"
    set "MULTI_MODEL_FILE=yolov8_multi_lowres.onnx"
    set "MODEL_NAME=YOLOv8 4 Classes (Low Res)"
    set "INFERENCE_SCRIPT=inference_yolo.py"
    set "MULTI_MODEL_FLAG=--multi-model-file yolov8_multi_lowres.onnx"
    goto YOLO4_DEFAULTS
)

echo.
echo ❌ Invalid choice. Please try again.
pause
goto YOLO4_SUBMODEL

:YOLO4_DEFAULTS
:: 自動啟用 per-class confidence 並設定預設值
set "CONF_0=0.45"
set "CONF_1=0.3"
set "CONF_2=0.2"
set "CONF_3=0.4"
set "CLASS_CONF_FLAG=--class-confidences 0:0.45,1:0.3,2:0.2,3:0.4"
set "USE_CLASS_CONF=Yes"
goto MENU

:MENU
cls
echo ================================================================
echo                   ⚙️ Crack Detection Tool Interface ⚙️
echo ================================================================
echo.
echo 🤖 Current Model: %MODEL_NAME%
echo ⚙️ Current Settings:
if /i "%USE_CLASS_CONF%"=="Yes" (
    if "%MODEL_TYPE%"=="yolov8_single" (
        echo    • Per-Class Confidence:
        echo       • Class 0 - Crack: 0.2
    ) else (
        echo   • Per-Class Confidence:
        echo      • Transverse:    %CONF_0%
        echo      • Longitudinal:  %CONF_1%
        echo      • Joint:         %CONF_2%
        echo      • Alligator:     %CONF_3%
    )
) else (
    echo   • Per-Class Confidence:  No
    echo   • Global Confidence:     %CONFIDENCE%
)
echo   • Image Enhancement:     %ENHANCED%
echo   • Test Time Aug (TTA):   %TTA%
echo   • 4-Tile Slicing:        %TILING%
echo.
echo ----------------------------------------------------------------
echo Choose your action:
echo.
echo 1. 🖼️  Analyze a single image
echo 2. 📁 Analyze all images in a folder
echo 3. 🔧 Adjust settings (Confidence, Enhancement, TTA, Slicing)
echo 4. 🎯 Per-class confidence settings
echo 5. 🔄 Change model
echo 6. ❌ Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto SINGLE_IMAGE
if "%choice%"=="2" goto FOLDER_IMAGES
if "%choice%"=="3" goto OPTIONS
if "%choice%"=="4" goto CLASS_CONF_OPTIONS
if "%choice%"=="5" goto MODEL_SELECTION
if "%choice%"=="6" goto EXIT

echo.
echo ❌ Invalid choice. Please try again.
pause
goto MENU

:SINGLE_IMAGE
cls
echo ================================================================
echo                      🖼️ Single Image Analysis
echo ================================================================
echo.
echo Model: %MODEL_NAME%
echo Script: %INFERENCE_SCRIPT%
echo.
echo Please drag and drop an image file here, or type the full path:
echo.
set /p "image_path=> "

if "%image_path%"=="" (
    echo ⚠️ No path entered.
    pause
    goto MENU
)

:: Remove quotes and trim spaces
set "image_path=!image_path:"=!"
for /f "tokens=*" %%a in ("!image_path!") do set "image_path=%%a"

echo.
echo 🚀 Starting analysis of "%image_path%"...
echo.

if "%MODEL_TYPE%"=="faster_rcnn" (
    echo Running Faster R-CNN inference...
    python %INFERENCE_SCRIPT% "%image_path%" %SAVE_FLAG% %ENHANCE_FLAG% %TTA_FLAG% %CLASS_CONF_FLAG% --confidence %CONFIDENCE% --model %MODEL_TYPE%
) else (
    echo Running YOLO inference...
    python %INFERENCE_SCRIPT% "%image_path%" %SAVE_FLAG% %ENHANCE_FLAG% %TTA_FLAG% %TILING_FLAG% %CLASS_CONF_FLAG% --confidence %CONFIDENCE% --model %MODEL_TYPE% %MULTI_MODEL_FLAG%
)

echo.
pause
goto MENU

:FOLDER_IMAGES
cls
echo ================================================================
echo                     📁 Folder Analysis
echo ================================================================
echo.
echo Model: %MODEL_NAME%
echo Script: %INFERENCE_SCRIPT%
echo.
echo Please drag and drop a folder of images here, or type the full path:
echo.
set /p "folder_path=> "

if "%folder_path%"=="" (
    echo ⚠️ No path entered.
    pause
    goto MENU
)

set "folder_path=!folder_path:"=!"
for /f "tokens=*" %%a in ("!folder_path!") do set "folder_path=%%a"

echo.
echo 🚀 Starting analysis of images in "%folder_path%"...
echo.

if "%MODEL_TYPE%"=="faster_rcnn" (
    echo Running Faster R-CNN inference...
    python %INFERENCE_SCRIPT% "%folder_path%" %SAVE_FLAG% %ENHANCE_FLAG% %TTA_FLAG% %CLASS_CONF_FLAG% --confidence %CONFIDENCE% --model %MODEL_TYPE%
) else (
    echo Running YOLO inference...
    python %INFERENCE_SCRIPT% "%folder_path%" %SAVE_FLAG% %ENHANCE_FLAG% %TTA_FLAG% %TILING_FLAG% %CLASS_CONF_FLAG% --confidence %CONFIDENCE% --model %MODEL_TYPE% %MULTI_MODEL_FLAG%
)

echo.
pause
goto MENU

:OPTIONS
cls
echo ================================================================
echo                          🔧 Settings
echo ================================================================
echo.
echo Current Settings:
echo ----------------------------------------------------------------
echo • Model:                 %MODEL_NAME%
echo • Inference Script:      %INFERENCE_SCRIPT%
if /i "%USE_CLASS_CONF%"=="Yes" (
    if "%MODEL_TYPE%"=="yolov8_single" (
        echo    • Per-Class Confidence:
        echo       • Class 0 - Crack: 0.2
    ) else (
        echo   • Per-Class Confidence:
        echo      • Transverse:    %CONF_0%
        echo      • Longitudinal:  %CONF_1%
        echo      • Joint:         %CONF_2%
        echo      • Alligator:     %CONF_3%
    )
) else (
    echo   • Per-Class Confidence:  No
    echo   • Global Confidence:     %CONFIDENCE%
)
echo • Image Enhancement:     %ENHANCED%
echo • Test Time Aug (TTA):   %TTA%
echo • 4-Tile Slicing:        %TILING%
echo ----------------------------------------------------------------
echo.
set /p new_conf="Enter new global confidence (0.0-1.0, leave blank for current): "
if not "%new_conf%"=="" set "CONFIDENCE=%new_conf%"

set /p new_enhance="Enable image enhancement? (1=Yes, 0=No, leave blank for current): "
if "%new_enhance%"=="1" (
    set "ENHANCE_FLAG=--enhance"
    set "ENHANCED=Enabled"
) else if "%new_enhance%"=="0" (
    set "ENHANCE_FLAG="
    set "ENHANCED=Disabled"
)

set /p new_tta="Enable Test Time Augmentation (TTA)? (1=Yes, 0=No, leave blank for current): "
if "%new_tta%"=="1" (
    set "TTA_FLAG=--tta"
    set "TTA=Enabled"
) else if "%new_tta%"=="0" (
    set "TTA_FLAG="
    set "TTA=Disabled"
)

set /p new_tiling="Enable 4-Tile Slicing Inference? (1=Yes, 0=No, leave blank for current): "
if "%new_tiling%"=="1" (
    set "TILING_FLAG=--tiling"
    set "TILING=Enabled"
) else if "%new_tiling%"=="0" (
    set "TILING_FLAG="
    set "TILING=Disabled"
)

echo.
echo ✅ Settings updated.
echo.
pause
goto MENU

:CLASS_CONF_OPTIONS
cls
echo ================================================================
echo                   🎯 Per-Class Confidence Settings
echo ================================================================
echo.
echo Current Model: %MODEL_NAME%
echo.

if "%MODEL_TYPE%"=="yolov8_single" goto SINGLE_CLASS_CONF
goto MULTI_CLASS_CONF

:SINGLE_CLASS_CONF
echo Single class model (Crack only)
echo.
echo Current Settings:
echo ----------------------------------------------------------------
echo • Class 0 - Crack:       %CONF_0%
echo ----------------------------------------------------------------
echo.
set /p new_conf_0="Enter confidence for Crack (leave blank for current): "
if not "%new_conf_0%"=="" set "CONF_0=%new_conf_0%"

set /p use_class="Use per-class confidence? (1=Yes, 0=No): "
if "%use_class%"=="1" (
    set "CLASS_CONF_FLAG=--class-confidences 0:%CONF_0%"
    set "USE_CLASS_CONF=Yes (Crack: %CONF_0%)"
) else (
    set "CLASS_CONF_FLAG="
    set "USE_CLASS_CONF=No"
)
echo.
echo ✅ Per-class confidence settings updated.
pause
goto MENU

:MULTI_CLASS_CONF
echo Multi-class model (4 crack types)
echo.
echo Current Settings:
echo ----------------------------------------------------------------
echo • Class 0 - Transverse:    %CONF_0%
echo • Class 1 - Longitudinal:  %CONF_1%
echo • Class 2 - Joint:         %CONF_2%
echo • Class 3 - Alligator:     %CONF_3%
echo ----------------------------------------------------------------
echo.
set /p new_conf_0="Transverse confidence (leave blank for current): "
if not "%new_conf_0%"=="" set "CONF_0=%new_conf_0%"

set /p new_conf_1="Longitudinal confidence (leave blank for current): "
if not "%new_conf_1%"=="" set "CONF_1=%new_conf_1%"

set /p new_conf_2="Joint confidence (leave blank for current): "
if not "%new_conf_2%"=="" set "CONF_2=%new_conf_2%"

set /p new_conf_3="Alligator confidence (leave blank for current): "
if not "%new_conf_3%"=="" set "CONF_3=%new_conf_3%"

set /p use_class="Use per-class confidence? (1=Yes, 0=No): "
if "%use_class%"=="1" (
    set "CLASS_CONF_FLAG=--class-confidences 0:%CONF_0%,1:%CONF_1%,2:%CONF_2%,3:%CONF_3%"
    set "USE_CLASS_CONF=Yes"
) else (
    set "CLASS_CONF_FLAG="
    set "USE_CLASS_CONF=No"
)
echo.
echo ✅ Per-class confidence settings updated.
pause
goto MENU

:EXIT
echo.
echo 👋 Thank you for using the Crack Detection Tool!
echo.
timeout /t 2 /nobreak >nul
exit