@echo off
title VisionSign AI - Traffic Sign Recognition Web System
echo =====================================================================
echo  Starting VisionSign AI - Deep Learning Traffic Sign Recognition Web HUD
echo =====================================================================
echo.

IF EXIST "C:\Users\Asus\anaconda3\envs\isl\python.exe" (
    echo Using Anaconda 'isl' Python environment with TensorFlow...
    "C:\Users\Asus\anaconda3\envs\isl\python.exe" app.py
) ELSE (
    echo Using default system Python...
    python app.py
)

pause
