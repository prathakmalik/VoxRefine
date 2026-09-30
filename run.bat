@echo off
title VoxRefine Server
echo ========================================
echo       Starting VoxRefine Server
echo ========================================
echo.

:: Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Using 'uv' to start the server...
    set START_CMD=uv run python -m app.main
) else (
    echo 'uv' not found, falling back to standard 'python'...
    set START_CMD=python -m app.main
)

echo Starting server...
echo.

:: Run the server directly in this window.
:: This allows Ctrl+C to be captured by the Python process and uvicorn.
%START_CMD%

echo.
echo ========================================
echo       Server has stopped.
echo ========================================
pause
