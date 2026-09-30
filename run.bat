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

echo Starting server in background...
start /b %START_CMD%

echo Waiting for server to initialize...
timeout /t 5 /nobreak > nul

echo Launching browser...
start http://localhost:8000/static/index.html

echo.
echo ========================================
echo   Server is running. 
echo   Press Ctrl+C to stop the server.
echo ========================================
echo.

:: Keep the window open and the shell active to show server logs
cmd /k
