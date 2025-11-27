@echo off
REM ##############################################################################
REM SAM2 Demo - Windows Edge Launcher
REM
REM This script launches Microsoft Edge with special flags to allow the SAM2 demo
REM to work over HTTP (instead of requiring HTTPS).
REM
REM Usage:
REM   Double-click this file, or run from Command Prompt:
REM   start-sam2-edge-windows.bat
REM ##############################################################################

REM IMPORTANT: This file must be saved as ANSI/GBK encoding to handle Chinese paths!

REM Change to system default code page for Chinese
chcp 936 >nul 2>&1

REM Create log file using a safe English-only path
set LOG_FILE=%TEMP%\sam2-edge-launcher-debug.log

REM Clear previous log
echo. > "%LOG_FILE%" 2>&1

REM Log basic info
echo [DEBUG] Script started at %DATE% %TIME% >> "%LOG_FILE%" 2>&1
echo [DEBUG] Script path: %~dp0 >> "%LOG_FILE%" 2>&1
echo [DEBUG] Current directory: %CD% >> "%LOG_FILE%" 2>&1
echo. >> "%LOG_FILE%" 2>&1

REM Show on console
echo.
echo ==========================================
echo   SAM2 Demo - Edge Launcher
echo ==========================================
echo.
echo Log file: %LOG_FILE%
echo.

setlocal EnableDelayedExpansion

REM Configuration
set SAM2_URL=http://ai.bygpu.com:55305/sam2/
set EDGE_USER_DATA_DIR=%TEMP%\edge-sam2
set EDGE_PATH_1=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
set EDGE_PATH_2=C:\Program Files\Microsoft\Edge\Application\msedge.exe

echo [DEBUG] Configuration done >> "%LOG_FILE%" 2>&1

REM Find Edge installation
echo Searching for Edge...
echo [DEBUG] Searching for Edge... >> "%LOG_FILE%" 2>&1

set EDGE_PATH=
if exist "%EDGE_PATH_1%" (
    set "EDGE_PATH=%EDGE_PATH_1%"
    echo [DEBUG] Found at PATH_1 >> "%LOG_FILE%" 2>&1
    echo [OK] Found Edge
    goto :edge_found
)

echo [DEBUG] Not at PATH_1 >> "%LOG_FILE%" 2>&1

if exist "%EDGE_PATH_2%" (
    set "EDGE_PATH=%EDGE_PATH_2%"
    echo [DEBUG] Found at PATH_2 >> "%LOG_FILE%" 2>&1
    echo [OK] Found Edge
    goto :edge_found
)

echo [DEBUG] Edge not found! >> "%LOG_FILE%" 2>&1
echo.
echo [ERROR] Microsoft Edge is not installed!
echo.
echo Searched in:
echo   - %EDGE_PATH_1%
echo   - %EDGE_PATH_2%
echo.
echo Log: %LOG_FILE%
echo.
pause
exit /b 1

:edge_found
echo. >> "%LOG_FILE%" 2>&1

REM Close existing Edge
echo Closing existing Edge...
echo [DEBUG] Closing Edge... >> "%LOG_FILE%" 2>&1

taskkill /F /IM msedge.exe >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Closed Edge processes
    echo [DEBUG] Closed >> "%LOG_FILE%" 2>&1
) else (
    echo [INFO] No Edge running
    echo [DEBUG] None running >> "%LOG_FILE%" 2>&1
)

timeout /t 2 /nobreak >nul 2>&1

REM Clean temp directory
echo Cleaning temp directory...
echo [DEBUG] Cleaning: %EDGE_USER_DATA_DIR% >> "%LOG_FILE%" 2>&1

if exist "%EDGE_USER_DATA_DIR%" (
    rmdir /S /Q "%EDGE_USER_DATA_DIR%" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Cleaned
        echo [DEBUG] Cleaned >> "%LOG_FILE%" 2>&1
    ) else (
        echo [WARNING] Clean failed
        echo [DEBUG] Clean failed >> "%LOG_FILE%" 2>&1
    )
) else (
    echo [INFO] Nothing to clean
    echo [DEBUG] Nothing to clean >> "%LOG_FILE%" 2>&1
)
echo.

REM Launch Edge
echo Launching Edge...
echo [DEBUG] Launching... >> "%LOG_FILE%" 2>&1
echo [DEBUG] Path: !EDGE_PATH! >> "%LOG_FILE%" 2>&1
echo [DEBUG] URL: %SAM2_URL% >> "%LOG_FILE%" 2>&1
echo.
echo [WARNING] Development mode - do NOT use for regular browsing!
echo.

start "" "!EDGE_PATH!" --unsafely-treat-insecure-origin-as-secure="http://ai.bygpu.com:55305" --user-data-dir="%EDGE_USER_DATA_DIR%" --disable-features=SecureContextCheck --no-first-run --no-default-browser-check "%SAM2_URL%" 2>> "%LOG_FILE%"

if !errorlevel! neq 0 (
    echo [DEBUG] Launch failed: !errorlevel! >> "%LOG_FILE%" 2>&1
    echo.
    echo [ERROR] Failed to launch Edge!
    echo Error: !errorlevel!
    echo.
    echo Log: %LOG_FILE%
    echo.
    pause
    exit /b 1
)

echo [DEBUG] Launch command executed >> "%LOG_FILE%" 2>&1

timeout /t 3 /nobreak >nul 2>&1

echo.
echo [OK] Edge launched!
echo.
echo To verify WebCodecs:
echo   1. Press F12 in Edge
echo   2. Console tab
echo   3. Run: console.log('VideoEncoder' in window)
echo   4. Should return: true
echo.
echo Log: %LOG_FILE%
echo.
echo [DEBUG] Completed at %DATE% %TIME% >> "%LOG_FILE%" 2>&1
echo.
pause
