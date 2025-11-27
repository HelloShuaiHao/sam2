@echo off
REM ##############################################################################
REM SAM2 Demo - Windows Chrome Launcher (FIXED FOR CHINESE PATHS)
REM
REM This version handles Chinese characters in paths correctly
REM ##############################################################################

REM IMPORTANT: Save this file as ANSI/GBK encoding, NOT UTF-8!

REM Change to system default code page for Chinese
chcp 936 >nul 2>&1

REM Create log file using a safe English-only path
set LOG_FILE=%TEMP%\sam2-chrome-launcher-debug.log

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
echo   SAM2 Demo - Chrome Launcher
echo ==========================================
echo.
echo Log file: %LOG_FILE%
echo.

setlocal EnableDelayedExpansion

REM Configuration
set SAM2_URL=http://ai.bygpu.com:55305/sam2/
set CHROME_USER_DATA_DIR=%TEMP%\chrome-sam2
set CHROME_PATH_1=C:\Program Files\Google\Chrome\Application\chrome.exe
set CHROME_PATH_2=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe
set CHROME_PATH_3=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe

echo [DEBUG] Configuration done >> "%LOG_FILE%" 2>&1

REM Find Chrome installation
echo Searching for Chrome...
echo [DEBUG] Searching for Chrome... >> "%LOG_FILE%" 2>&1

set CHROME_PATH=
if exist "%CHROME_PATH_1%" (
    set "CHROME_PATH=%CHROME_PATH_1%"
    echo [DEBUG] Found at PATH_1 >> "%LOG_FILE%" 2>&1
    echo [OK] Found Chrome
    goto :chrome_found
)

echo [DEBUG] Not at PATH_1 >> "%LOG_FILE%" 2>&1

if exist "%CHROME_PATH_2%" (
    set "CHROME_PATH=%CHROME_PATH_2%"
    echo [DEBUG] Found at PATH_2 >> "%LOG_FILE%" 2>&1
    echo [OK] Found Chrome
    goto :chrome_found
)

echo [DEBUG] Not at PATH_2 >> "%LOG_FILE%" 2>&1

if exist "%CHROME_PATH_3%" (
    set "CHROME_PATH=%CHROME_PATH_3%"
    echo [DEBUG] Found at PATH_3 >> "%LOG_FILE%" 2>&1
    echo [OK] Found Chrome
    goto :chrome_found
)

echo [DEBUG] Chrome not found! >> "%LOG_FILE%" 2>&1
echo.
echo [ERROR] Google Chrome is not installed!
echo.
echo Searched in:
echo   - %CHROME_PATH_1%
echo   - %CHROME_PATH_2%
echo   - %CHROME_PATH_3%
echo.
echo Log: %LOG_FILE%
echo.
pause
exit /b 1

:chrome_found
echo. >> "%LOG_FILE%" 2>&1

REM Close existing Chrome
echo Closing existing Chrome...
echo [DEBUG] Closing Chrome... >> "%LOG_FILE%" 2>&1

taskkill /F /IM chrome.exe >nul 2>&1
if !errorlevel! equ 0 (
    echo [OK] Closed Chrome processes
    echo [DEBUG] Closed >> "%LOG_FILE%" 2>&1
) else (
    echo [INFO] No Chrome running
    echo [DEBUG] None running >> "%LOG_FILE%" 2>&1
)

timeout /t 2 /nobreak >nul 2>&1

REM Clean temp directory
echo Cleaning temp directory...
echo [DEBUG] Cleaning: %CHROME_USER_DATA_DIR% >> "%LOG_FILE%" 2>&1

if exist "%CHROME_USER_DATA_DIR%" (
    rmdir /S /Q "%CHROME_USER_DATA_DIR%" >nul 2>&1
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

REM Launch Chrome
echo Launching Chrome...
echo [DEBUG] Launching... >> "%LOG_FILE%" 2>&1
echo [DEBUG] Path: !CHROME_PATH! >> "%LOG_FILE%" 2>&1
echo [DEBUG] URL: %SAM2_URL% >> "%LOG_FILE%" 2>&1
echo.
echo [WARNING] Development mode - do NOT use for regular browsing!
echo.

start "" "!CHROME_PATH!" --unsafely-treat-insecure-origin-as-secure="http://ai.bygpu.com:55305" --user-data-dir="%CHROME_USER_DATA_DIR%" --disable-features=SecureContextCheck --no-first-run --no-default-browser-check "%SAM2_URL%" 2>> "%LOG_FILE%"

if !errorlevel! neq 0 (
    echo [DEBUG] Launch failed: !errorlevel! >> "%LOG_FILE%" 2>&1
    echo.
    echo [ERROR] Failed to launch Chrome!
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
echo [OK] Chrome launched!
echo.
echo To verify WebCodecs:
echo   1. Press F12 in Chrome
echo   2. Console tab
echo   3. Run: console.log('VideoEncoder' in window)
echo   4. Should return: true
echo.
echo Log: %LOG_FILE%
echo.
echo [DEBUG] Completed at %DATE% %TIME% >> "%LOG_FILE%" 2>&1
echo.
pause
