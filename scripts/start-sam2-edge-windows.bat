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

setlocal

REM Configuration
set SAM2_URL=http://ai.bygpu.com:55305/sam2/
set EDGE_USER_DATA_DIR=%TEMP%\edge-sam2
set EDGE_PATH_1=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
set EDGE_PATH_2=C:\Program Files\Microsoft\Edge\Application\msedge.exe

echo.
echo ==========================================
echo   SAM2 Demo - Edge Launcher (Windows)
echo ==========================================
echo.

REM Find Edge installation
set EDGE_PATH=
if exist "%EDGE_PATH_1%" (
    set EDGE_PATH=%EDGE_PATH_1%
) else if exist "%EDGE_PATH_2%" (
    set EDGE_PATH=%EDGE_PATH_2%
) else (
    echo [ERROR] Microsoft Edge is not installed!
    echo Edge should be pre-installed on Windows 10/11.
    echo If missing, download from: https://www.microsoft.com/edge
    echo.
    pause
    exit /b 1
)

echo Edge found at: %EDGE_PATH%
echo.

REM Close existing Edge instances
echo Closing existing Edge instances...
taskkill /F /IM msedge.exe 2>nul
timeout /t 2 /nobreak >nul

REM Clean up old user data directory
if exist "%EDGE_USER_DATA_DIR%" (
    echo Cleaning up previous session data...
    rmdir /S /Q "%EDGE_USER_DATA_DIR%" 2>nul
)

REM Launch Edge with special flags
echo Launching Edge for SAM2 Demo...
echo.
echo URL: %SAM2_URL%
echo User Data Dir: %EDGE_USER_DATA_DIR%
echo.
echo [WARNING] This Edge instance uses special flags for development.
echo [WARNING] Do NOT use it for regular browsing!
echo.

start "" "%EDGE_PATH%" ^
  --unsafely-treat-insecure-origin-as-secure="http://ai.bygpu.com:55305" ^
  --user-data-dir="%EDGE_USER_DATA_DIR%" ^
  --disable-features=SecureContextCheck ^
  --no-first-run ^
  --no-default-browser-check ^
  "%SAM2_URL%"

timeout /t 2 /nobreak >nul
echo.
echo Edge launched successfully!
echo.
echo To verify WebCodecs is working, press F12 in Edge and run:
echo   console.log('VideoEncoder' in window)
echo.
echo Should return: true
echo.
echo Press any key to exit...
pause >nul
