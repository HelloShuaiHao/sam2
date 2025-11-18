@echo off
REM ##############################################################################
REM SAM2 Demo - Windows Chrome Launcher
REM
REM This script launches Google Chrome with special flags to allow the SAM2 demo
REM to work over HTTP (instead of requiring HTTPS).
REM
REM Usage:
REM   Double-click this file, or run from Command Prompt:
REM   start-sam2-chrome-windows.bat
REM ##############################################################################

setlocal

REM Configuration
set SAM2_URL=http://ai.bygpu.com:55305/sam2/
set CHROME_USER_DATA_DIR=%TEMP%\chrome-sam2
set CHROME_PATH_1=C:\Program Files\Google\Chrome\Application\chrome.exe
set CHROME_PATH_2=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe

echo.
echo ==========================================
echo   SAM2 Demo - Chrome Launcher (Windows)
echo ==========================================
echo.

REM Find Chrome installation
set CHROME_PATH=
if exist "%CHROME_PATH_1%" (
    set CHROME_PATH=%CHROME_PATH_1%
) else if exist "%CHROME_PATH_2%" (
    set CHROME_PATH=%CHROME_PATH_2%
) else (
    echo [ERROR] Google Chrome is not installed!
    echo Please install Google Chrome from: https://www.google.com/chrome/
    echo.
    pause
    exit /b 1
)

echo Chrome found at: %CHROME_PATH%
echo.

REM Close existing Chrome instances
echo Closing existing Chrome instances...
taskkill /F /IM chrome.exe 2>nul
timeout /t 2 /nobreak >nul

REM Clean up old user data directory
if exist "%CHROME_USER_DATA_DIR%" (
    echo Cleaning up previous session data...
    rmdir /S /Q "%CHROME_USER_DATA_DIR%" 2>nul
)

REM Launch Chrome with special flags
echo Launching Chrome for SAM2 Demo...
echo.
echo URL: %SAM2_URL%
echo User Data Dir: %CHROME_USER_DATA_DIR%
echo.
echo [WARNING] This Chrome instance uses special flags for development.
echo [WARNING] Do NOT use it for regular browsing!
echo.

start "" "%CHROME_PATH%" ^
  --unsafely-treat-insecure-origin-as-secure="http://ai.bygpu.com:55305" ^
  --user-data-dir="%CHROME_USER_DATA_DIR%" ^
  --disable-features=SecureContextCheck ^
  --no-first-run ^
  --no-default-browser-check ^
  "%SAM2_URL%"

timeout /t 2 /nobreak >nul
echo.
echo Chrome launched successfully!
echo.
echo To verify WebCodecs is working, press F12 in Chrome and run:
echo   console.log('VideoEncoder' in window)
echo.
echo Should return: true
echo.
echo Press any key to exit...
pause >nul
