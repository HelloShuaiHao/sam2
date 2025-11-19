@echo off
REM ##############################################################################
REM SAM2 Demo - Windows Firefox Launcher
REM
REM This script launches Firefox with a separate profile for SAM2 demo,
REM without affecting your regular Firefox browsing data.
REM
REM Usage:
REM   Double-click this file, or run from Command Prompt:
REM   start-sam2-firefox-windows.bat
REM ##############################################################################

setlocal

REM Configuration
set SAM2_URL=http://ai.bygpu.com:55305/sam2/
set FIREFOX_PROFILE_DIR=%TEMP%\firefox-sam2-profile
set FIREFOX_PATH_1=C:\Program Files\Mozilla Firefox\firefox.exe
set FIREFOX_PATH_2=C:\Program Files (x86)\Mozilla Firefox\firefox.exe

echo.
echo ==========================================
echo   SAM2 Demo - Firefox Launcher (Windows)
echo ==========================================
echo.

REM Find Firefox installation
set FIREFOX_PATH=
if exist "%FIREFOX_PATH_1%" (
    set FIREFOX_PATH=%FIREFOX_PATH_1%
) else if exist "%FIREFOX_PATH_2%" (
    set FIREFOX_PATH=%FIREFOX_PATH_2%
) else (
    echo [ERROR] Mozilla Firefox is not installed!
    echo Please install Firefox from: https://www.mozilla.org/firefox/
    echo.
    pause
    exit /b 1
)

echo Firefox found at: %FIREFOX_PATH%
echo.

REM Create profile directory if it doesn't exist
if not exist "%FIREFOX_PROFILE_DIR%" (
    echo Creating dedicated Firefox profile for SAM2...
    mkdir "%FIREFOX_PROFILE_DIR%"

    REM Create basic profile configuration
    (
        echo // SAM2 Demo Firefox Profile Configuration
        echo user_pref^("security.fileuri.strict_origin_policy", false^);
        echo user_pref^("privacy.file_unique_origin", false^);
        echo user_pref^("media.autoplay.default", 0^);
        echo user_pref^("media.autoplay.blocking_policy", 0^);
        echo user_pref^("permissions.default.camera", 1^);
        echo user_pref^("permissions.default.microphone", 1^);
        echo user_pref^("browser.startup.homepage", "about:blank"^);
        echo user_pref^("browser.shell.checkDefaultBrowser", false^);
        echo user_pref^("devtools.chrome.enabled", true^);
        echo user_pref^("devtools.debugger.remote-enabled", true^);
    ) > "%FIREFOX_PROFILE_DIR%\prefs.js"

    echo Profile created!
    echo.
)

REM Launch Firefox with dedicated profile
echo Launching Firefox for SAM2 Demo...
echo.
echo URL: %SAM2_URL%
echo Profile Dir: %FIREFOX_PROFILE_DIR%
echo.
echo [NOTE] This uses a separate Firefox profile.
echo [NOTE] Your regular Firefox browsing data remains intact!
echo.

start "" "%FIREFOX_PATH%" -profile "%FIREFOX_PROFILE_DIR%" -no-remote "%SAM2_URL%"

timeout /t 2 /nobreak >nul
echo.
echo Firefox launched successfully!
echo.
echo To verify WebCodecs support, press F12 in Firefox and run:
echo   console.log('VideoEncoder' in window)
echo.
echo Should return: true
echo.
echo Press any key to exit...
pause >nul
