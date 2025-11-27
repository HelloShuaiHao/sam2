@echo off
REM ##############################################################################
REM SAM2 Demo - Docker Restart (FIXED FOR CHINESE PATHS)
REM
REM This version handles Chinese characters in paths correctly
REM ##############################################################################

REM IMPORTANT: Save this file as ANSI/GBK encoding, NOT UTF-8!

REM Change to system default code page for Chinese
chcp 936 >nul 2>&1

REM Create log file using a safe English-only path
set LOG_FILE=%TEMP%\sam2-docker-restart-debug.log

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
echo   SAM2 Docker - Restart ^& Rebuild
echo ==========================================
echo.
echo Log file: %LOG_FILE%
echo.

setlocal EnableDelayedExpansion

REM Check if docker-compose.yaml exists
echo Checking for docker-compose.yaml...
echo [DEBUG] Checking for docker-compose.yaml... >> "%LOG_FILE%" 2>&1

if not exist "docker-compose.yaml" (
    echo [ERROR] docker-compose.yaml not found! >> "%LOG_FILE%" 2>&1
    echo [DEBUG] Current dir: %CD% >> "%LOG_FILE%" 2>&1
    echo.
    echo [ERROR] docker-compose.yaml not found!
    echo Please run from the sam2 directory.
    echo.
    echo Current: %CD%
    echo.
    pause
    exit /b 1
)
echo [DEBUG] Found docker-compose.yaml >> "%LOG_FILE%" 2>&1
echo [OK] Found docker-compose.yaml
echo.

REM Detect Docker Compose
echo Detecting Docker...
echo [DEBUG] Detecting Docker Compose... >> "%LOG_FILE%" 2>&1

set DOCKER_COMPOSE_CMD=

docker compose version >nul 2>&1
if !errorlevel! equ 0 (
    set DOCKER_COMPOSE_CMD=docker compose
    echo [DEBUG] Found V2 >> "%LOG_FILE%" 2>&1
    echo [OK] Found Docker Compose V2
    goto :docker_found
)

echo [DEBUG] Trying V1... >> "%LOG_FILE%" 2>&1
docker-compose version >nul 2>&1
if !errorlevel! equ 0 (
    set DOCKER_COMPOSE_CMD=docker-compose
    echo [DEBUG] Found V1 >> "%LOG_FILE%" 2>&1
    echo [OK] Found Docker Compose V1
    goto :docker_found
)

echo [ERROR] Docker Compose not found! >> "%LOG_FILE%" 2>&1
echo.
echo [ERROR] Docker Compose not installed!
echo.
echo Install from: https://www.docker.com/products/docker-desktop/
echo.
pause
exit /b 1

:docker_found
echo [DEBUG] Using: !DOCKER_COMPOSE_CMD! >> "%LOG_FILE%" 2>&1
echo.

REM Stop containers
echo Step 1/4: Stopping containers...
echo [DEBUG] Stopping... >> "%LOG_FILE%" 2>&1

!DOCKER_COMPOSE_CMD! down >> "%LOG_FILE%" 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Stop failed: !errorlevel! >> "%LOG_FILE%" 2>&1
    echo [ERROR] Failed to stop!
    echo.
    pause
    exit /b 1
)
echo [DEBUG] Stopped >> "%LOG_FILE%" 2>&1
echo [OK] Stopped
echo.

timeout /t 3 /nobreak >nul 2>&1

REM Check GPU
echo GPU Status:
echo [DEBUG] Checking GPU... >> "%LOG_FILE%" 2>&1
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >> "%LOG_FILE%" 2>&1
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
) else (
    echo [DEBUG] nvidia-smi not found >> "%LOG_FILE%" 2>&1
    echo nvidia-smi not available
)
echo.

REM Rebuild backend
echo Step 2/4: Rebuilding backend...
echo [DEBUG] Building... >> "%LOG_FILE%" 2>&1
echo This may take several minutes...
echo.

!DOCKER_COMPOSE_CMD! build backend >> "%LOG_FILE%" 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Build failed: !errorlevel! >> "%LOG_FILE%" 2>&1
    echo [ERROR] Failed to build!
    echo.
    pause
    exit /b 1
)
echo [DEBUG] Built >> "%LOG_FILE%" 2>&1
echo [OK] Backend rebuilt
echo.

REM Start services
echo Step 3/4: Starting services...
echo [DEBUG] Starting... >> "%LOG_FILE%" 2>&1

!DOCKER_COMPOSE_CMD! up -d >> "%LOG_FILE%" 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Start failed: !errorlevel! >> "%LOG_FILE%" 2>&1
    echo [ERROR] Failed to start!
    echo.
    pause
    exit /b 1
)
echo [DEBUG] Started >> "%LOG_FILE%" 2>&1
echo [OK] Services started
echo.

echo Waiting for initialization...
timeout /t 5 /nobreak >nul 2>&1

REM Show status
echo Step 4/4: Service status...
echo [DEBUG] Checking status... >> "%LOG_FILE%" 2>&1
!DOCKER_COMPOSE_CMD! ps >> "%LOG_FILE%" 2>&1
!DOCKER_COMPOSE_CMD! ps
echo.

REM GPU after restart
echo GPU After Restart:
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >> "%LOG_FILE%" 2>&1
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
)
echo.

echo ==========================================
echo   Success!
echo ==========================================
echo.
echo Access:
echo   Frontend: http://ai.bygpu.com:55305/sam2/
echo   Backend: http://ai.bygpu.com:55305/api/sam2
echo.
echo Logs:
echo   Backend: docker compose logs -f backend
echo   All: docker compose logs -f
echo.
echo Debug log: %LOG_FILE%
echo.
echo [DEBUG] Completed at %DATE% %TIME% >> "%LOG_FILE%" 2>&1
echo.
pause
