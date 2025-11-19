@echo off
REM ##############################################################################
REM SAM2 Demo - Docker Restart Script with GPU Memory Cleanup (Windows)
REM
REM This script stops, rebuilds, and restarts the SAM2 Docker containers
REM with the latest code changes (including GPU memory leak fixes)
REM
REM Usage:
REM   Run from Command Prompt in the sam2 directory:
REM   restart-sam2-docker-windows.bat
REM ##############################################################################

setlocal

echo.
echo ==========================================
echo   SAM2 Docker - Restart ^& Rebuild
echo ==========================================
echo.

REM Check if docker-compose.yaml exists
if not exist "docker-compose.yaml" (
    echo [ERROR] docker-compose.yaml not found!
    echo Please run this script from the sam2 directory.
    pause
    exit /b 1
)

REM Step 1: Stop all containers
echo Step 1/4: Stopping all containers...
docker compose down
if errorlevel 1 (
    echo [ERROR] Failed to stop containers!
    pause
    exit /b 1
)
echo [OK] Containers stopped
echo.

REM Wait for containers to fully stop
timeout /t 3 /nobreak >nul

REM Step 2: Show current GPU memory usage
echo Current GPU Memory Status:
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
) else (
    echo nvidia-smi not available
)
echo.

REM Step 3: Rebuild backend image with latest code
echo Step 2/4: Rebuilding backend image with GPU memory leak fixes...
echo This includes the following fixes:
echo   * Session close cleanup (predictor.py)
echo   * Reset state GPU cache clearing (sam2_video_predictor.py)
echo   * Propagation completion cleanup
echo   * Frame output tensor cleanup
echo   * Auto-close old sessions before starting new ones
echo.

docker compose build backend
if errorlevel 1 (
    echo [ERROR] Failed to rebuild backend!
    pause
    exit /b 1
)
echo [OK] Backend rebuilt with memory leak fixes
echo.

REM Step 4: Start all services
echo Step 3/4: Starting all services...
docker compose up -d
if errorlevel 1 (
    echo [ERROR] Failed to start services!
    pause
    exit /b 1
)
echo [OK] Services started
echo.

REM Wait for services to initialize
echo Waiting for services to initialize...
timeout /t 5 /nobreak >nul

REM Step 5: Show service status
echo Step 4/4: Checking service status...
docker compose ps
echo.

REM Show GPU memory after restart
echo GPU Memory After Restart:
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
)
echo.

echo ==========================================
echo   SAM2 Demo Restarted Successfully!
echo ==========================================
echo.
echo Access the demo at:
echo   Frontend: http://ai.bygpu.com:55305/sam2/
echo   Backend API: http://ai.bygpu.com:55305/api/sam2
echo.
echo To view backend logs:
echo   docker compose logs -f backend
echo.
echo To view all logs:
echo   docker compose logs -f
echo.
echo GPU Memory Leak Fixes Applied:
echo   * Sessions now properly clean up GPU memory on close
echo   * Old sessions auto-close when uploading new videos
echo   * Periodic GPU cache clearing during propagation
echo   * Frame output tensors cleaned up immediately
echo.
echo Test by uploading multiple videos consecutively!
echo.
pause
