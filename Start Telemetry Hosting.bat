@echo off
setlocal
title SUZUKA Telemetry - Local Hosting + ngrok

echo.
echo   ============================================
echo    SUZUKA TELEMETRY - LOCAL HOSTING + NGROK
echo    Close this window (or Ctrl+C) to stop.
echo   ============================================
echo.

rem --- Sanity checks ---
if not exist "%~dp0server\app.py" (
  echo [ERROR] server\app.py not found next to this script. & pause & exit /b 1
)
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [ERROR] .venv missing. Create it first:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt & pause & exit /b 1
)
where ngrok >nul 2>&1 || (
  echo [ERROR] ngrok not found on PATH. Install:  winget install Ngrok.Ngrok & pause & exit /b 1
)

rem --- Clean up any previous run ---
taskkill /f /im ngrok.exe >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*uvicorn*8321*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout.exe /t 1 /nobreak >nul

rem --- 1. Telemetry server (port 8321, localhost only) ---
echo [1/2] Starting telemetry server on port 8321...
start "mt-server-8321" /min "%~dp0.venv\Scripts\python.exe" -m uvicorn app:app --app-dir "%~dp0server" --host 127.0.0.1 --port 8321

rem --- Wait until the server answers (max ~30 s) ---
set /a tries=0
:waitloop
curl -s --max-time 2 http://127.0.0.1:8321/ >nul 2>&1 && goto ready
set /a tries+=1
if %tries% geq 15 (
  echo [ERROR] Server did not start in ~30 s. Check the minimized "mt-server-8321" window for errors. & pause & exit /b 1
)
timeout.exe /t 2 /nobreak >nul
goto waitloop

:ready
echo       Local demo ready: http://localhost:8321
echo.

rem --- 2. ngrok tunnel (your public URL appears below and stays live in this window) ---
echo [2/2] Starting ngrok tunnel...
echo       NOTE: first browser visit shows an ngrok "Visit Site" interstitial - click it once.
echo.
ngrok http 8321

rem --- Cleanup when the tunnel stops ---
echo.
echo Tunnel stopped - shutting down telemetry server...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*uvicorn*8321*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
echo Done. You can close this window.
pause
