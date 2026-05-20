@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PORT=8600"
set "HEALTH_URL=http://127.0.0.1:%PORT%/_stcore/health"

REM Check if Streamlit is already running
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%HEALTH_URL%'; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% EQU 0 (
  echo [INFO] Service already running. Opening browser...
  start "" "http://127.0.0.1:%PORT%"
  exit /b 0
)

set "STREAMLIT_PATH="
if exist ".venv311\Scripts\streamlit.exe" (
  set "STREAMLIT_PATH=.venv311\Scripts\streamlit.exe"
) else if exist ".venv\Scripts\streamlit.exe" (
  set "STREAMLIT_PATH=.venv\Scripts\streamlit.exe"
)

if not defined STREAMLIT_PATH (
  echo [ERROR] Cannot find .venv\Scripts\streamlit.exe or .venv311\Scripts\streamlit.exe
  echo Please create the virtual environment first.
  pause
  exit /b 1
)

echo [INFO] Starting dashboard service on port %PORT%...

if not exist "logs" mkdir "logs"
start "" /B "%CD%\!STREAMLIT_PATH!" run "%CD%\dashboard.py" --server.port %PORT% --server.headless true > "%CD%\logs\streamlit.log" 2>&1

echo [INFO] Waiting for service to start...
set /a TRY=0
:wait_loop
set /a TRY+=1
if %TRY% GTR 40 goto wait_fail

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%HEALTH_URL%'; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% EQU 0 goto wait_ok

timeout /t 1 /nobreak >nul
goto wait_loop

:wait_ok
echo [INFO] Service started successfully.
start "" "http://127.0.0.1:%PORT%"
exit /b 0

:wait_fail
echo [ERROR] Failed to start service. Check logs\streamlit.log
pause
exit /b 1
