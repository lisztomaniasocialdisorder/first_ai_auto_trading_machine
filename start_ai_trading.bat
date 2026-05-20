@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PYTHON_EXE="
if exist ".venv311\Scripts\python.exe" (
  set "PYTHON_EXE=.venv311\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

if not exist "logs" mkdir "logs"

echo [INFO] AITrading integrated launcher
echo [INFO] Python: %PYTHON_EXE%
echo [INFO] Runner log: logs\auto_trading.log

REM Prevent duplicate auto runner processes.
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*auto_trade_runner.py*' -and $_.CommandLine -like '*btc-1-k-ai-100-ma*' }; if($p){exit 0}else{exit 1}"
if %ERRORLEVEL% EQU 0 (
  echo [INFO] Auto runner already running. Skip duplicate start.
) else (
  echo [INFO] Starting auto runner in background...
  start "AI Auto Trading Runner" /MIN cmd /c "\"%PYTHON_EXE%\" -u \"%CD%\auto_trade_runner.py\" --symbol BTCUSDT --interval 1h >> \"%CD%\logs\auto_trading.log\" 2>&1"
)

echo [INFO] Starting dashboard...
call "%CD%\start_dashboard.bat"

echo [INFO] Done. Dashboard is available and auto runner keeps running in background.
exit /b 0
