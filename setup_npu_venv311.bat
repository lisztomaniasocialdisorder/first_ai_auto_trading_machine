@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [1/5] ?? Python 3.11 ...
py -3.11 -c "import sys; print(sys.version)" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ??? Python 3.11?
  echo ??? https://www.python.org/downloads/release/python-3119/ ??????
  pause
  exit /b 1
)

echo [2/5] ?? .venv311 ...
if not exist ".venv311\Scripts\python.exe" (
  py -3.11 -m venv .venv311
  if errorlevel 1 (
    echo [ERROR] ?? .venv311 ??
    pause
    exit /b 1
  )
)

echo [3/5] ?? pip ...
.\.venv311\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] pip ????
  pause
  exit /b 1
)

echo [4/5] ?????? ...
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] requirements ????
  pause
  exit /b 1
)

echo [5/5] ?? NPU ?? torch-directml ...
.\.venv311\Scripts\python.exe -m pip install torch-directml
if errorlevel 1 (
  echo [ERROR] torch-directml ????
  pause
  exit /b 1
)

echo [OK] NPU ?????.venv311
pause
