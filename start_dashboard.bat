@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\start_dashboard.ps1"
exit /b %ERRORLEVEL%
