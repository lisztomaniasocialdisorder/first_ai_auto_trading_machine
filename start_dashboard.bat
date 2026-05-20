@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\start_dashboard.ps1"
exit /b %ERRORLEVEL%
