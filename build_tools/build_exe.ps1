param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m pip install pyinstaller

pyinstaller --noconfirm --clean --onefile --windowed --name BTC_AI_Backtest `
  --add-data "src;src" `
  --add-data ".env.example;." `
  --add-data "README.md;." `
  app_gui.py

Write-Host "Build done: dist\BTC_AI_Backtest.exe"
