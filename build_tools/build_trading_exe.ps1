param()

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

python -m pip install pyinstaller | Out-Null

# Bundle the full app folder so launcher can run streamlit dashboard from inside onefile.
pyinstaller --noconfirm --clean --onefile --name trading `
  --add-data "dashboard.py;app" `
  --add-data "src;app/src" `
  --add-data ".env;app" `
  --collect-all "streamlit" `
  --collect-all "plotly" `
  --collect-all "pandas" `
  --collect-all "numpy" `
  launcher_trading.py

Write-Host "Build done: dist\\trading.exe"
