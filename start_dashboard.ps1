param()

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$port = 8600
$url = "http://127.0.0.1:$port"

$streamlitPath = $null
if (Test-Path -LiteralPath ".venv311\Scripts\streamlit.exe") {
  $streamlitPath = ".venv311\Scripts\streamlit.exe"
} elseif (Test-Path -LiteralPath ".venv\Scripts\streamlit.exe") {
  $streamlitPath = ".venv\Scripts\streamlit.exe"
}

if (-not $streamlitPath) {
  Write-Host "[ERROR] Cannot find .venv311\\Scripts\\streamlit.exe or .venv\\Scripts\\streamlit.exe"
  exit 1
}

if (-not (Test-Path -LiteralPath "dashboard.py")) {
  Write-Host "[ERROR] Cannot find dashboard.py"
  exit 1
}

$existing = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -in @("streamlit.exe", "python.exe", "pythonw.exe")) -and
  $_.CommandLine -like "*streamlit*" -and
  $_.CommandLine -like "*dashboard.py*" -and
  $_.CommandLine -like "*btc-1-k-ai-100-ma*"
}

if ($existing) {
  Write-Host "[INFO] Dashboard already running. Opening browser only."
  Start-Process $url
  exit 0
}

Write-Host "[INFO] Starting dashboard service on port $port..."
Start-Process -FilePath $streamlitPath -ArgumentList @("run", "dashboard.py", "--server.port", $port, "--server.headless", "true") -WindowStyle Hidden
Start-Sleep -Seconds 2
Start-Process $url
Write-Host "[INFO] Dashboard opening: $url"
