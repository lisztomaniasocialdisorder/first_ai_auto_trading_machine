param()

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$port = 8600

if (-not (Test-Path -LiteralPath ".venv\\Scripts\\streamlit.exe")) {
  Write-Host "[ERROR] 找不到 .venv\\Scripts\\streamlit.exe"
  Write-Host "請先執行: python -m venv .venv; .venv\\Scripts\\activate; pip install -r requirements.txt"
  exit 1
}

if (-not (Test-Path -LiteralPath "dashboard.py")) {
  Write-Host "[ERROR] 找不到 dashboard.py"
  exit 1
}

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = Join-Path $logDir "streamlit.out.log"
$errLog = Join-Path $logDir "streamlit.err.log"

Write-Host "[INFO] 啟動儀表板服務 (port $port)..."
Start-Process -FilePath ".venv\\Scripts\\streamlit.exe" -ArgumentList @("run","dashboard.py","--server.port",$port,"--server.headless","true") -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden

$health = "http://127.0.0.1:$port/_stcore/health"
Write-Host "[INFO] 等待服務啟動..."
for ($i=0; $i -lt 40; $i++) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 $health
    if ($r.StatusCode -eq 200) { break }
  } catch { }
  Start-Sleep -Seconds 1
}

try {
  $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 $health
  if ($r.StatusCode -ne 200) { throw "health check failed" }
} catch {
  Write-Host "[ERROR] 服務沒有成功啟動，請查看 $outLog 與 $errLog"
  exit 1
}

Start-Process "http://127.0.0.1:$port"
Write-Host "[INFO] 已開啟瀏覽器：http://127.0.0.1:$port"
Write-Host "[INFO] 伺服器已在背景執行；日誌：$outLog / $errLog"
