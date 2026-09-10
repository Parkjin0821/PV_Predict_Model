$ErrorActionPreference = "Stop"
$dataDir = Split-Path -Parent $PSScriptRoot
$project = Split-Path -Parent $dataDir
$python = "C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$collector = Join-Path $PSScriptRoot "collect_kma_nwp_nc_live_v1_2026-09-09.py"
$statusGenerator = Join-Path $dataDir "generate_collection_status_v1_2026-09-02.py"
$logDir = Join-Path $PSScriptRoot "logs\kim_nc_live"
$lockPath = Join-Path $logDir "running.lock"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
if (Test-Path -LiteralPath $lockPath) {
    $age = (Get-Date) - (Get-Item -LiteralPath $lockPath).LastWriteTime
    if ($age.TotalMinutes -lt 25) { exit 0 }
    Remove-Item -LiteralPath $lockPath -Force
}

New-Item -ItemType File -Path $lockPath -Force | Out-Null
$logPath = Join-Path $logDir ((Get-Date).ToString("yyyyMMdd") + ".log")
try {
    & $python $collector --all-regions 2>&1 | Out-File -LiteralPath $logPath -Append -Encoding utf8
    $collectorExit = $LASTEXITCODE
    & $python $statusGenerator 2>&1 | Out-File -LiteralPath $logPath -Append -Encoding utf8
    if ($collectorExit -ne 0) { exit $collectorExit }
    exit $LASTEXITCODE
}
finally {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
