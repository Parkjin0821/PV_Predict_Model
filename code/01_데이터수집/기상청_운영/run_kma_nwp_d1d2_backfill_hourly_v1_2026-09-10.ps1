$ErrorActionPreference = "Stop"
$python = "C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$script = Join-Path $PSScriptRoot "run_kma_nwp_d1d2_backfill_hourly_v1_2026-09-10.py"
$logDir = Join-Path $PSScriptRoot "logs\d1d2_backfill_hourly"
$lockPath = Join-Path $logDir "running.lock"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
if (Test-Path -LiteralPath $lockPath) {
    $age = (Get-Date) - (Get-Item -LiteralPath $lockPath).LastWriteTime
    if ($age.TotalMinutes -lt 15) { exit 0 }
    Remove-Item -LiteralPath $lockPath -Force
}

New-Item -ItemType File -Path $lockPath -Force | Out-Null
try {
    & $python $script
    exit $LASTEXITCODE
}
finally {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
