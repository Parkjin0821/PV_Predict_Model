$ErrorActionPreference = "Stop"
$python = "C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$script = Join-Path $PSScriptRoot "recheck_blockdata_plant_info_weekly_v1_2026-09-09.py"
$statusGenerator = "C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\01_데이터수집\generate_collection_status_v1_2026-09-02.py"
$logDir = Join-Path $PSScriptRoot "logs\blockdata_plant_recheck_task"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ((Get-Date).ToString("yyyyMMdd_HHmmss") + ".log")

& $python $script 2>&1 | Out-File -LiteralPath $logPath -Append -Encoding utf8
$exit = $LASTEXITCODE
& $python $statusGenerator 2>&1 | Out-File -LiteralPath $logPath -Append -Encoding utf8
exit $exit
