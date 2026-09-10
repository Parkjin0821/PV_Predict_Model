# 부안·김제·영광 NWP D+1+D+2 백필의 자정 자동 이어받기 실행기.
# 완료 issue_date는 SQLite run_status로 자동 건너뛰며, 일일 API 예산에
# 도달하면 정상 종료한다. 다음 자정에 같은 명령이 미완료 날짜부터 재개한다.

$ErrorActionPreference = 'Stop'

$projectRoot = 'C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19'
$pythonExe = 'C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$collector = Join-Path $projectRoot '01_데이터수집\기상청_운영\collect_kma_nwp_nc_d1d2_extended_v1_2026-09-09.py'
$lockPath = Join-Path $projectRoot '01_데이터수집\기상청_운영\kma_nwp_d1d2_backfill.lock'
$logDir = Join-Path $projectRoot '01_데이터수집\기상청_운영\logs'
$logPath = Join-Path $logDir ('kma_nwp_d1d2_resume_{0}.log' -f (Get-Date -Format 'yyyyMMdd'))

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 수동 실행 또는 이전 예약 실행이 살아 있으면 중복 호출을 하지 않는다.
if (Test-Path -LiteralPath $lockPath) {
    $previousPid = $null
    try { $previousPid = [int](Get-Content -LiteralPath $lockPath -Raw).Trim() } catch {}
    if ($previousPid -and (Get-Process -Id $previousPid -ErrorAction SilentlyContinue)) {
        "[$(Get-Date -Format s)] skip: existing collector PID=$previousPid" | Add-Content -LiteralPath $logPath
        exit 0
    }
    Remove-Item -LiteralPath $lockPath -Force
}

$lock = $null
try {
    $lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes("$PID")
    $lock.Write($bytes, 0, $bytes.Length)
    $lock.Flush()

    "[$(Get-Date -Format s)] start: midnight resume" | Add-Content -LiteralPath $logPath
    & $pythonExe $collector --all-regions --start-date 2024-08-25 --end-date 2026-08-04 --live --max-api-calls 19000 *>> $logPath
    $exitCode = $LASTEXITCODE
    "[$(Get-Date -Format s)] end: exit_code=$exitCode" | Add-Content -LiteralPath $logPath
    exit $exitCode
}
finally {
    if ($lock) { $lock.Dispose() }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
