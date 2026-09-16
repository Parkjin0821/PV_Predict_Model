# 부안·김제·영광 라이브연계 결합본(JOIN_PARQUET·CSV) 매일 자동 리빌드.
# 09-14 이후 코덱스 용량저하로 리빌드가 18시간+ 정지된 것을 09-15에
# 발견해, 기존 검증된 생성기를 그대로 순서대로 실행하는 예약작업으로
# 등록함(재구현 없음). 부안은 1단계(시간집계)→2단계(결합) 순서를 반드시
# 지켜야 하므로 한 파일 안에서 순차 실행한다. 한 지역이 실패해도 나머지
# 지역은 계속 진행한다(예외를 로그에 남기고 다음으로 넘어감).

$ErrorActionPreference = "Stop"
$python = "C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$root = "C:\Users\u-cube\JIN\태양광 발전\광주_PV_예측모델_통합_v1_2026-08-19\02_전처리"
$logDir = Join-Path $root "logs\live_bridge_rebuild_daily"
$lockPath = Join-Path $logDir "running.lock"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

if (Test-Path -LiteralPath $lockPath) {
    $age = (Get-Date) - (Get-Item -LiteralPath $lockPath).LastWriteTime
    if ($age.TotalMinutes -lt 30) { exit 0 }
    Remove-Item -LiteralPath $lockPath -Force
}
New-Item -ItemType File -Path $lockPath -Force | Out-Null

$logPath = Join-Path $logDir ((Get-Date -Format "yyyyMMdd") + ".log")
$stamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz")
Add-Content -LiteralPath $logPath -Encoding utf8 -Value ""
Add-Content -LiteralPath $logPath -Encoding utf8 -Value "=== $stamp ==="

function Run-Step($name, $scriptPath, $scriptArgs) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value "--- $name 시작 ---"
    try {
        $out = & $python $scriptPath @scriptArgs 2>&1 | Out-String
        Add-Content -LiteralPath $logPath -Encoding utf8 -Value $out
        Add-Content -LiteralPath $logPath -Encoding utf8 -Value "--- $name 종료(exit=$LASTEXITCODE) ---"
    } catch {
        Add-Content -LiteralPath $logPath -Encoding utf8 -Value "--- $name 예외: $($_.Exception.Message) ---"
    }
}

try {
    Run-Step "부안 1단계(시간집계)" (Join-Path $root "부안\buan_live_bridge_v1_2026-09-14.py") @()
    Run-Step "부안 2단계(결합)"     (Join-Path $root "부안\build_buan_weather_power_live_bridge_v1_2026-09-14.py") @()
    Run-Step "김제(결합)"          (Join-Path $root "build_regional_weather_power_live_bridge_v1_2026-09-14.py") @("--region","김제")
    Run-Step "영광(결합)"          (Join-Path $root "build_regional_weather_power_live_bridge_v1_2026-09-14.py") @("--region","영광")
}
finally {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
