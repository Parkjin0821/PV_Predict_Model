$ErrorActionPreference = "Stop"
$dataDir = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$collector = Join-Path $PSScriptRoot "collect_kma_nwp_nc_d1d2_extended_v1_2026-09-09.py"
$statusGenerator = Join-Path $dataDir "generate_collection_status_v1_2026-09-02.py"
$logDir = Join-Path $PSScriptRoot "logs\kim_nc_d1d2_live"
$lockPath = Join-Path $logDir "running.lock"
$sharedLockDir = Join-Path (Split-Path -Parent $logDir) "kim_nwp_d1d2_db_write.lockdir"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ((Get-Date).ToString("yyyyMMdd") + ".log")

# 09-11: 최상단에서 예외가 나도 무로그로 사라지지 않게 trap 추가
# (같은 날 run_kma_nwp_nc_live_scheduled에서 BOM 없는 파일을 Windows
# PowerShell 5.1이 시스템 코드페이지로 잘못 읽어 파싱 자체가 깨지는
# 사고를 겪음 - trap은 파싱 오류는 못 잡지만 런타임 오류는 잡는다).
trap {
    $message = $_.Exception.Message
    "[$(Get-Date -Format s)] wrapper FAIL: $message" |
        Out-File -LiteralPath $logPath -Append -Encoding utf8 -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    exit 1
}

# ★09-15 추가(Claude, 재수정 - 최초 버전이 trap 등록 전에 있어 null.Trim()
# 런타임오류로 무로그 exit=1을 냄, 09-15 13:33 실측으로 발견·수정)★:
# 사용자가 KIM NC 전용 신규 authKey를 별도 활용신청 완료 - 이 수집기
# (NC +24h/+48h 확장)에만 새 키를 적용한다. 파이썬 쪽 override
# (collect_kma_nwp_d1d2_extended_v1_2026-09-08.py의 KMA_AUTH_KEY_BACKFILL
# 환경변수 체크)를 그대로 재사용 - 이 스크립트가 그 legacy 모듈을
# monkeypatch해서 쓰므로 별도 파이썬 수정 불필요. trap 등록 "이후"에
# 두고, Get-Content가 빈 파일에서 $null을 반환해도 안전하도록 null
# 체크를 먼저 한다. 키 값은 코드에 절대 안 넣음 - 사용자가 이 파일에
# 직접 붙여넣는다.
$ncAuthKeyFile = "C:\Users\u-cube\JIN\태양광 발전\nc_d1d2_authkey_local.txt"
if (Test-Path -LiteralPath $ncAuthKeyFile) {
    $ncKeyRaw = Get-Content -LiteralPath $ncAuthKeyFile -Raw -ErrorAction SilentlyContinue
    if ($null -ne $ncKeyRaw -and $ncKeyRaw.Trim()) {
        $env:KMA_AUTH_KEY_BACKFILL = $ncKeyRaw.Trim()
    }
}

if (Test-Path -LiteralPath $lockPath) {
    $lockItem = Get-Item -LiteralPath $lockPath -ErrorAction SilentlyContinue
    if ($null -eq $lockItem) { exit 0 }
    $age = (Get-Date) - $lockItem.LastWriteTime
    if ($age.TotalMinutes -lt 25) { exit 0 }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType File -Path $lockPath -Force | Out-Null

# 09-11: 이 수집기와 D1D2 백필(run_kma_nwp_d1d2_backfill_hourly)이
# 지역별로 같은 kma_nwp_d1d2_live.sqlite3에 쓴다 - 동시쓰기 충돌
# 방지용 공유 락(원자적 디렉터리 생성)을 여기도 대칭 적용한다.
$sharedLockAcquired = $false
try {
    New-Item -ItemType Directory -Path $sharedLockDir -ErrorAction Stop | Out-Null
    $sharedLockAcquired = $true
}
catch {
    try {
        $sharedItem = Get-Item -LiteralPath $sharedLockDir -ErrorAction SilentlyContinue
        if ($null -ne $sharedItem) {
            $sharedAge = (Get-Date) - $sharedItem.LastWriteTime
            if ($sharedAge.TotalMinutes -ge 20) {
                Remove-Item -LiteralPath $sharedLockDir -Recurse -Force -ErrorAction SilentlyContinue
                New-Item -ItemType Directory -Path $sharedLockDir -ErrorAction Stop | Out-Null
                $sharedLockAcquired = $true
            }
        }
    }
    catch {
        $sharedLockAcquired = $false
    }
}
if (-not $sharedLockAcquired) {
    "[$(Get-Date -Format s)] collector WAIT: 백필/NC라이브가 공유 DB 사용 중, 다음 30분 주기에 재시도" |
        Out-File -LiteralPath $logPath -Append -Encoding utf8
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    exit 0
}
@{ owner = "nc_d1d2_live"; started_at = (Get-Date).ToString("o") } |
    ConvertTo-Json | Out-File -LiteralPath (Join-Path $sharedLockDir "owner.json") -Encoding utf8

function Invoke-NativeLogged {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $stdoutPath = Join-Path $logDir ("{0}_{1}.stdout.tmp" -f $Label, $PID)
    $stderrPath = Join-Path $logDir ("{0}_{1}.stderr.tmp" -f $Label, $PID)
    try {
        "[$(Get-Date -Format s)] $Label start" | Out-File -LiteralPath $logPath -Append -Encoding utf8
        $quotedArgs = @($Arguments | ForEach-Object {
            if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
        })
        $proc = Start-Process -FilePath $Executable -ArgumentList $quotedArgs `
            -WindowStyle Hidden -Wait -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        if (Test-Path -LiteralPath $stdoutPath) {
            Get-Content -LiteralPath $stdoutPath -Raw -Encoding UTF8 -ErrorAction SilentlyContinue |
                Out-File -LiteralPath $logPath -Append -Encoding utf8
        }
        if ((Test-Path -LiteralPath $stderrPath) -and (Get-Item -LiteralPath $stderrPath).Length -gt 0) {
            "[$Label stderr]" | Out-File -LiteralPath $logPath -Append -Encoding utf8
            Get-Content -LiteralPath $stderrPath -Raw -Encoding UTF8 -ErrorAction SilentlyContinue |
                Out-File -LiteralPath $logPath -Append -Encoding utf8
        }
        "[$(Get-Date -Format s)] $Label end exit=$($proc.ExitCode)" |
            Out-File -LiteralPath $logPath -Append -Encoding utf8
        return $proc.ExitCode
    }
    finally {
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

try {
    # NC 발행시각을 확인하는 운영 수집. 80/80이면 수집기가 스스로 건너뛴다.
    $collectorExit = Invoke-NativeLogged -Executable $python `
        -Arguments @($collector, "--all-regions", "--live", "--max-api-calls", "2000") -Label "collector"
        # ★09-15 변경(Claude)★: 100→500. 이 수집기는 이제 전용 신규
        # authKey(nc_d1d2_authkey_local.txt)를 쓰므로 ASOS/GRID/NWP실시간/
        # 백필과 할당량을 더 이상 안 나눔 - 경쟁 우려 없이 상향(사용자 요청).
        # ★09-16 변경(Claude, 사용자 지적 - "광주만 실패 뜬다")★: 500→2000.
        # 매일 KIM NC 당일분 발행시각(13~14시)에 4지역이 한 실행에서 동시에
        # "file is not exist"(1콜)에서 "실제 값 수백건 필요"로 동시 전환되며
        # 500 예산을 나눠 쓰다가 REGIONS 순서상 마지막인 광주가 굶었다
        # (13:00 실행 실측: api_calls_used=500=limit, last_result=광주
        # ApiDailyBudgetReached). 사용자가 API 호출 현황 스크린샷으로 확인한
        # 실제 사용량은 4,717/20,000회(23.6%) - 500이라는 캡 자체가
        # 지나치게 보수적이었음(진짜 기상청 쿼터 문제 아니었음).
        # 4지역 동시 신규수신 최악의 경우도 2000이면 충분히 여유.
    $statusExit = Invoke-NativeLogged -Executable $python `
        -Arguments @($statusGenerator) -Label "status_generator"
    if ($collectorExit -ne 0) { exit $collectorExit }
    exit $statusExit
}
finally {
    if ($sharedLockAcquired) {
        Remove-Item -LiteralPath $sharedLockDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
