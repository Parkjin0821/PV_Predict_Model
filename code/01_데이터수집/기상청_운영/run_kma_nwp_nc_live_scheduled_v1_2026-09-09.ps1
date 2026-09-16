$ErrorActionPreference = "Stop"
$dataDir = Split-Path -Parent $PSScriptRoot
$project = Split-Path -Parent $dataDir
$python = "C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$collector = Join-Path $PSScriptRoot "collect_kma_nwp_nc_live_v1_2026-09-09.py"
$statusGenerator = Join-Path $dataDir "generate_collection_status_v1_2026-09-02.py"
$logDir = Join-Path $PSScriptRoot "logs\kim_nc_live"
$lockPath = Join-Path $logDir "running.lock"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ((Get-Date).ToString("yyyyMMdd") + ".log")

# Log every unhandled wrapper error, including pre-collector lock failures.
trap {
    $message = $_.Exception.Message
    "[$(Get-Date -Format s)] wrapper FAIL: $message" |
        Out-File -LiteralPath $logPath -Append -Encoding utf8 -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    exit 1
}

if (Test-Path -LiteralPath $lockPath) {
    $lockItem = Get-Item -LiteralPath $lockPath -ErrorAction SilentlyContinue
    if ($null -eq $lockItem) { exit 0 }
    $age = (Get-Date) - $lockItem.LastWriteTime
    if ($age.TotalMinutes -lt 25) { exit 0 }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType File -Path $lockPath -Force | Out-Null

# 09-11 정정: 이 스크립트(collect_kma_nwp_nc_live_v1_2026-09-09.py)는
# kma_live_inputs.sqlite3에 쓴다 - 백필/NC-D1D2-live가 쓰는
# kma_nwp_d1d2_live.sqlite3와는 완전히 다른 파일이라 원래 공유락이
# 불필요했다(grep으로 직접 확인: collect_kma_nwp_nc_live_v1_2026-09-09.py
# 에는 "kma_live_inputs.sqlite3"만 나오고 "kma_nwp_d1d2_live"는 안 나옴).
# 이 락을 여기 걸어놨더니 같은 :00·:30 스케줄인
# UCUBE_KIM_NC_D1D2_Live_30min과 매번 경합해서 11:21~15:00까지 4시간
# 가까이 단 한 번도 못 뚫고 계속 밀렸다(로그 실측 12회 연속 "busy").
# 실제 공유 대상(백필 vs NC-D1D2-live)만 락을 걸고, 이 스크립트는 락 제거.

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
    $collectorExit = Invoke-NativeLogged -Executable $python `
        -Arguments @($collector, "--all-regions") -Label "collector"
    $statusExit = Invoke-NativeLogged -Executable $python `
        -Arguments @($statusGenerator) -Label "status_generator"
    if ($collectorExit -ne 0) { exit $collectorExit }
    exit $statusExit
}
finally {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
