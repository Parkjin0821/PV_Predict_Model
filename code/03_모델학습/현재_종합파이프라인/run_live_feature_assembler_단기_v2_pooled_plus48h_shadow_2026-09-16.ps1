$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$assembler = (Get-ChildItem -LiteralPath $PSScriptRoot -Filter "live_feature_assembler_*v2_pooled_plus48h_2026-09-16.py" | Select-Object -First 1).FullName
if (-not $assembler) { throw "assembler not found" }

# +24h pooled 러너(2026-09-10)와 동일하게, stdout/stderr를 날짜별 로그로 남긴다
# (09-13 그 파일의 무로그 실패 사고 재발 방지 패턴 그대로 재사용).
$logDir = Join-Path $PSScriptRoot "logs\short_pooled48h_shadow"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ((Get-Date).ToString("yyyyMMdd") + ".log")

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

$exit = Invoke-NativeLogged -Executable $pythonExe -Arguments @("-X", "utf8", $assembler) -Label "assembler"
exit $exit
