$ErrorActionPreference = "Stop"

$pythonPath = "C:\Users\u-cube\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$scriptPath = Join-Path $PSScriptRoot "download_kma_public_asos.py"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python 실행파일을 찾지 못했습니다: $pythonPath"
}

& $pythonPath $scriptPath

if ($LASTEXITCODE -ne 0) {
    throw "기상청 환경자료 수집이 완료되지 않았습니다. 위 오류 내용을 확인하세요."
}

Write-Host "기상청 ASOS 환경자료 수집이 완료되었습니다." -ForegroundColor Green
