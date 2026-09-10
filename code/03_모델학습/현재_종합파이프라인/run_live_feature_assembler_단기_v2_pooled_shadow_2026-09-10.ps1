$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\u-cube\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$assembler = (Get-ChildItem -LiteralPath $PSScriptRoot -Filter "live_feature_assembler_*v2_pooled_2026-09-10.py" | Select-Object -First 1).FullName
if (-not $assembler) { throw "assembler not found" }

& $pythonExe -X utf8 $assembler
exit $LASTEXITCODE
