param([string]$Python = 'python', [string]$BuildDirectory = (Join-Path $env:TEMP 'IWG_PressGuard_build'))
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath 'pumpguard\model.json')) { throw 'Train the model before building.' }
$pageAsset = Join-Path $PSScriptRoot 'dashboard.html'
$modelAsset = Join-Path $PSScriptRoot 'pumpguard\model.json'
$demoAsset = Join-Path $PSScriptRoot 'demo'
New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
& $Python -m PyInstaller --noconfirm --clean --onefile --windowed --name PressGuard --distpath (Join-Path $PSScriptRoot 'portable') --workpath (Join-Path $BuildDirectory 'work') --specpath $BuildDirectory --add-data "$pageAsset;." --add-data "$modelAsset;pumpguard" --add-data "$demoAsset;demo" app.py
if ($LASTEXITCODE -ne 0) { throw 'EXE build failed.' }
