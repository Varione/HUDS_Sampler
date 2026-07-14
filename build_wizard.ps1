# Build script for HUDS_Wizard
# Usage: powershell -ExecutionPolicy Bypass -File build_wizard.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = "D:\miniconda\envs\agents\python.exe"

Write-Output "=== Step 1: PyInstaller build ==="
& $Python -m PyInstaller (Join-Path $ProjectRoot "HUDS_Wizard.spec") --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

Write-Output "`n=== Step 2: Post-build torch source restore ==="
& $Python (Join-Path $ProjectRoot "post_build_torch.py")
if ($LASTEXITCODE -ne 0) { throw "Post-build script failed" }

Write-Output "`n=== Build complete ==="
Write-Output "Output: $(Join-Path $ProjectRoot 'dist\HUDS_Wizard\HUDS_Wizard.exe')"
