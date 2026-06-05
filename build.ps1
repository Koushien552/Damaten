# Build the HexAI engine with MSVC.
#
#   powershell -ExecutionPolicy Bypass -File .\build.ps1
#
# Produces .\build\hexai.exe from src\main.cpp.
#
# Flags:
#   /O2        optimize
#   /arch:AVX2 vectorize the CNN convolutions (requires an AVX2-capable CPU,
#              i.e. Intel Haswell 2013+/AMD Excavator+). With the default
#              /fp:precise this does NOT change results: the MLP path stays
#              bit-identical and the CNN output is unchanged.
#   /std:c++17 /EHsc /W4
#
# Pass -NoAvx2 to build a portable binary without AVX2 (slower CNN inference).

param(
    [string]$Source = (Join-Path $PSScriptRoot "src\main.cpp"),
    [string]$Out    = (Join-Path $PSScriptRoot "build\hexai.exe"),
    [switch]$NoAvx2
)

$ErrorActionPreference = "Stop"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere not found. Install Visual Studio with the C++ workload."
}
$vsPath = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw "No Visual Studio with the C++ (VC) tools was found."
}
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars"
}

$outDir = Split-Path -Parent $Out
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }

$arch = if ($NoAvx2) { "" } else { "/arch:AVX2" }
$flags = "/nologo /std:c++17 /O2 $arch /EHsc /W4"

Write-Host "Compiling $Source -> $Out"
Write-Host "Flags: $flags"
Push-Location $outDir
try {
    cmd /c "`"$vcvars`" >nul 2>&1 && cl $flags /Fe:`"$Out`" `"$Source`""
    if ($LASTEXITCODE -ne 0) { throw "compile failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
}

if (Test-Path $Out) {
    Write-Host "OK: $Out"
} else {
    throw "build did not produce $Out"
}
