# Build the Damaten Hex engine (src/main.cpp) to WebAssembly for the static
# web UI in this folder. Produces docs/hexai.js, docs/hexai.wasm, docs/hexai.data
# (the .data bundles models/hex_model.nn at the virtual path /hex_model.nn).
#
#   powershell -ExecutionPolicy Bypass -File docs\build_wasm.ps1          # scalar (portable, default)
#   powershell -ExecutionPolicy Bypass -File docs\build_wasm.ps1 -Simd    # AVX2 -> WASM-SIMD (faster, experimental)
#
# Needs the Emscripten SDK; the script auto-activates it from C:\Users\ryama\emsdk
# if emcc is not already on PATH.
param(
    [switch]$Simd,
    [string]$EmsdkRoot = "C:\Users\ryama\emsdk"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot   # ...\Damaten

if (-not (Get-Command emcc -ErrorAction SilentlyContinue)) {
    $envPs = Join-Path $EmsdkRoot "emsdk_env.ps1"
    if (-not (Test-Path $envPs)) { throw "emcc not found and emsdk_env.ps1 missing at $envPs" }
    Write-Host "Activating Emscripten from $EmsdkRoot ..."
    . $envPs | Out-Null
}

$model = Join-Path $repo "models\hex_model.nn"
if (-not (Test-Path $model)) { throw "model not found: $model" }

$simdArgs = @()
if ($Simd) { $simdArgs = @("-msimd128", "-mavx2") }  # enables the __AVX2__ intrinsic path under WASM SIMD

Push-Location $repo
try {
    & emcc src/main.cpp -O3 -std=c++17 `
        -sMODULARIZE=1 -sEXPORT_NAME=HexAI `
        "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap" `
        "-sEXPORTED_FUNCTIONS=_hex_init,_hex_best_move,_hex_policy,_malloc,_free" `
        -sALLOW_MEMORY_GROWTH=1 `
        --preload-file "models/hex_model.nn@/hex_model.nn" `
        --no-entry `
        @simdArgs `
        -o docs/hexai.js
    if ($LASTEXITCODE -ne 0) { throw "emcc failed (exit $LASTEXITCODE)" }
    Write-Host "OK -> docs/hexai.js, docs/hexai.wasm, docs/hexai.data"
    Get-ChildItem (Join-Path $repo "docs") -Filter "hexai.*" | Select-Object Name, Length | Format-Table -AutoSize
}
finally { Pop-Location }
