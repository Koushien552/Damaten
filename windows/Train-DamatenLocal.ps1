param(
    [string]$Config = "",
    [int]$Limit = 0,
    [int]$Epochs = 0,
    [int]$GateIters = 0,
    [switch]$NoPause,
    [switch]$NoPromote
)

# Local replacement for the Colab GPU trainer + 12:00 pull. Trains the HEXCNN_V1
# conv-ResNet on this machine with PyTorch-CPU, warm-started from the current
# model, then gates the candidate against the current model with an SPRT and
# promotes it only if it is measurably stronger. No GitHub, Colab, or LFS.

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\Damaten.Common.ps1"

$cfg = Read-DamatenConfig $Config

$py = [string]$cfg.PythonPath
$trainScript = [string]$cfg.TrainScript
$model = [string]$cfg.ModelPath
$cand = [string]$cfg.CandidateModelPath
$limit = if ($Limit -gt 0) { $Limit } else { [int]$cfg.TrainLimit }
$epochs = if ($Epochs -gt 0) { $Epochs } else { [int]$cfg.TrainEpochs }
$gateIters = if ($GateIters -gt 0) { $GateIters } else { [int]$cfg.GateIters }
$n = [int]$cfg.BoardSize

if (!(Test-Path -LiteralPath $trainScript)) { throw "trainer not found: $trainScript" }
if (!(Test-Path -LiteralPath $cfg.DataPath)) { throw "training data not found: $($cfg.DataPath)" }
& $py -c "import torch, numpy"
if ($LASTEXITCODE -ne 0) {
    throw "Python/torch not usable at: $py  (install: pip install torch numpy)"
}
$warmStart = Test-Path -LiteralPath $model

# Pick the warm-start source by architecture. The trainer keeps the shape of
# whatever --model-in it loads (ignoring --channels/--blocks), so warm-starting
# from a mismatched net would silently train the old shape forever. When the
# production model doesn't match the configured target (e.g. growing 32x4 ->
# 64x6), continue last night's candidate lineage instead so the bigger net
# improves cumulatively until it can pass the gate; with no matching file,
# train from scratch.
function Get-HexCnnArch([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path) -or !(Test-Path -LiteralPath $Path)) { return $null }
    $f = (Get-Content -LiteralPath $Path -TotalCount 1) -split '\s+'
    if ($f.Count -lt 4 -or $f[0] -ne "HEXCNN_V1") { return $null }
    return ,@([int]$f[2], [int]$f[3])
}
$wantC = [int]$cfg.TrainChannels
$wantB = [int]$cfg.TrainBlocks
$warmSrc = $null
foreach ($src in @($model, $cand)) {
    $arch = Get-HexCnnArch $src
    if ($arch -and $arch[0] -eq $wantC -and $arch[1] -eq $wantB) { $warmSrc = $src; break }
}

Write-DamatenLog $cfg ("local train start: limit={0} epochs={1} gateIters={2} target={3}ch x {4}blk warm-start={5}" -f $limit, $epochs, $gateIters, $wantC, $wantB, $(if ($warmSrc) { $warmSrc } else { "(scratch)" }))

$paused = $false
try {
    if (-not $NoPause -and $cfg.TrainPauseSelfplay -ne $false) {
        $paused = Stop-DamatenSelfPlay $cfg
    }

    # Train the candidate. Warm-start from the current model so each run
    # continues the lineage (same as the previous Colab loop did).
    $trainThreads = if ($cfg.TrainThreads) { [int]$cfg.TrainThreads } else { [Environment]::ProcessorCount }
    $env:OMP_NUM_THREADS = [string]$trainThreads
    $env:MKL_NUM_THREADS = [string]$trainThreads
    # Keep the nightly train (and its SPRT gate) from pegging this low-power CPU;
    # the child python/HexAI processes inherit this priority.
    try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch {}
    $trainArgs = @(
        $trainScript, "--n", $n,
        "--data", $cfg.DataPath, "--recent", "--limit", $limit,
        "--model-out", $cand,
        "--epochs", $epochs,
        "--channels", [int]$cfg.TrainChannels, "--blocks", [int]$cfg.TrainBlocks,
        "--device", "cpu"
    )
    if ($warmSrc) { $trainArgs += @("--model-in", $warmSrc) }
    & $py @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "training failed: exit $LASTEXITCODE" }
    if (!(Test-Path -LiteralPath $cand)) { throw "training produced no candidate: $cand" }
    Write-DamatenLog $cfg "local train: candidate saved -> $cand"

    if ($NoPromote) {
        Write-DamatenLog $cfg "promotion skipped (-NoPromote); candidate left at $cand"
        return
    }
    if (-not $warmStart) {
        if (Promote-DamatenModel $cfg $cand) {
            Write-DamatenLog $cfg "no previous model; adopted candidate as production model"
        }
        return
    }

    # Gate: SPRT candidate (new) vs current model (old). Promote only on PASS.
    & "$PSScriptRoot\Run-DamatenSprt.ps1" -Config $cfg.ConfigPath -NewModel $cand -OldModel $model -Iters $gateIters -Label "local-train-gate"
    $gate = $LASTEXITCODE
    switch ($gate) {
        0 {
            if (Promote-DamatenModel $cfg $cand) {
                Write-DamatenLog $cfg "candidate PASSED SPRT gate -> promoted to production model"
            }
        }
        3 { Write-DamatenLog $cfg "candidate FAILED SPRT gate (not stronger); keeping current model" }
        default { Write-DamatenLog $cfg "candidate gate INCONCLUSIVE (code $gate); keeping current model" }
    }
}
finally {
    if ($paused) { Start-DamatenSelfPlay $cfg }
    # Refresh the convergence report so it always reflects the latest gate.
    try {
        & $py "$PSScriptRoot\..\tools\track_convergence.py" --config $cfg.ConfigPath | Out-Null
        Write-DamatenLog $cfg "convergence report updated"
    } catch {
        Write-DamatenLog $cfg "convergence report update failed: $($_.Exception.Message)"
    }
}
