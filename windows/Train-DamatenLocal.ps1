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

Write-DamatenLog $cfg ("local train start: limit={0} epochs={1} gateIters={2} warm-start={3}" -f $limit, $epochs, $gateIters, $warmStart)

$paused = $false
try {
    if (-not $NoPause -and $cfg.TrainPauseSelfplay -ne $false) {
        $paused = Stop-DamatenSelfPlay $cfg
    }

    # Train the candidate. Warm-start from the current model so each run
    # continues the lineage (same as the previous Colab loop did).
    $env:OMP_NUM_THREADS = [string][Environment]::ProcessorCount
    $env:MKL_NUM_THREADS = [string][Environment]::ProcessorCount
    $trainArgs = @(
        $trainScript, "--n", $n,
        "--data", $cfg.DataPath, "--recent", "--limit", $limit,
        "--model-out", $cand,
        "--epochs", $epochs,
        "--channels", [int]$cfg.TrainChannels, "--blocks", [int]$cfg.TrainBlocks,
        "--device", "cpu"
    )
    if ($warmStart) { $trainArgs += @("--model-in", $model) }
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
}
