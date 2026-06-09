param([string]$Config = "")

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\Damaten.Common.ps1"

$cfg = Read-DamatenConfig $Config

Invoke-DamatenGit $cfg @("pull", "--rebase", "--autostash", "origin", $cfg.Branch)
Invoke-DamatenGit $cfg @("lfs", "pull")

$modelInRepo = Join-Path $cfg.RepoPath $cfg.ModelSourceInRepo
if (!(Test-Path -LiteralPath $modelInRepo)) {
    Write-DamatenLog $cfg "trained model not found in GitHub repo yet: $modelInRepo"
    exit 0
}

# Preserve the model currently in use as the "old" model before overwriting it,
# so the freshly trained model can be measured against it after the swap.
$hadPrevious = $false
$previousHash = $null
if (Test-Path -LiteralPath $cfg.ModelPath) {
    $previousHash = (Get-FileHash -LiteralPath $cfg.ModelPath -Algorithm SHA256).Hash
    Copy-DamatenAtomic -Source $cfg.ModelPath -Destination $cfg.PrevModelPath
    $hadPrevious = $true
}

Copy-DamatenAtomic -Source $modelInRepo -Destination $cfg.ModelPath
Write-DamatenLog $cfg "received trained model: $modelInRepo -> $($cfg.ModelPath)"

# Daily new-vs-old strength check. Best-effort: a failure here must never block
# the model swap (self-play already picks up the new model from ModelPath).
if ($cfg.EvalEnabled -eq $false) {
    Write-DamatenLog $cfg "eval disabled (EvalEnabled=false); skipping new-vs-old match"
    exit 0
}
if (-not $hadPrevious) {
    Write-DamatenLog $cfg "no previous model to compare against; skipping new-vs-old match"
    exit 0
}
$newHash = (Get-FileHash -LiteralPath $cfg.ModelPath -Algorithm SHA256).Hash
if ($newHash -eq $previousHash) {
    Write-DamatenLog $cfg "model unchanged since last pull (identical hash); skipping new-vs-old match"
    exit 0
}

try {
    if ($cfg.EvalMode -eq "sprt") {
        & "$PSScriptRoot\Run-DamatenSprt.ps1" -Config $Config -NewModel $cfg.ModelPath -OldModel $cfg.PrevModelPath -Label "daily-pull"
    } else {
        & "$PSScriptRoot\Eval-DamatenModels.ps1" -Config $Config -NewModel $cfg.ModelPath -OldModel $cfg.PrevModelPath -Label "daily-pull"
    }
} catch {
    Write-DamatenLog $cfg "new-vs-old eval failed (non-fatal): $($_.Exception.Message)"
}
