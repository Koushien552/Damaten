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

Copy-DamatenAtomic -Source $modelInRepo -Destination $cfg.ModelPath
Write-DamatenLog $cfg "received trained model: $modelInRepo -> $($cfg.ModelPath)"
