param([string]$Config = "")

# One-shot training cutoff. Stops the Damaten pipeline cleanly and finalizes:
#   1. disables the watchdog (so it cannot relaunch self-play mid-shutdown),
#   2. disables the daily-train task,
#   3. disables + stops self-play and kills its workers,
#   4. regenerates the final convergence report,
#   5. commits + pushes the final model so the Colab Play_vs_Damaten notebook
#      can use it,
#   6. removes the one-shot "Damaten Cutoff" scheduled task.
#
# The best model file is left in place; only the automation is turned off.
# Re-enable the tasks manually if you ever want to resume.

$ErrorActionPreference = "Continue"
. "$PSScriptRoot\Damaten.Common.ps1"
$cfg = Read-DamatenConfig $Config

Write-DamatenLog $cfg "==== DAMATEN CUTOFF: stopping the training pipeline ===="

# 1) watchdog first, 2) daily train, 3) self-play
schtasks /Change /TN "Damaten SelfPlay Watchdog" /DISABLE 2>$null | Out-Null
schtasks /Change /TN "Damaten Train Local 0300" /DISABLE 2>$null | Out-Null
schtasks /Change /TN "Damaten SelfPlay Forever" /DISABLE 2>$null | Out-Null
schtasks /End /TN "Damaten SelfPlay Forever" 2>$null | Out-Null

# Killing the workers makes the forever supervisor's Wait-Job fail, so it exits
# and releases its lock on its own (no CIM needed).
Get-Process HexAI -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5
$left = @(Get-Process HexAI -ErrorAction SilentlyContinue).Count
Write-DamatenLog $cfg "cutoff: tasks disabled; HexAI workers remaining=$left"

# 4) final convergence report
try {
    & $cfg.PythonPath "$PSScriptRoot\..\tools\track_convergence.py" --config $cfg.ConfigPath | Out-Null
    Write-DamatenLog $cfg "cutoff: final convergence report generated"
} catch {
    Write-DamatenLog $cfg "cutoff: report generation failed: $($_.Exception.Message)"
}

# 5) commit + push the final model for Colab
try {
    $repo = [string]$cfg.RepoPath
    Copy-Item -LiteralPath $cfg.ModelPath -Destination (Join-Path $repo "models\hex_model.nn") -Force
    & git -C $repo add "models/hex_model.nn"
    & git -C $repo commit -m "Final model at training cutoff" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
    if ($LASTEXITCODE -eq 0) {
        & git -C $repo push origin ([string]$cfg.Branch)
        if ($LASTEXITCODE -eq 0) {
            Write-DamatenLog $cfg "cutoff: final model committed and pushed"
        } else {
            Write-DamatenLog $cfg "cutoff: push FAILED (exit $LASTEXITCODE); model committed locally - push manually"
        }
    } else {
        Write-DamatenLog $cfg "cutoff: no model change to commit (or commit failed)"
    }
} catch {
    Write-DamatenLog $cfg "cutoff: model push step error: $($_.Exception.Message)"
}

Write-DamatenLog $cfg "==== DAMATEN CUTOFF complete. Training stopped; final model retained. ===="

# 6) remove this one-shot task
schtasks /Delete /TN "Damaten Cutoff" /F 2>$null | Out-Null
