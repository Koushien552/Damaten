param([string]$Config = "")

# Heartbeat watchdog for the self-play loop. Runs on a short repeating schedule
# (separate from "Damaten SelfPlay Forever") and restarts self-play when it has
# stopped making progress. This covers BOTH failure modes that a plain
# restart-on-exit cannot:
#   1. the supervisor process died and left orphaned/blocked HexAI workers, and
#   2. the supervisor is still alive but hung (e.g. Wait-Job never returns),
# which is what silently stalled the pipeline for ~3 days on 2026-06-13. In that
# stall the workers were alive but using ~0 CPU.
#
# Liveness is judged primarily by whether the self-play workers are still
# *burning CPU* (sampled over a few seconds), with the newest write under
# PartsDir as a secondary "recently alive" signal. CPU progress is the reliable
# signal because a single self-play game at high --iters can take a couple of
# minutes, so file writes legitimately pause for minutes (especially right after
# a resume), whereas a genuine hang shows no CPU movement at all.
#
# The watchdog stays out of the way while a local training run has paused
# self-play, because Stop-DamatenSelfPlay *disables* the task and we skip
# whenever the task is Disabled.

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\Damaten.Common.ps1"

$cfg = Read-DamatenConfig $Config
$spTask = "Damaten SelfPlay Forever"
$staleMinutes = if ($cfg.PSObject.Properties.Name -contains "WatchdogStaleMinutes" -and $cfg.WatchdogStaleMinutes) {
    [int]$cfg.WatchdogStaleMinutes
} else { 20 }

# Single-instance guard so overlapping fires never fight each other.
$wdLockPath = Join-Path $cfg.RuntimeDir "damaten_watchdog.lock"
$wdLock = $null
try {
    $wdLock = [System.IO.File]::Open($wdLockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
} catch {
    exit 0
}

function Get-DamatenSelfPlayWorkerPids {
    return @(Get-CimInstance Win32_Process -Filter "Name='HexAI.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*selfplay*' } |
        ForEach-Object { $_.ProcessId })
}

function Get-DamatenWorkerCpuSeconds {
    param([int[]]$Pids)
    $sum = 0.0
    foreach ($id in $Pids) {
        $p = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($p) { $sum += [double]$p.CPU }
    }
    return $sum
}

try {
    $task = Get-ScheduledTask -TaskName $spTask -ErrorAction SilentlyContinue
    if (-not $task) { exit 0 }
    if ($task.State -eq "Disabled") { exit 0 }   # paused by local training -> hands off

    # Secondary signal: newest write under PartsDir.
    $newest = Get-ChildItem -LiteralPath $cfg.PartsDir -Filter "cycle_*_part_*.tsv" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $lastWrite = if ($newest) { $newest.LastWriteTime } else { [datetime]::MinValue }
    $ageMin = ((Get-Date) - $lastWrite).TotalMinutes

    $pids = Get-DamatenSelfPlayWorkerPids
    $reason = $null

    if ($pids.Count -eq 0) {
        # No workers. Healthy only if a write happened very recently (the brief
        # merge / next-cycle-start gap); otherwise self-play is down.
        if ($ageMin -le $staleMinutes) { exit 0 }
        $reason = ("no self-play workers and no progress for {0:N1} min" -f $ageMin)
    } else {
        # Workers exist: are they actually computing? Sample CPU over a few seconds.
        $cpu1 = Get-DamatenWorkerCpuSeconds $pids
        Start-Sleep -Seconds 20
        $cpu2 = Get-DamatenWorkerCpuSeconds (Get-DamatenSelfPlayWorkerPids)
        $cpuDelta = $cpu2 - $cpu1
        if ($cpuDelta -gt 2.0) { exit 0 }                 # advancing -> healthy (covers slow first game)
        if ($ageMin -le $staleMinutes) { exit 0 }         # idle CPU but just wrote -> brief disk phase, ok
        $reason = ("workers not computing (CPU +{0:N1}s over 20s) and no progress for {1:N1} min" -f $cpuDelta, $ageMin)
    }

    Write-DamatenLog $cfg ("watchdog: self-play stalled ({0}); restarting" -f $reason)

    # Kill the (possibly hung) forever supervisor and any self-play workers.
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*Run-DamatenSelfPlayForever.ps1*' } |
        ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }
    Get-CimInstance Win32_Process -Filter "Name='HexAI.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*selfplay*' } |
        ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }
    Start-Sleep -Seconds 3

    # Clear a stale self-play lock if its holder is gone (open succeeds only when
    # no live process holds it exclusively).
    $selfLock = Join-Path $cfg.RuntimeDir "damaten_selfplay.lock"
    if (Test-Path -LiteralPath $selfLock) {
        try {
            $probe = [System.IO.File]::Open($selfLock, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
            $probe.Dispose()
            Remove-Item -LiteralPath $selfLock -Force -ErrorAction SilentlyContinue
        } catch {}
    }

    try { Stop-ScheduledTask -TaskName $spTask -ErrorAction SilentlyContinue } catch {}
    Start-ScheduledTask -TaskName $spTask
    Start-Sleep -Seconds 8
    $now = (Get-DamatenSelfPlayWorkerPids).Count
    Write-DamatenLog $cfg ("watchdog: restarted self-play task (workers now={0})" -f $now)
}
finally {
    if ($wdLock) { $wdLock.Dispose() }
}
