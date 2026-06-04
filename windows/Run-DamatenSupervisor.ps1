param([string]$Config = "")

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\Damaten.Common.ps1"

$script:SupervisorConfigPath = if ([string]::IsNullOrWhiteSpace($Config)) { Get-DamatenDefaultConfigPath } else { $Config }
$cfg = Read-DamatenConfig $Config

$supervisorLockPath = Join-Path $cfg.RuntimeDir "damaten_supervisor.lock"
$supervisorLock = $null
try {
    $supervisorLock = [System.IO.File]::Open($supervisorLockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
} catch {
    Write-DamatenLog $cfg "Supervisor already appears to be running. Lock file: $supervisorLockPath"
    exit 0
}

function Start-DamatenSelfPlayIfNeeded {
    param([object]$Config)

    $selfLockPath = Join-Path $Config.RuntimeDir "damaten_selfplay.lock"
    if (Test-Path -LiteralPath $selfLockPath) {
        try {
            $probe = [System.IO.File]::Open($selfLockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
            $probe.Dispose()
            Remove-Item -LiteralPath $selfLockPath -Force -ErrorAction SilentlyContinue
        } catch {
            return
        }
    }

    $ps = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    $script = Join-Path $PSScriptRoot "Run-DamatenSelfPlayForever.ps1"
    Write-DamatenLog $Config "starting self-play worker"
    Start-Process -FilePath $ps -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $script,
        "-Config", $script:SupervisorConfigPath
    ) -WindowStyle Hidden | Out-Null
}

function Invoke-DamatenTimedJob {
    param(
        [object]$Config,
        [string]$Name,
        [string]$ScriptName,
        [string]$MarkerName
    )

    $markerDir = Join-Path $Config.RuntimeDir "damaten_markers"
    Ensure-DamatenDirectory $markerDir
    $date = Get-Date -Format "yyyy-MM-dd"
    $marker = Join-Path $markerDir "$date.$MarkerName.done"
    if (Test-Path -LiteralPath $marker) {
        return
    }

    $script = Join-Path $PSScriptRoot $ScriptName
    Write-DamatenLog $Config "timed job start: $Name"
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $script -Config $script:SupervisorConfigPath
        if ($LASTEXITCODE -ne 0) {
            throw "timed job exited with code ${LASTEXITCODE}"
        }
        Set-Content -LiteralPath $marker -Value (Get-Date).ToString("o") -Encoding UTF8
        Write-DamatenLog $Config "timed job done: $Name"
    } catch {
        Write-DamatenLog $Config "timed job failed: $Name ; $($_.Exception.Message)"
    }
}

function Test-DamatenScheduledTaskExists {
    param([string]$TaskName)
    $result = & schtasks.exe /Query /TN $TaskName 2>$null
    return $LASTEXITCODE -eq 0
}

Write-DamatenLog $cfg "Damaten supervisor started"
$pushTaskExists = Test-DamatenScheduledTaskExists "Damaten Push Results 0830"
$pullTaskExists = Test-DamatenScheduledTaskExists "Damaten Pull Model 1200"
Write-DamatenLog $cfg "scheduled push task exists=$pushTaskExists pull task exists=$pullTaskExists"

try {
    while ($true) {
        Start-DamatenSelfPlayIfNeeded $cfg

        $now = Get-Date
        $minutes = $now.Hour * 60 + $now.Minute
        if (!$pushTaskExists -and $minutes -ge (8 * 60 + 30)) {
            Invoke-DamatenTimedJob $cfg "push results 08:30" "Push-DamatenResults.ps1" "push0830"
        }
        if (!$pullTaskExists -and $minutes -ge (12 * 60)) {
            Invoke-DamatenTimedJob $cfg "pull model 12:00" "Pull-DamatenModel.ps1" "pull1200"
        }

        Start-Sleep -Seconds 60
    }
}
finally {
    if ($supervisorLock) { $supervisorLock.Dispose() }
    Write-DamatenLog $cfg "Damaten supervisor stopped"
}
