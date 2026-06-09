$script:DamatenRepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Get-DamatenDefaultConfigPath {
    return Join-Path $script:DamatenRepoRoot "config\damaten.local.json"
}

function Read-DamatenConfig {
    param([string]$Config)

    if ([string]::IsNullOrWhiteSpace($Config)) {
        $Config = Get-DamatenDefaultConfigPath
    }
    if (!(Test-Path -LiteralPath $Config)) {
        throw "Config file not found: $Config. Run windows\Install-DamatenAutomation.ps1 first."
    }

    $cfg = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
    if (!$cfg.RepoPath) { $cfg | Add-Member -NotePropertyName RepoPath -NotePropertyValue $script:DamatenRepoRoot }
    if (!$cfg.Branch) { $cfg | Add-Member -NotePropertyName Branch -NotePropertyValue "main" }
    if (!$cfg.MachineId) {
        $machine = ($env:COMPUTERNAME -replace '[^A-Za-z0-9_.-]', '_')
        $cfg | Add-Member -NotePropertyName MachineId -NotePropertyValue $machine
    }
    if (!$cfg.MinPartAgeMinutes) { $cfg | Add-Member -NotePropertyName MinPartAgeMinutes -NotePropertyValue 5 }

    # New-vs-old evaluation defaults. Derived so existing config files keep
    # working without re-running the installer.
    if ($cfg.ModelPath) {
        $modelDir = Split-Path -Parent $cfg.ModelPath
        if (!$cfg.PrevModelPath) {
            $modelBase = [System.IO.Path]::GetFileNameWithoutExtension($cfg.ModelPath)
            $modelExt = [System.IO.Path]::GetExtension($cfg.ModelPath)
            $prev = Join-Path $modelDir ("{0}.prev{1}" -f $modelBase, $modelExt)
            $cfg | Add-Member -NotePropertyName PrevModelPath -NotePropertyValue $prev
        }
        if (!$cfg.EvalHistoryPath) {
            $cfg | Add-Member -NotePropertyName EvalHistoryPath -NotePropertyValue (Join-Path $modelDir "eval_history.csv")
        }
        if (!$cfg.SprtHistoryPath) {
            $cfg | Add-Member -NotePropertyName SprtHistoryPath -NotePropertyValue (Join-Path $modelDir "sprt_history.csv")
        }
    }
    if (!$cfg.EvalGames) { $cfg | Add-Member -NotePropertyName EvalGames -NotePropertyValue 100 }
    if (!$cfg.EvalIters) {
        $evalIters = if ($cfg.Iters) { [int]$cfg.Iters } else { 1200 }
        $cfg | Add-Member -NotePropertyName EvalIters -NotePropertyValue $evalIters
    }
    if ($null -eq $cfg.EvalEnabled) { $cfg | Add-Member -NotePropertyName EvalEnabled -NotePropertyValue $true }
    # EvalMode: "fixed" (play EvalGames and report a CI) or "sprt" (play until
    # the sequential test decides or SprtMaxGames is hit).
    if (!$cfg.EvalMode) { $cfg | Add-Member -NotePropertyName EvalMode -NotePropertyValue "fixed" }

    # SPRT defaults. H0 = +SprtElo0, H1 = +SprtElo1 Elo for the new model.
    if ($null -eq $cfg.SprtElo0) { $cfg | Add-Member -NotePropertyName SprtElo0 -NotePropertyValue 0 }
    if (!$cfg.SprtElo1) { $cfg | Add-Member -NotePropertyName SprtElo1 -NotePropertyValue 30 }
    if (!$cfg.SprtAlpha) { $cfg | Add-Member -NotePropertyName SprtAlpha -NotePropertyValue 0.05 }
    if (!$cfg.SprtBeta) { $cfg | Add-Member -NotePropertyName SprtBeta -NotePropertyValue 0.05 }
    if (!$cfg.SprtBatch) { $cfg | Add-Member -NotePropertyName SprtBatch -NotePropertyValue 8 }
    if (!$cfg.SprtMaxGames) { $cfg | Add-Member -NotePropertyName SprtMaxGames -NotePropertyValue 400 }
    return $cfg
}

function Ensure-DamatenDirectory {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Write-DamatenLog {
    param(
        [object]$Config,
        [string]$Message
    )
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    if ($Config -and $Config.LogPath) {
        Ensure-DamatenDirectory (Split-Path -Parent $Config.LogPath)
        Add-Content -LiteralPath $Config.LogPath -Value $line -Encoding UTF8
    }
}

function Invoke-DamatenGit {
    param(
        [object]$Config,
        [string[]]$Arguments
    )
    $repo = [string]$Config.RepoPath
    $display = "git -C `"$repo`" " + ($Arguments -join " ")
    Write-DamatenLog $Config $display
    & git -C $repo @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git failed with exit code ${LASTEXITCODE}: $display"
    }
}

function Ensure-DamatenDataHeader {
    param([object]$Config)
    Ensure-DamatenDirectory (Split-Path -Parent $Config.DataPath)
    if (!(Test-Path -LiteralPath $Config.DataPath) -or (Get-Item -LiteralPath $Config.DataPath).Length -eq 0) {
        Set-Content -LiteralPath $Config.DataPath -Value "# HEXSELFPLAY_V1" -Encoding UTF8
        Add-Content -LiteralPath $Config.DataPath -Value "# n`tplayer`tboard`tpolicy`twinner`tvalue" -Encoding UTF8
    }
}

function Get-DamatenNextCycle {
    param([object]$Config)
    Ensure-DamatenDirectory $Config.PartsDir
    $max = 0
    Get-ChildItem -LiteralPath $Config.PartsDir -Filter "cycle_*_part_*.tsv" -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.Name -match '^cycle_(\d+)_part_') {
                $max = [Math]::Max($max, [int]$Matches[1])
            }
        }
    return $max + 1
}

function Copy-DamatenAtomic {
    param(
        [string]$Source,
        [string]$Destination
    )
    Ensure-DamatenDirectory (Split-Path -Parent $Destination)
    $tmp = "$Destination.download"
    Copy-Item -LiteralPath $Source -Destination $tmp -Force
    Move-Item -LiteralPath $tmp -Destination $Destination -Force
}
