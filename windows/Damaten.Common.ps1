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
    if (!$cfg.ConfigPath) { $cfg | Add-Member -NotePropertyName ConfigPath -NotePropertyValue (Resolve-Path -LiteralPath $Config).Path }
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

    # Local CNN training defaults (Plan B: replace the Colab GPU trainer with a
    # local PyTorch-CPU run). Derived so existing configs keep working.
    if (!$cfg.PythonPath) {
        $probe = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
        $pyDefault = if (Test-Path -LiteralPath $probe) { $probe } else { "python" }
        $cfg | Add-Member -NotePropertyName PythonPath -NotePropertyValue $pyDefault
    }
    if (!$cfg.TrainScript -and $cfg.RepoPath) {
        $cfg | Add-Member -NotePropertyName TrainScript -NotePropertyValue (Join-Path $cfg.RepoPath "colab\colab_train_cnn.py")
    }
    if ($cfg.ModelPath -and !$cfg.CandidateModelPath) {
        $modelDir = Split-Path -Parent $cfg.ModelPath
        $modelBase = [System.IO.Path]::GetFileNameWithoutExtension($cfg.ModelPath)
        $modelExt = [System.IO.Path]::GetExtension($cfg.ModelPath)
        $cfg | Add-Member -NotePropertyName CandidateModelPath -NotePropertyValue (Join-Path $modelDir ("{0}.candidate{1}" -f $modelBase, $modelExt))
    }
    if (!$cfg.TrainLimit) { $cfg | Add-Member -NotePropertyName TrainLimit -NotePropertyValue 300000 }
    if (!$cfg.TrainEpochs) { $cfg | Add-Member -NotePropertyName TrainEpochs -NotePropertyValue 6 }
    if (!$cfg.TrainChannels) { $cfg | Add-Member -NotePropertyName TrainChannels -NotePropertyValue 32 }
    if (!$cfg.TrainBlocks) { $cfg | Add-Member -NotePropertyName TrainBlocks -NotePropertyValue 4 }
    if ($null -eq $cfg.TrainPauseSelfplay) { $cfg | Add-Member -NotePropertyName TrainPauseSelfplay -NotePropertyValue $true }
    if (!$cfg.GateIters) { $cfg | Add-Member -NotePropertyName GateIters -NotePropertyValue 400 }
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

# Stop the forever self-play loop and its worker processes so a CPU-heavy local
# training run can use all cores. Excludes the calling process. Returns $true
# when no self-play workers remain.
function Stop-DamatenSelfPlay {
    param([object]$Config)
    # If self-play runs as a scheduled task, disable it first so its daily
    # watchdog trigger cannot relaunch self-play in the middle of training,
    # then stop the running instance.
    $spTask = "Damaten SelfPlay Forever"
    if (Get-ScheduledTask -TaskName $spTask -ErrorAction SilentlyContinue) {
        try { Disable-ScheduledTask -TaskName $spTask -ErrorAction Stop | Out-Null } catch {}
        try { Stop-ScheduledTask -TaskName $spTask -ErrorAction Stop } catch {}
    }
    $self = $PID
    $sup = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.ProcessId -ne $self -and $_.CommandLine -like '*Run-DamatenSelfPlayForever.ps1*' }
    foreach ($s in $sup) {
        try { Stop-Process -Id $s.ProcessId -Force -ErrorAction Stop; Write-DamatenLog $Config "selfplay pause: stopped supervisor pid=$($s.ProcessId)" } catch {}
    }
    Start-Sleep -Seconds 2
    $workers = Get-CimInstance Win32_Process -Filter "Name='HexAI.exe'" |
        Where-Object { $_.CommandLine -like '*selfplay*' }
    foreach ($w in $workers) { try { Stop-Process -Id $w.ProcessId -Force -ErrorAction Stop } catch {} }
    Start-Sleep -Seconds 2
    $remain = @(Get-CimInstance Win32_Process -Filter "Name='HexAI.exe'" | Where-Object { $_.CommandLine -like '*selfplay*' }).Count
    Write-DamatenLog $Config "selfplay paused: workers remaining=$remain"
    return ($remain -eq 0)
}

# Resume the forever self-play loop. Prefer the scheduled task (durable: it runs
# under Task Scheduler and survives the caller exiting); fall back to a detached
# process only if the task is not registered.
function Start-DamatenSelfPlay {
    param([object]$Config)
    $spTask = "Damaten SelfPlay Forever"
    if (Get-ScheduledTask -TaskName $spTask -ErrorAction SilentlyContinue) {
        try { Enable-ScheduledTask -TaskName $spTask -ErrorAction Stop | Out-Null } catch {}
        try { Start-ScheduledTask -TaskName $spTask -ErrorAction Stop; Write-DamatenLog $Config "selfplay resume: started scheduled task '$spTask'" }
        catch { Write-DamatenLog $Config "selfplay resume: Start-ScheduledTask failed: $($_.Exception.Message)" }
    } else {
        $psh = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
        $script = Join-Path $Config.RepoPath "windows\Run-DamatenSelfPlayForever.ps1"
        $cfgPath = if ($Config.ConfigPath) { $Config.ConfigPath } else { Get-DamatenDefaultConfigPath }
        Start-Process -FilePath $psh -WindowStyle Hidden -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden',
            '-File', $script, '-Config', $cfgPath
        )
        Write-DamatenLog $Config "selfplay resume: launched detached process (no task registered)"
    }
    Start-Sleep -Seconds 8
    $workers = @(Get-CimInstance Win32_Process -Filter "Name='HexAI.exe'" | Where-Object { $_.CommandLine -like '*selfplay*' }).Count
    Write-DamatenLog $Config "selfplay resumed: workers=$workers"
}

# Atomically replace the production model with a new file, backing up the
# current one to PrevModelPath first. Retries on a transient sharing violation
# (a self-play worker briefly reading the model at cycle start).
function Promote-DamatenModel {
    param(
        [object]$Config,
        [string]$Candidate
    )
    $model = [string]$Config.ModelPath
    if (Test-Path -LiteralPath $model) {
        Copy-DamatenAtomic -Source $model -Destination $Config.PrevModelPath
    }
    $staged = "$model.promote"
    Copy-Item -LiteralPath $Candidate -Destination $staged -Force
    for ($i = 1; $i -le 5; $i++) {
        try { Move-Item -LiteralPath $staged -Destination $model -Force -ErrorAction Stop; return $true }
        catch { Start-Sleep -Seconds 2 }
    }
    Remove-Item -LiteralPath $staged -Force -ErrorAction SilentlyContinue
    Write-DamatenLog $Config "promote failed: model file busy after retries: $model"
    return $false
}
