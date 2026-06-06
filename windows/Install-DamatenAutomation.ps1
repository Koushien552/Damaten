param(
    [string]$RepoPath = (Split-Path -Parent $PSScriptRoot),
    [string]$HexRoot = "C:\path\to\hex-visual-studio-project",  # pass -HexRoot or edit before running
    [string]$ExePath = "",
    [string]$GitRemote = "https://github.com/Koushien552/Damaten.git",
    [string]$Branch = "main",
    [int]$BoardSize = 9,
    [int]$Jobs = 4,
    [int]$GamesPerCycle = 100,
    [int]$Iters = 1200,
    [string]$GitUserName = "Damaten Windows AI",
    [string]$GitUserEmail = "damaten-windows-ai@local",
    [switch]$SkipTasks
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $ExePath = Join-Path $HexRoot "visual_studio\x64\Release\HexAI.exe"
}

$RepoPath = (Resolve-Path -LiteralPath $RepoPath).Path
$RuntimeDir = $HexRoot
$ConfigPath = Join-Path $RepoPath "config\damaten.local.json"
$LogPath = Join-Path $RuntimeDir "damaten_windows.log"

if (!(Test-Path -LiteralPath $ExePath)) {
    throw "HexAI.exe not found: $ExePath"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ConfigPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RuntimeDir "selfplay_parts") | Out-Null

$machine = ($env:COMPUTERNAME -replace '[^A-Za-z0-9_.-]', '_')
$config = [ordered]@{
    RepoPath = $RepoPath
    Branch = $Branch
    GitRemote = $GitRemote
    HexRoot = $HexRoot
    ExePath = $ExePath
    RuntimeDir = $RuntimeDir
    DataPath = Join-Path $RuntimeDir "hex_selfplay.tsv"
    ModelPath = Join-Path $RuntimeDir "hex_model.nn"
    PartsDir = Join-Path $RuntimeDir "selfplay_parts"
    LogPath = $LogPath
    BoardSize = $BoardSize
    Jobs = $Jobs
    GamesPerCycle = $GamesPerCycle
    Iters = $Iters
    MachineId = $machine
    MinPartAgeMinutes = 5
    ModelSourceInRepo = "models/hex_model.nn"
}
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8

git -C $RepoPath remote remove origin 2>$null
git -C $RepoPath remote add origin $GitRemote
git -C $RepoPath checkout -B $Branch
git -C $RepoPath config user.name $GitUserName
git -C $RepoPath config user.email $GitUserEmail
git -C $RepoPath lfs install
git -C $RepoPath lfs track "*.nn" "*.tsv" "*.zip"

Write-Host "Config written: $ConfigPath"

if ($SkipTasks) {
    Write-Host "Skipped task registration."
    exit 0
}

$powerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$TaskWrapperDir = Join-Path $RepoPath "task_wrappers"
New-Item -ItemType Directory -Force -Path $TaskWrapperDir | Out-Null
$script:TaskRegistrationFailed = $false

function New-DamatenTaskWrapper {
    param(
        [string]$TaskName,
        [string]$Script
    )
    $safeName = ($TaskName -replace '[^A-Za-z0-9_.-]', '_')
    $wrapper = Join-Path $TaskWrapperDir "$safeName.cmd"
    $content = @"
@echo off
"$powerShell" -NoProfile -ExecutionPolicy Bypass -File "$Script" -Config "$ConfigPath"
"@
    Set-Content -LiteralPath $wrapper -Value $content -Encoding ASCII
    return $wrapper
}

function Register-DamatenTask {
    param(
        [string]$TaskName,
        [string]$Script,
        [string]$Schedule,
        [string]$Time = ""
    )
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Script`" -Config `"$ConfigPath`""
    $wrapper = New-DamatenTaskWrapper -TaskName $TaskName -Script $Script
    $action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments
    if ($Schedule -eq "ONLOGON") {
        $Trigger = New-ScheduledTaskTrigger -AtLogOn
    } else {
        $Trigger = New-ScheduledTaskTrigger -Daily -At $Time
    }
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 5)
    try {
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $Trigger -Settings $settings -Force | Out-Null
        Write-Host "Registered task: $TaskName"
    } catch {
        Write-Host "Register-ScheduledTask failed; falling back to schtasks.exe for $TaskName"
        $taskRun = "`"$wrapper`""
        $cmd = @("/Create", "/F", "/TN", $TaskName, "/TR", $taskRun, "/SC", $Schedule, "/RL", "LIMITED")
        if ($Schedule -eq "DAILY") {
            $cmd += @("/ST", $Time)
        }
        & schtasks.exe @cmd
        if ($LASTEXITCODE -ne 0) {
            Write-Host "schtasks.exe failed with exit code ${LASTEXITCODE} for $TaskName"
            $script:TaskRegistrationFailed = $true
            return
        }
        Write-Host "Registered task via schtasks.exe: $TaskName"
    }
}

function Install-DamatenStartupSupervisor {
    $startup = [Environment]::GetFolderPath("Startup")
    $wrapper = Join-Path $startup "Damaten Supervisor.cmd"
    $script = Join-Path $RepoPath "windows\Run-DamatenSupervisor.ps1"
    $content = @"
@echo off
"$powerShell" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "$script" -Config "$ConfigPath"
"@
    Set-Content -LiteralPath $wrapper -Value $content -Encoding ASCII
    Write-Host "Installed startup supervisor: $wrapper"
}

Register-DamatenTask `
    -TaskName "Damaten SelfPlay Forever" `
    -Script (Join-Path $RepoPath "windows\Run-DamatenSelfPlayForever.ps1") `
    -Schedule "ONLOGON"

Register-DamatenTask `
    -TaskName "Damaten Push Results 0830" `
    -Script (Join-Path $RepoPath "windows\Push-DamatenResults.ps1") `
    -Schedule "DAILY" `
    -Time "08:30"

Register-DamatenTask `
    -TaskName "Damaten Pull Model 1200" `
    -Script (Join-Path $RepoPath "windows\Pull-DamatenModel.ps1") `
    -Schedule "DAILY" `
    -Time "12:00"

if ($script:TaskRegistrationFailed) {
    Install-DamatenStartupSupervisor
    Write-Host ""
    Write-Host "Task Scheduler registration was blocked by policy/permissions."
    Write-Host "Installed a Startup-folder supervisor instead."
    Write-Host "It runs self-play continuously and triggers push after 08:30 and pull after 12:00 once per day."
}

Write-Host ""
Write-Host "Installed. To start immediately:"
Write-Host "Start-ScheduledTask -TaskName 'Damaten SelfPlay Forever'"
Write-Host "Or, if Task Scheduler was blocked:"
Write-Host "powershell -ExecutionPolicy Bypass -File `"$RepoPath\windows\Run-DamatenSupervisor.ps1`" -Config `"$ConfigPath`""
Write-Host ""
Write-Host "First GitHub upload, after your git credentials are ready:"
Write-Host "powershell -ExecutionPolicy Bypass -File `"$RepoPath\windows\Push-DamatenResults.ps1`" -Config `"$ConfigPath`""
