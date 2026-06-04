param([string]$Config = "")

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\Damaten.Common.ps1"

$cfg = Read-DamatenConfig $Config

if (!(Test-Path -LiteralPath $cfg.ExePath)) {
    throw "HexAI.exe not found: $($cfg.ExePath)"
}

Ensure-DamatenDirectory $cfg.RuntimeDir
Ensure-DamatenDirectory $cfg.PartsDir
Ensure-DamatenDataHeader $cfg

$lockPath = Join-Path $cfg.RuntimeDir "damaten_selfplay.lock"
$lock = $null
try {
    $lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
} catch {
    Write-DamatenLog $cfg "Self-play already appears to be running. Lock file: $lockPath"
    exit 0
}

$gamesPerJob = [Math]::Ceiling([double]$cfg.GamesPerCycle / [double]$cfg.Jobs)
$cycle = Get-DamatenNextCycle $cfg
Write-DamatenLog $cfg "start self-play forever: cycle=$cycle n=$($cfg.BoardSize) jobs=$($cfg.Jobs) gamesPerCycle=$($cfg.GamesPerCycle) iters=$($cfg.Iters)"

try {
    while ($true) {
        Write-DamatenLog $cfg "cycle $cycle : parallel selfplay"

        $running = 1..([int]$cfg.Jobs) | ForEach-Object {
            $i = $_
            $out = Join-Path $cfg.PartsDir ("cycle_{0:D6}_part_{1:D2}.tsv" -f $cycle, $i)
            $seed = 300000000 + $cycle * 100 + $i
            if (Test-Path -LiteralPath $out) {
                Remove-Item -LiteralPath $out -Force
            }

            Start-Job -ScriptBlock {
                param($exe, $root, $out, $n, $games, $iters, $seed, $model)
                Set-Location -LiteralPath $root
                if (Test-Path -LiteralPath $model) {
                    & $exe selfplay --n $n --games $games --iters $iters --out $out --model $model --seed $seed
                } else {
                    & $exe selfplay --n $n --games $games --iters $iters --out $out --seed $seed
                }
                if ($LASTEXITCODE -ne 0) {
                    throw "selfplay failed with exit code $LASTEXITCODE"
                }
            } -ArgumentList $cfg.ExePath, $cfg.RuntimeDir, $out, $cfg.BoardSize, $gamesPerJob, $cfg.Iters, $seed, $cfg.ModelPath
        }

        $running | Wait-Job
        $failed = $running | Where-Object { $_.State -ne "Completed" }
        $running | Receive-Job
        $running | Remove-Job
        if ($failed) {
            throw "one or more selfplay jobs failed at cycle $cycle"
        }

        Write-DamatenLog $cfg "cycle $cycle : merge local data"
        $added = 0
        Get-ChildItem -LiteralPath $cfg.PartsDir -Filter ("cycle_{0:D6}_part_*.tsv" -f $cycle) |
            ForEach-Object {
                Get-Content -LiteralPath $_.FullName |
                    Where-Object { $_ -and ($_ -notmatch '^#') } |
                    ForEach-Object {
                        Add-Content -LiteralPath $cfg.DataPath -Value $_
                        $script:added++
                    }
            }

        Write-DamatenLog $cfg "cycle $cycle : done"
        $cycle++
    }
}
finally {
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
    if ($lock) { $lock.Dispose() }
    Write-DamatenLog $cfg "self-play forever stopped"
}
