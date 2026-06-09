param(
    [string]$Config = "",
    [string]$NewModel = "",
    [string]$OldModel = "",
    [double]$Elo0 = [double]::NaN,
    [double]$Elo1 = [double]::NaN,
    [double]$Alpha = 0,
    [double]$Beta = 0,
    [int]$Batch = 0,
    [int]$MaxGames = 0,
    [int]$Iters = 0,
    [int]$Seed = 0,
    [string]$Label = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\Damaten.Common.ps1"

$cfg = Read-DamatenConfig $Config

if ([string]::IsNullOrWhiteSpace($NewModel)) { $NewModel = $cfg.ModelPath }
if ([string]::IsNullOrWhiteSpace($OldModel)) { $OldModel = $cfg.PrevModelPath }
if ([double]::IsNaN($Elo0)) { $Elo0 = [double]$cfg.SprtElo0 }
if ([double]::IsNaN($Elo1)) { $Elo1 = [double]$cfg.SprtElo1 }
if ($Alpha -le 0) { $Alpha = [double]$cfg.SprtAlpha }
if ($Beta -le 0) { $Beta = [double]$cfg.SprtBeta }
if ($Batch -le 0) { $Batch = [int]$cfg.SprtBatch }
if ($MaxGames -le 0) { $MaxGames = [int]$cfg.SprtMaxGames }
if ($Iters -le 0) { $Iters = [int]$cfg.EvalIters }
if ($Seed -le 0) { $Seed = [int](Get-Date -Format "yyyyMMdd") }

if ($Elo1 -le $Elo0) { throw "SprtElo1 ($Elo1) must be greater than SprtElo0 ($Elo0)." }
if (!(Test-Path -LiteralPath $cfg.ExePath)) { throw "HexAI.exe not found: $($cfg.ExePath)" }
if (!(Test-Path -LiteralPath $NewModel)) {
    Write-DamatenLog $cfg "sprt skipped: new model not found: $NewModel"
    exit 0
}
if (!(Test-Path -LiteralPath $OldModel)) {
    Write-DamatenLog $cfg "sprt skipped: old model not found: $OldModel (nothing to compare yet)"
    exit 0
}

# Convert each Elo hypothesis into a per-game win probability for the NEW model
# (Hex has no draws, so the game outcome is a clean Bernoulli trial).
function ConvertFrom-Elo {
    param([double]$Elo)
    return 1.0 / (1.0 + [math]::Pow(10.0, -$Elo / 400.0))
}
$p0 = ConvertFrom-Elo $Elo0
$p1 = ConvertFrom-Elo $Elo1

# SPRT acceptance boundaries on the log-likelihood ratio.
$lower = [math]::Log($Beta / (1.0 - $Alpha))   # <= this -> accept H0
$upper = [math]::Log((1.0 - $Beta) / $Alpha)    # >= this -> accept H1
$winTerm = [math]::Log($p1 / $p0)
$lossTerm = [math]::Log((1.0 - $p1) / (1.0 - $p0))

$n = [int]$cfg.BoardSize
Write-DamatenLog $cfg ("sprt start: H0=+{0} H1=+{1} Elo  alpha={2} beta={3}  bounds=[{4}, {5}]  batch={6} maxGames={7} iters={8}" -f `
    $Elo0, $Elo1, $Alpha, $Beta, [math]::Round($lower, 3), [math]::Round($upper, 3), $Batch, $MaxGames, $Iters)
Write-DamatenLog $cfg ("sprt models: A(new)={0}  B(old)={1}" -f (Split-Path -Leaf $NewModel), (Split-Path -Leaf $OldModel))

function Format-Signed {
    param([double]$Value)
    $r = [math]::Round($Value)
    if ($r -gt 0) { return "+$r" }
    return "$r"
}

$histPath = $cfg.SprtHistoryPath
Ensure-DamatenDirectory (Split-Path -Parent $histPath)
$runId = (Get-Date -Format "yyyyMMdd_HHmmss")

$totalNew = 0
$totalOld = 0
$games = 0
$batchIdx = 0
$llr = 0.0
$decision = ""
$status = "running"

while ($games -lt $MaxGames) {
    $batchIdx++
    $remaining = $MaxGames - $games
    $thisBatch = [math]::Min($Batch, $remaining)
    $batchSeed = $Seed + $batchIdx

    $matchArgs = @(
        "match", "--n", $n, "--games", $thisBatch,
        "--model-a", $NewModel, "--model-b", $OldModel,
        "--iters", $Iters, "--seed", $batchSeed
    )
    $output = & $cfg.ExePath @matchArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-DamatenLog $cfg "sprt failed: match exited with code $LASTEXITCODE"
        $output | ForEach-Object { Write-DamatenLog $cfg "  match> $_" }
        exit 1
    }

    $aWins = $null; $bWins = $null
    foreach ($line in $output) {
        $text = [string]$line
        if ($text -match '^A wins (\d+)/(\d+)') { $aWins = [int]$Matches[1] }
        elseif ($text -match '^B wins (\d+)/(\d+)') { $bWins = [int]$Matches[1] }
    }
    if ($null -eq $aWins -or $null -eq $bWins) {
        Write-DamatenLog $cfg "sprt failed: could not parse match output"
        $output | ForEach-Object { Write-DamatenLog $cfg "  match> $_" }
        exit 1
    }

    $totalNew += $aWins
    $totalOld += $bWins
    $games += ($aWins + $bWins)
    $llr = ($totalNew * $winTerm) + ($totalOld * $lossTerm)

    $p = if ($games -gt 0) { $totalNew / $games } else { 0 }
    $eps = 0.5 / [math]::Max(1, $games)
    $q = [math]::Min(1.0 - $eps, [math]::Max($eps, $p))
    $eloEst = -400.0 * [math]::Log10((1.0 / $q) - 1.0)

    if ($llr -ge $upper) { $status = "H1"; $decision = "PASS: new model accepted as stronger (>= +$Elo1 Elo region)" }
    elseif ($llr -le $lower) { $status = "H0"; $decision = "FAIL: new model not a +$Elo1 Elo improvement" }

    Write-DamatenLog $cfg ("sprt progress: games={0} NEW {1}-{2} OLD  winrate={3}%  Elo~{4}  LLR={5} (bounds {6}..{7})" -f `
        $games, $totalNew, $totalOld, [math]::Round($p * 100.0, 1), (Format-Signed $eloEst), `
        [math]::Round($llr, 3), [math]::Round($lower, 2), [math]::Round($upper, 2))

    $row = [pscustomobject][ordered]@{
        run_id    = $runId
        timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        batch     = $batchIdx
        games     = $games
        new_wins  = $totalNew
        old_wins  = $totalOld
        winrate   = [math]::Round($p, 4)
        elo_est   = [math]::Round($eloEst, 1)
        llr       = [math]::Round($llr, 4)
        lower     = [math]::Round($lower, 4)
        upper     = [math]::Round($upper, 4)
        elo0      = $Elo0
        elo1      = $Elo1
        status    = $status
        label     = $Label
    }
    if (!(Test-Path -LiteralPath $histPath)) {
        $row | Export-Csv -LiteralPath $histPath -NoTypeInformation -Encoding UTF8
    } else {
        $row | ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1 | Add-Content -LiteralPath $histPath -Encoding UTF8
    }

    if ($status -ne "running") { break }
}

if ($status -eq "running") {
    $decision = "INCONCLUSIVE: reached MaxGames=$MaxGames without crossing a boundary"
}
Write-DamatenLog $cfg ("sprt result: {0}  (games={1}, NEW {2}-{3} OLD, LLR={4})" -f `
    $decision, $games, $totalNew, $totalOld, [math]::Round($llr, 3))

# Exit code communicates the outcome to callers: 0 PASS, 3 FAIL, 4 inconclusive.
switch ($status) {
    "H1" { exit 0 }
    "H0" { exit 3 }
    default { exit 4 }
}
