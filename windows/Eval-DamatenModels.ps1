param(
    [string]$Config = "",
    [string]$NewModel = "",
    [string]$OldModel = "",
    [int]$Games = 0,
    [int]$Iters = 0,
    [int]$Seed = 0,
    [string]$Label = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\Damaten.Common.ps1"

$cfg = Read-DamatenConfig $Config

if ([string]::IsNullOrWhiteSpace($NewModel)) { $NewModel = $cfg.ModelPath }
if ([string]::IsNullOrWhiteSpace($OldModel)) { $OldModel = $cfg.PrevModelPath }
if ($Games -le 0) { $Games = [int]$cfg.EvalGames }
if ($Iters -le 0) { $Iters = [int]$cfg.EvalIters }
if ($Seed -le 0) {
    # Vary the games from day to day, yet stay reproducible within a single day.
    $Seed = [int](Get-Date -Format "yyyyMMdd")
}

if (!(Test-Path -LiteralPath $cfg.ExePath)) {
    throw "HexAI.exe not found: $($cfg.ExePath)"
}
if (!(Test-Path -LiteralPath $NewModel)) {
    Write-DamatenLog $cfg "eval skipped: new model not found: $NewModel"
    exit 0
}
if (!(Test-Path -LiteralPath $OldModel)) {
    Write-DamatenLog $cfg "eval skipped: old model not found: $OldModel (nothing to compare yet)"
    exit 0
}

$n = [int]$cfg.BoardSize
Write-DamatenLog $cfg ("eval start: {0} games @ iters={1}  A(new)={2}  B(old)={3}  seed={4}" -f `
    $Games, $Iters, (Split-Path -Leaf $NewModel), (Split-Path -Leaf $OldModel), $Seed)

# Side A is always the NEW model, side B the OLD model. The engine alternates
# colors each game, so the black/white advantage is shared evenly.
$matchArgs = @(
    "match", "--n", $n, "--games", $Games,
    "--model-a", $NewModel, "--model-b", $OldModel,
    "--iters", $Iters, "--seed", $Seed
)
$output = & $cfg.ExePath @matchArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-DamatenLog $cfg "eval failed: match exited with code $LASTEXITCODE"
    $output | ForEach-Object { Write-DamatenLog $cfg "  match> $_" }
    exit 1
}

$aWins = $null; $bWins = $null
$aBlack = 0; $aWhite = 0; $bBlack = 0; $bWhite = 0
foreach ($line in $output) {
    $text = [string]$line
    if ($text -match '^A wins (\d+)/(\d+) \(as black (\d+), as white (\d+)\)') {
        $aWins = [int]$Matches[1]; $aBlack = [int]$Matches[3]; $aWhite = [int]$Matches[4]
    } elseif ($text -match '^B wins (\d+)/(\d+) \(as black (\d+), as white (\d+)\)') {
        $bWins = [int]$Matches[1]; $bBlack = [int]$Matches[3]; $bWhite = [int]$Matches[4]
    }
}
if ($null -eq $aWins -or $null -eq $bWins) {
    Write-DamatenLog $cfg "eval failed: could not parse match output"
    $output | ForEach-Object { Write-DamatenLog $cfg "  match> $_" }
    exit 1
}

$g = $aWins + $bWins
if ($g -le 0) {
    Write-DamatenLog $cfg "eval failed: zero games played"
    exit 1
}

# Win rate of the NEW model and a Wilson 95% confidence interval (Hex has no
# draws, so this is a clean binomial proportion).
$p = $aWins / $g
$z = 1.959963984540054
$denom = 1.0 + ($z * $z) / $g
$center = ($p + ($z * $z) / (2.0 * $g)) / $denom
$margin = ($z * [math]::Sqrt(($p * (1.0 - $p) / $g) + ($z * $z) / (4.0 * $g * $g))) / $denom
$lo = [math]::Max(0.0, $center - $margin)
$hi = [math]::Min(1.0, $center + $margin)

# Convert a win probability into an Elo difference. Clamp away from 0/1 using a
# half-game continuity margin so a clean sweep still yields a finite number.
function ConvertTo-Elo {
    param([double]$Prob, [int]$N)
    $eps = 0.5 / $N
    $q = [math]::Min(1.0 - $eps, [math]::Max($eps, $Prob))
    return -400.0 * [math]::Log10((1.0 / $q) - 1.0)
}
$elo = ConvertTo-Elo -Prob $p -N $g
$eloLo = ConvertTo-Elo -Prob $lo -N $g
$eloHi = ConvertTo-Elo -Prob $hi -N $g

if ($lo -gt 0.5) {
    $verdict = "NEW stronger (95% CI excludes 50%)"
} elseif ($hi -lt 0.5) {
    $verdict = "REGRESSION: new weaker (95% CI excludes 50%)"
} else {
    $verdict = "inconclusive (CI spans 50%; needs more games)"
}

function Format-Signed {
    param([double]$Value)
    $r = [math]::Round($Value)
    if ($r -gt 0) { return "+$r" }
    return "$r"
}

$pPct = [math]::Round($p * 100.0, 1)
$loPct = [math]::Round($lo * 100.0, 1)
$hiPct = [math]::Round($hi * 100.0, 1)
Write-DamatenLog $cfg ("eval result: NEW {0}-{1} OLD  winrate={2}% (95% CI {3}-{4}%)  Elo={5} [{6}, {7}]  => {8}" -f `
    $aWins, $bWins, $pPct, $loPct, $hiPct, (Format-Signed $elo), (Format-Signed $eloLo), (Format-Signed $eloHi), $verdict)

# Append one row to the long-term history so the day-by-day trend is preserved.
$row = [pscustomobject][ordered]@{
    timestamp    = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    games        = $g
    iters        = $Iters
    seed         = $Seed
    new_model    = (Split-Path -Leaf $NewModel)
    old_model    = (Split-Path -Leaf $OldModel)
    new_wins     = $aWins
    old_wins     = $bWins
    winrate      = [math]::Round($p, 4)
    ci_low       = [math]::Round($lo, 4)
    ci_high      = [math]::Round($hi, 4)
    elo          = [math]::Round($elo, 1)
    elo_low      = [math]::Round($eloLo, 1)
    elo_high     = [math]::Round($eloHi, 1)
    new_as_black = $aBlack
    new_as_white = $aWhite
    verdict      = $verdict
    label        = $Label
}

$histPath = $cfg.EvalHistoryPath
Ensure-DamatenDirectory (Split-Path -Parent $histPath)
if (!(Test-Path -LiteralPath $histPath)) {
    $row | Export-Csv -LiteralPath $histPath -NoTypeInformation -Encoding UTF8
} else {
    $row | ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1 | Add-Content -LiteralPath $histPath -Encoding UTF8
}
Write-DamatenLog $cfg "eval history appended: $histPath"
exit 0
