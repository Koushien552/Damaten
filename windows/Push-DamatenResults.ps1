param([string]$Config = "")

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\Damaten.Common.ps1"

$cfg = Read-DamatenConfig $Config
Ensure-DamatenDirectory $cfg.RepoPath
Ensure-DamatenDirectory $cfg.PartsDir

Invoke-DamatenGit $cfg @("lfs", "install")
Invoke-DamatenGit $cfg @("lfs", "track", "*.nn", "*.tsv", "*.zip")

try {
    Invoke-DamatenGit $cfg @("pull", "--rebase", "--autostash", "origin", $cfg.Branch)
} catch {
    Write-DamatenLog $cfg "pull skipped or failed, continuing for first push: $($_.Exception.Message)"
}

$machine = [string]$cfg.MachineId
$date = Get-Date -Format "yyyy-MM-dd"
$targetDir = Join-Path $cfg.RepoPath "selfplay\windows\$machine\$date"
Ensure-DamatenDirectory $targetDir

$manifestDir = Join-Path $cfg.RepoPath "manifests\windows"
Ensure-DamatenDirectory $manifestDir
$sentPath = Join-Path $manifestDir "$machine.sent.json"
if (Test-Path -LiteralPath $sentPath) {
    $sentDoc = Get-Content -LiteralPath $sentPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $sentSet = @{}
    @($sentDoc.sent_parts) | ForEach-Object { $sentSet[[string]$_] = $true }
} else {
    $sentDoc = [pscustomobject]@{ sent_parts = @() }
    $sentSet = @{}
}

$cutoff = (Get-Date).AddMinutes(-[double]$cfg.MinPartAgeMinutes)
$parts = Get-ChildItem -LiteralPath $cfg.PartsDir -Filter "cycle_*_part_*.tsv" -File |
    Where-Object { $_.LastWriteTime -lt $cutoff -and $_.Length -gt 0 } |
    Sort-Object Name

$copied = 0
foreach ($part in $parts) {
    $key = "{0}:{1}" -f $part.Name, $part.Length
    if ($sentSet.ContainsKey($key)) {
        continue
    }
    $dest = Join-Path $targetDir $part.Name
    Copy-Item -LiteralPath $part.FullName -Destination $dest -Force
    $sentSet[$key] = $true
    $copied++
}

$sentOut = [ordered]@{
    machine = $machine
    updated_at = (Get-Date).ToString("o")
    sent_parts = @($sentSet.Keys | Sort-Object)
}
$sentOut | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $sentPath -Encoding UTF8

# Diagnostic logs stay local only (see .gitignore); they contain absolute
# machine paths and the repo is public, so they are not committed.

# Manifest intentionally omits absolute paths (source_parts_dir/data_path/
# model_path) to avoid leaking the local directory layout into a public repo.
$manifest = [ordered]@{
    machine = $machine
    pushed_at = (Get-Date).ToString("o")
    copied_new_parts = $copied
    eligible_parts = @($parts).Count
    data_bytes = if (Test-Path -LiteralPath $cfg.DataPath) { (Get-Item -LiteralPath $cfg.DataPath).Length } else { 0 }
    model_bytes = if (Test-Path -LiteralPath $cfg.ModelPath) { (Get-Item -LiteralPath $cfg.ModelPath).Length } else { 0 }
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $manifestDir "$machine.json") -Encoding UTF8

Invoke-DamatenGit $cfg @("add", ".gitattributes", "selfplay", "manifests")

$status = & git -C $cfg.RepoPath status --porcelain
if ([string]::IsNullOrWhiteSpace(($status -join "`n"))) {
    Write-DamatenLog $cfg "nothing to push"
    exit 0
}

$message = "windows selfplay results $date $machine"
Invoke-DamatenGit $cfg @("commit", "-m", $message)
Invoke-DamatenGit $cfg @("push", "-u", "origin", $cfg.Branch)
Write-DamatenLog $cfg "pushed results: copied_new_parts=$copied"
