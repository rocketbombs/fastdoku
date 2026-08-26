# Head-to-head: fastdoku vs tdoku (and optionally the JCZSolve rust port),
# same machine, same datasets, same harness protocol, same checksum.
param(
    [string]$Data = $(if ($env:TDOKU_DIR) { "$env:TDOKU_DIR\data" } else { "C:\Claude\tdoku-ref\data" }),
    [int]$Limit = 50000,
    [int]$Rounds = 5,
    [switch]$Full
)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..

$sets = @(
    @{f = "puzzles0_kaggle"; r = $Rounds },
    @{f = "puzzles1_unbiased"; r = $Rounds },
    @{f = "puzzles2_17_clue"; r = $Rounds },
    @{f = "puzzles7_serg_benchmark"; r = ($Rounds * 2) },
    @{f = "puzzles3_magictour_top1465"; r = ($Rounds * 3) },
    @{f = "puzzles5_forum_hardest_1905_11+"; r = $Rounds },
    @{f = "puzzles6_forum_hardest_1106"; r = ($Rounds * 2) }
)
if ($Full) { $sets += @{f = "puzzles4_forum_hardest_1905"; r = 2 } }

foreach ($s in $sets) {
    $path = Join-Path $Data $s.f
    if (-not (Test-Path $path)) { Write-Host "skip $($s.f) (not found)"; continue }
    & .\target\release\fastdoku.exe bench $path --rounds $s.r --limit $Limit
    if (Test-Path .\bench\out\tdoku_bench.exe) {
        & .\bench\out\tdoku_bench.exe $path --rounds $s.r --limit $Limit
    }
    if (Test-Path .\bench\out\tdoku_fastpath.exe) {
        & .\bench\out\tdoku_fastpath.exe $path --rounds $s.r --limit $Limit
    }
    Write-Host ""
}
