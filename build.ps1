# Release build for fastdoku.
#
# PGO is the default: it is worth 1.5-4% on the triad engine. (It measured a
# small regression on the older band engine -- re-measure after big changes.)
# `-C target-cpu=native` comes from .cargo/config.toml and enables AVX2.
#
# Pass -NoPgo for a plain optimized build.
param([switch]$NoPgo)
# NB: PowerShell variable names are case-insensitive, and a variable bound to
# a [switch] parameter stays type-constrained -- so naming the profile
# directory $pgo alongside a -Pgo switch silently coerces the path to "True".
# Hence $pgoDir.
$usePgo = -not $NoPgo

# Native tools write progress to stderr, which "Stop" would treat as fatal.
$ErrorActionPreference = "Continue"
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
Set-Location $PSScriptRoot

if (-not $usePgo) {
    cargo build --release 2>$null
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    Write-Host "build complete: target\release\fastdoku.exe"
    return
}

$pgoDir = Join-Path $PSScriptRoot "target\pgo"
Remove-Item -Recurse -Force $pgoDir -ErrorAction SilentlyContinue

# Profile across difficulties so the profile is not biased toward either the
# propagation-only path or the deep-search path.
$tdata = if ($env:TDOKU_DIR) { "$env:TDOKU_DIR\data" } else { "C:\Claude\tdoku-ref\data" }
if (Test-Path $tdata) {
    $runs = @(
        @{f = "$tdata\puzzles0_kaggle"; r = 2; n = 20000 },
        @{f = "$tdata\puzzles1_unbiased"; r = 2; n = 20000 },
        @{f = "$tdata\puzzles3_magictour_top1465"; r = 3; n = 2000 },
        @{f = "$tdata\puzzles6_forum_hardest_1106"; r = 2; n = 400 }
    )
} else {
    $runs = @(
        @{f = "data\gen5000.txt"; r = 3; n = 5000 },
        @{f = "data\top95.txt"; r = 5; n = 95 },
        @{f = "data\hardest11.txt"; r = 20; n = 11 }
    )
}

$env:RUSTFLAGS = "-Ctarget-cpu=native -Cprofile-generate=$pgoDir"
cargo build --release 2>$null
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
foreach ($p in $runs) {
    & .\target\release\fastdoku.exe bench $p.f --rounds $p.r --limit $p.n | Out-Null
}

$profdata = Get-ChildItem "$env:USERPROFILE\.rustup\toolchains\*\lib\rustlib\x86_64-pc-windows-msvc\bin\llvm-profdata.exe" | Select-Object -First 1
& $profdata.FullName merge -o "$pgoDir\merged.profdata" $pgoDir

$env:RUSTFLAGS = "-Ctarget-cpu=native -Cprofile-use=$pgoDir\merged.profdata"
cargo build --release 2>$null
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
$env:RUSTFLAGS = $null

Write-Host "PGO build complete: target\release\fastdoku.exe"
