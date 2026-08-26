# Builds the tdoku reference benchmark binary for same-machine comparison.
# Requires: LLVM/clang (winget install LLVM.LLVM) and a tdoku checkout.
$ErrorActionPreference = "Stop"
$env:Path = "C:\Program Files\LLVM\bin;$env:Path"

$tdoku = if ($env:TDOKU_DIR) { $env:TDOKU_DIR } else { "C:\Claude\tdoku-ref" }
if (-not (Test-Path "$tdoku\src\solver_dpll_triad_simd.cc")) {
    throw "tdoku checkout not found at $tdoku (git clone https://github.com/t-dillon/tdoku)"
}

$out = Join-Path $PSScriptRoot "out"
New-Item -ItemType Directory -Force $out | Out-Null

foreach ($src in @("solver_dpll_triad_simd", "util")) {
    clang++ -O3 -march=native -flto -std=c++17 -DNDEBUG -D_CRT_SECURE_NO_WARNINGS -w `
        -I"$tdoku\include" -I"$tdoku\src" -c "$tdoku\src\$src.cc" -o "$out\$src.o"
}

# Link with LTO so tdoku gets the same whole-program optimization the Rust
# build gets from fat LTO, keeping the comparison fair.
clang++ -O3 -march=native -flto -fuse-ld=lld -std=c++17 -DNDEBUG -D_CRT_SECURE_NO_WARNINGS -w `
    -I"$tdoku\include" `
    (Join-Path $PSScriptRoot "tdoku_bench.cc") "$out\solver_dpll_triad_simd.o" "$out\util.o" `
    -o "$out\tdoku_bench.exe"

# Also build tdoku with the hot/cold split fastdoku uses (see tdoku_fastpath.cc),
# so the comparison isolates that optimization from everything else.
if (Test-Path (Join-Path $PSScriptRoot "tdoku_fastpath.cc")) {
    clang++ -O3 -march=native -std=c++17 -DNDEBUG -D_CRT_SECURE_NO_WARNINGS -w `
        -I"$tdoku\include" -I"$tdoku\src" `
        -c (Join-Path $PSScriptRoot "tdoku_fastpath.cc") -o "$out\tdoku_fastpath.o"
    clang++ -O3 -march=native -flto -fuse-ld=lld -std=c++17 -DNDEBUG -D_CRT_SECURE_NO_WARNINGS -w `
        -I"$tdoku\include" `
        (Join-Path $PSScriptRoot "tdoku_bench.cc") "$out\tdoku_fastpath.o" "$out\util.o" `
        -o "$out\tdoku_fastpath.exe"
}

Write-Host "built $out\tdoku_bench.exe"
