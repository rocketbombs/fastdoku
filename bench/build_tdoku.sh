#!/usr/bin/env bash
# Builds the tdoku reference benchmark binaries for same-machine comparison.
# POSIX counterpart of build_tdoku.ps1, used by the portability workflow.
#
#   TDOKU_DIR=/path/to/tdoku ./bench/build_tdoku.sh
#
# Requires clang++ (or g++ via CXX=) and a tdoku checkout. tdoku's SIMD layer
# is x86-only, so this exits 0 without building on other architectures --
# callers should treat a missing binary as "not available here".
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tdoku="${TDOKU_DIR:-$here/../../tdoku-ref}"
out="$here/out"
cxx="${CXX:-clang++}"

case "$(uname -m)" in
    x86_64 | amd64) ;;
    *)
        echo "tdoku's SIMD layer is x86-only; skipping on $(uname -m)"
        exit 0
        ;;
esac

if [ ! -f "$tdoku/src/solver_dpll_triad_simd.cc" ]; then
    echo "tdoku checkout not found at $tdoku" >&2
    exit 1
fi

mkdir -p "$out"

# Windows toolchains produce foo.exe, and the benchmark harness looks for that
# name, so ask for it explicitly rather than relying on the driver.
ext=""
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) ext=".exe" ;; esac

# -march=native so tdoku gets the same machine-specific tuning the Rust build
# gets from target-cpu=native, and LTO so it gets the same whole-program
# optimization fastdoku gets from fat LTO. Keeps the comparison fair.
common=(-O3 -march=native -std=c++17 -DNDEBUG -w -I"$tdoku/include" -I"$tdoku/src")

# clang needs an LTO-capable linker and will not choose one itself: on Windows
# it stops with "LTO requires -fuse-ld=lld", while Apple's ld64 does LTO
# natively and ships no lld to ask for. Probe instead of guessing per-platform.
link=()
probe="$(mktemp -d)"
printf 'int main(){return 0;}\n' > "$probe/t.cc"
if "$cxx" -flto -fuse-ld=lld "$probe/t.cc" -o "$probe/t$ext" >/dev/null 2>&1; then
    link+=(-fuse-ld=lld)
fi
rm -rf "$probe"

for src in solver_dpll_triad_simd util; do
    "$cxx" "${common[@]}" -flto -c "$tdoku/src/$src.cc" -o "$out/$src.o"
done
"$cxx" "${common[@]}" -flto "${link[@]}" "$here/tdoku_bench.cc" \
    "$out/solver_dpll_triad_simd.o" "$out/util.o" -o "$out/tdoku_bench$ext"

# Also build tdoku with the hot/cold split fastdoku uses (tdoku_fastpath.cc),
# so the comparison isolates that one optimization from everything else.
if [ -f "$here/tdoku_fastpath.cc" ]; then
    "$cxx" "${common[@]}" -c "$here/tdoku_fastpath.cc" -o "$out/tdoku_fastpath.o"
    "$cxx" "${common[@]}" -flto "${link[@]}" "$here/tdoku_bench.cc" \
        "$out/tdoku_fastpath.o" "$out/util.o" -o "$out/tdoku_fastpath$ext"
fi

echo "built $out/tdoku_bench$ext"
