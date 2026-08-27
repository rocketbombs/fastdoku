#!/usr/bin/env bash
# Portability benchmark: fastdoku (native and portable builds) against tdoku,
# tdoku+fastpath and rust_sudoku, on whatever machine this runs on.
#
# Everything is optional except fastdoku's native build: solvers that could
# not be built for this platform are reported as "n/a" rather than skipped
# silently, because their absence is itself a portability result.
#
#   DATA=/path/to/tdoku/data ./bench/ci_bench.sh
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here/.."
data="${DATA:-$root/../tdoku-ref/data}"
ext=""
case "${OS:-}" in Windows_NT) ext=".exe" ;; esac

native="${NATIVE_BIN:-$root/target/release/fastdoku$ext}"
portable="${PORTABLE_BIN:-}"
tdoku="${TDOKU_BIN:-$here/out/tdoku_bench$ext}"
fastpath="${FASTPATH_BIN:-$here/out/tdoku_fastpath$ext}"
rsbench="${RSBENCH:-}"

# Each column is "path|style". The style says which command line the binary
# wants: fastdoku takes a `bench` subcommand, the reference harnesses take the
# file directly. It is passed explicitly rather than inferred from the path,
# because "fastdoku" contains "tdoku" and pattern-matching gets that wrong.
columns=(
    "$native|fastdoku"
    "$portable|fastdoku"
    "$tdoku|plain"
    "$fastpath|plain"
    "$rsbench|plain"
)

# corpus:rounds:limit -- limits keep CI runtime bounded; the 82 MB unbiased and
# 175 MB forum_hardest_1905 sets are sampled rather than run whole.
sets=(
    "puzzles0_kaggle:10:20000"
    "puzzles7_serg_benchmark:10:20000"
    "puzzles1_unbiased:10:20000"
    "puzzles2_17_clue:10:20000"
    "puzzles3_magictour_top1465:10:20000"
    "puzzles5_forum_hardest_1905_11+:5:5000"
    "puzzles6_forum_hardest_1106:10:20000"
)

# Echo one solver's output line for one corpus, or nothing if unavailable.
raw() {
    local bin="$1" style="$2" file="$3" rounds="$4" limit="$5"
    [ -n "$bin" ] && [ -x "$bin" ] || return 1
    if [ "$style" = "fastdoku" ]; then
        "$bin" bench "$file" --rounds "$rounds" --limit "$limit" 2>/dev/null
    else
        "$bin" "$file" --rounds "$rounds" --limit "$limit" 2>/dev/null
    fi
}

# ns/puzzle, or "n/a".
timing() {
    local out v u
    out=$(raw "$@") || { echo "n/a"; return; }
    v=$(echo "$out" | awk '{print $2}')
    u=$(echo "$out" | awk '{print $3}')
    [ -n "$v" ] || { echo "n/a"; return; }
    [ "$u" = "us" ] && v=$(awk "BEGIN{printf \"%.1f\", $v*1000}")
    echo "$v"
}

echo "host: $(uname -s) $(uname -m)"
echo "corpus                            fastdoku  portable     tdoku +fastpath rust_sudoku   (ns/puzzle)"
for spec in "${sets[@]}"; do
    f="${spec%%:*}"; rest="${spec#*:}"; rounds="${rest%%:*}"; limit="${rest#*:}"
    path="$data/$f"
    [ -f "$path" ] || { printf "%-30s  (corpus not present)\n" "$f"; continue; }
    printf "%-30s" "$f"
    for col in "${columns[@]}"; do
        printf " %9s" "$(timing "${col%%|*}" "${col##*|}" "$path" "$rounds" "$limit")"
    done
    echo ""
done

# Checksums must agree across every solver that ran; a mismatch here is a
# portability bug, not a performance result.
echo ""
echo "checksum agreement (uniquely-solvable corpora):"
for spec in "${sets[@]}"; do
    f="${spec%%:*}"; limit="${spec##*:}"
    path="$data/$f"
    [ -f "$path" ] || continue
    case "$f" in *serg*) continue ;; esac   # multi-solution: engine-dependent
    sums=""; n=0
    for col in "${columns[@]}"; do
        o=$(raw "${col%%|*}" "${col##*|}" "$path" 1 "$limit") || continue
        s=$(echo "$o" | grep -o 'sum [0-9a-f]*' | awk '{print $2}')
        [ -n "$s" ] || continue
        sums="$sums $s"; n=$((n + 1))
    done
    u=$(echo "$sums" | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l | tr -d ' ')
    if [ "$n" -lt 2 ]; then
        printf "  %-30s only %s solver(s) ran\n" "$f" "$n"
    elif [ "$u" = "1" ]; then
        printf "  %-30s OK (%s solvers agree)\n" "$f" "$n"
    else
        printf "  %-30s MISMATCH:%s\n" "$f" "$sums"
    fi
done
