# fastdoku

A complete sudoku solver in Rust. Solves any valid 9x9 grid, proves
uniqueness, counts solutions to a limit, and generates minimal puzzles.

It is a port of [tdoku](https://github.com/t-dillon/tdoku)'s DPLL + triad +
SIMD architecture — the design is Tom Dillon's — plus a set of scheduling and
ABI changes that make it faster than tdoku on every one of tdoku's own
benchmark corpora.

**It is not the fastest solver in every regime.** On hard puzzles it leads
the field; on easy and typical ones
[rust_sudoku](https://github.com/Emerentius/sudoku) is meaningfully faster.
Both are measured below.

## Results

Same machine, same data, same harness. tdoku is built from source here and
driven by [`bench/tdoku_bench.cc`](bench/tdoku_bench.cc), which mirrors
fastdoku's `bench` command exactly: same parsing, same in-memory puzzle
vector, same best-of-N protocol, same solution checksum. Both are built
`-march=native` / `-C target-cpu=native`, both with LTO, both offered PGO
(it helps fastdoku 1.5-4%; it did nothing for tdoku).

Ryzen 7 5700X (Zen 3, 8C/16T, AVX2, no AVX-512), Windows 11, rustc 1.98 /
clang 22. Single thread, best-of-N, full corpora, time to first solution.

Corpora are ordered easy to hard. Bold marks the fastest of the four.

| Corpus | puzzles | fastdoku | tdoku | tdoku+fastpath\* | rust_sudoku |
|--------|--------:|---------:|------:|-----------------:|------------:|
| puzzles0_kaggle            |   100,000 | 892 ns   | 1109 ns | 1000 ns | **826 ns** |
| puzzles7_serg_benchmark    |    10,000 | 1.478 us | 1.830   | 1.627   | **1.459**  |
| puzzles1_unbiased          | 1,000,000 | 2.268 us | 2.814   | 2.533   | **1.579**  |
| puzzles2_17_clue           |    49,158 | 2.374 us | 3.056   | 2.731   | **1.670**  |
| puzzles3_magictour_top1465 |     1,465 | **4.781 us** | 5.771 | 5.252 | 5.355 |
| puzzles4_forum_hardest_1905| 2,135,371 | **18.90 us** | 21.29 | 19.55 | 21.38 |
| puzzles5_forum_hardest_11+ |    48,766 | **21.25 us** | 25.46 | 23.43 | 25.84 |
| puzzles6_forum_hardest_1106|       375 | 34.18 us | 40.87   | 37.61   | **31.67**  |

**Against tdoku:** faster on all eight, by 1.13-1.29x against upstream and
1.03-1.15x against the fastpath build.

**Against the field:** fastdoku leads on the three large hard corpora — the
regime tdoku's architecture targets — by 1.11-1.22x over rust_sudoku. It
trails on easy and typical puzzles, by up to 1.44x on `unbiased`, and on the
375-puzzle extreme set. tdoku's own README anticipates this: "for easy
minimal puzzles ... Rust Sudoku tends to be either fastest or on par with the
fastest."

So the accurate claim is *fastest known solver on hard puzzles*, not fastest
overall. Closing the easy-puzzle gap is the obvious open problem: rust_sudoku
is a JCZSolve derivative, so the two architectures trade wins by regime
rather than one dominating.

\* **`tdoku+fastpath` is tdoku with one of fastdoku's changes backported**
(the hot/cold split described below), not upstream tdoku. It exists so the
table shows how much of the lead comes from that single change — most of it.
Against a tdoku that has it, the remaining margin is **1.10-1.15x** on seven
corpora and 1.03x on the eighth.

That eighth is worth a note: `puzzles4` is 2.1M puzzles, ~170 MB of input,
and it is the one set large enough that the working set rather than the
propagation loop sets the pace. Its near-twin `puzzles5` (the same forum
collection filtered to rating 11+, 49k puzzles) is comparable in difficulty
per puzzle and shows the usual 1.10x. Optimizations to the inner loop show up
much less when the machine is fetching puzzles.

Multithreaded, 16 threads (solving is embarrassingly parallel):

| Corpus | per puzzle | throughput |
|--------|-----------:|-----------:|
| puzzles0_kaggle            |  85 ns | 11.76M/s |
| puzzles1_unbiased          | 197 ns |  5.07M/s |
| puzzles2_17_clue           | 220 ns |  4.55M/s |
| puzzles5_forum_hardest_11+ | 1.84 us |  543K/s |

rust_sudoku is measured with the same protocol by
[`bench/rust_sudoku_bench.rs`](bench/rust_sudoku_bench.rs), built as a
standalone crate: it is AGPL, so it is not a dependency of this one.
Infandoku was also measured (correct, checksums match) at 2.1 us / 15.6 us /
66.3 us on kaggle / 17-clue / magictour — 1.9x to 11.5x behind tdoku, despite
an upstream issue claiming a 2.2x win over it.

**Verification:** solution checksums match tdoku's exactly on all eight
corpora — 3.4M puzzles, bit-identical grids. rust_sudoku and Infandoku agree
on every uniquely-solvable corpus too. That includes
`puzzles7_serg_benchmark`, which is composed entirely of multi-solution
puzzles (`fastdoku check` reports 10,000/10,000): same algorithm, same
branching order, same solution chosen. Every returned grid is also validated
cell by cell.

## Usage

```
fastdoku solve <file|->    solve (81-char lines, . or 0 = blank, - reads stdin)
fastdoku check <file|->    classify: unique / multiple / unsolvable
fastdoku bench <file> [--rounds N] [--threads N] [--limit N] [--engine E]
fastdoku gen <count> [--seed N]    generate minimal unique puzzles
```

`#` comments and blank lines are skipped, so tdoku's corpora and Norvig's
lists work as-is.

```bash
cargo build --release
```

No dependencies. `.cargo/config.toml` sets `-C target-cpu=native`, enabling
the AVX2 engine; without AVX2 the build falls back to a portable scalar
engine. On Windows `.\build.ps1` adds a two-stage PGO build (`-NoPgo` to
skip).

To reproduce the comparison: clone tdoku, unzip its `data.zip`, then with
clang on PATH run `.\bench\build_tdoku.ps1` and `.\bench\compare.ps1`. Set
`TDOKU_DIR` if the checkout isn't at `C:\Claude\tdoku-ref`. Only the
comparison needs clang and tdoku; the solver needs neither.

## How it works

Credit for the architecture belongs to tdoku. In brief:

**A box is one 256-bit vector** — a 4x4 matrix of 9-bit candidate sets. The
3x3 corner holds the box's cells; the right column and bottom row hold
*negative triad* literals ("this digit is not in this minirow/minicol"). Two
constraint families then fall out uniformly: exactly-one along each matrix
row and column, and per-lane cardinality minimums (a cell keeps >= 1
candidate; a negative triad keeps >= 6, because exactly 3 of 9 digits live in
a triad). A lane at its minimum asserts everything remaining in it.

**A band is a set of configurations.** For each digit there are only six ways
its triads can sit in a band — the 3x3 permutation matrices — so a band is
six lanes of 9-bit digit masks. Boxes and bands exchange elimination messages
through byte-shuffle tables until mutual fixpoint.

**Branching is on (band, digit):** commit the lowest remaining configuration
or rule it out, choosing the band with fewest configurations overall and a
digit with fewest within it.

`box_restrict` inlines into `band_eliminate` and the three box peers are
unrolled, so one ~540-instruction function holds three copies of the
propagation fixpoint loop; recursion re-enters it by call. Essentially all
runtime is in that loop.

## What's different from tdoku

Two of these are Windows ABI artifacts that would not help on Linux. The
other two are scheduling changes that apply anywhere.

### Hot/cold split of the propagation functions — ~11%, Windows-specific

`box_restrict` and `band_eliminate` both begin with an early return that is
taken on the overwhelming majority of calls (nothing to eliminate). Upstream
they are single functions, so that early return still executes a prologue
that spills `xmm6`-`xmm15` — callee-saved under the Windows x64 ABI — before
discovering there is no work to do.

Splitting each into an always-inlined test plus a `noinline` cold body means
the common path never enters a frame. This is the single largest difference
and it is why the results table carries a `tdoku+fastpath` column: it applies
cleanly to tdoku's own source
([`bench/tdoku_fastpath.cc`](bench/tdoku_fastpath.cc)) and recovers most of
tdoku's deficit. On SysV targets, where no vector registers are callee-saved,
the effect should be much smaller.

### SysV calling convention for the recursive body — ~2%, Windows-specific

Even split, the cold body uses the whole vector file and pays to preserve ten
registers at every level of the propagation recursion. `extern "sysv64"`
makes all sixteen volatile. The prologue drops from 19 instructions (8
pushes, a 248-byte frame, 10 vector spills) to 7, leaving 2 genuine spills.
A no-op on platforms where SysV is already the default.

### Parallel column broadcast — 2-4%, applies anywhere

This one is a real scheduling improvement, not an ABI quirk.

Broadcasting an asserted digit across its column is a 4-way OR reduction.
Written the obvious way — `x |= rot(x)` twice — it uses only two cross-lane
permutes, but chains them: permute, or, permute, or. A cross-lane permute
costs 3 cycles on Zen 3, so that path is 8 cycles. Issuing three *independent*
permutes off the same source and combining them as a balanced tree is one
more instruction and 5 cycles.

The loop is latency-bound on its loop-carried dependency chain, not
throughput-bound, so trading an instruction for three cycles wins. The gain
scales with difficulty — largest on the hardest corpus, where the loop
iterates most — which is the signature of a critical-path effect. Upstream
tdoku uses the chained form.

### Fused band elimination message — ~0.5%, applies anywhere

Building the message from a box vector took shuffle + two `vextracti128` +
or + `vinserti128` per value. But the high half already holds the vertical
triads in the right positions, and of the three horizontal triads only one
lives in the other 128-bit lane. A half-swap plus two in-lane shuffles
reaches everything: 4 shuffle-port operations instead of 6.

### Smaller

- **Branchless clue scanning** at initialization: locate clue cells with
  three `vpcmpeqb` and a bit-scan rather than a per-cell branch (~5% on easy
  puzzles, where init dominates).
- **The box is carried in a register** across fixpoint iterations and written
  back once on exit; the intermediate stores were dead.
- **PGO** by default on Windows (1.5-4%).

## What didn't work

Recorded because each looks like an obvious win, and three of them are the
kind of thing instruction-counting recommends. Measured against a ±0.3%
run-to-run noise floor.

| Change | Expected | Measured |
|---|---|---|
| `vptest` instead of `movemask`+`cmp` for the contradiction check | −1 instruction | **~1% slower** — the loop is vector-port bound for throughput; `movemask` hands the test to idle integer ports |
| Balanced tree for `two_or_more` instead of running accumulation | 1 level shallower, same op count | **~0.7% slower** — keeps 3 rotations and 4 partials live, forcing rematerialization; loop grew 77→81 instructions |
| Pinning the popcount nibble table in a register via inline `asm!` | −1 instruction per iteration | **no change** — it did evict the `vbroadcasti128`, but spending 1 of 16 vector registers cost 3 instructions elsewhere. The broadcast issues on the load port, which this loop has to spare. LLVM was right |
| Sinking the dead per-iteration box store | −1 store per iteration | **no change** — store port is idle too. Kept anyway; the code is simpler |

The pattern: an instruction sitting on an idle port is free, and the compiler
already knows it. After this pass the loop is **80 instructions — two more
than before — and 3.5% faster**.

Earlier, before the port to tdoku's architecture, three original designs were
built and measured: bitboards per digit and band with a permutation-support
table, a dual-orientation variant holding each digit's board twice in one
register, and a classic cell-mask solver. All three are slower than the port
and all three survive as `--engine band|simd|baseline`. Cross-digit
cardinality inference, naked pairs, box hidden singles, and branching on band
configurations were tried in those engines and lost on cost.

## Correctness

- `cargo test` cross-validates all four engines against each other on
  hundreds of random puzzles — valid, minimal, over-clued, contradictory and
  multi-solution — asserting identical solution counts and validating every
  grid returned.
- `bench` verifies every solution and asserts identical checksums across
  rounds, threads and engines.
- Checksums match tdoku bit-for-bit on 3.4M benchmark puzzles.

Four independent implementations agreeing is what makes the unsafe SIMD code
maintainable; any change that breaks one engine's semantics gets caught.

## Credits and license

The architecture, tables and update rules are **tdoku's**, by Tom Dillon —
https://github.com/t-dillon/tdoku. This repository contributes a Rust port,
the changes above, and the head-to-head harness.

| File | Relationship to tdoku |
|---|---|
| `src/triad.rs` | Rust port of `src/solver_dpll_triad_simd.cc` |
| `bench/tdoku_fastpath.cc` | Modified copy of that file (hot/cold split only) |

BSD-2-Clause, matching upstream so the derivation stays clean — see
[LICENSE](LICENSE) for terms and [NOTICE](NOTICE) for what derives from
where.

Benchmark corpora belong to tdoku and are not redistributed here. `data/`
holds Peter Norvig's top95/hardest lists and generated minimal puzzles so the
tests and a non-PGO build work standalone.
