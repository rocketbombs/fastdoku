# fastdoku

A complete sudoku solver in Rust. Solves any valid 9x9 grid, proves
uniqueness, counts solutions to a limit, and generates minimal puzzles.
No dependencies.

**On x86-64, fastdoku solves every one of tdoku's eight benchmark corpora
faster than either [tdoku](https://github.com/t-dillon/tdoku) or
[rust_sudoku](https://github.com/Emerentius/sudoku)** — the two strongest
solvers in tdoku's own published comparison, and the ones that between them
held every regime from trivial to extreme. Margins over whichever of the two
is faster on a given corpus run **1.06x to 1.63x**, and the ordering
reproduces on three different machines.

**On ARM64 it does not.** Without AVX2 the SIMD engine is not compiled at
all, and rust_sudoku is 4-43% faster. That is measured below, not assumed.

## Results

Same machine, same data, same harness. tdoku is built from source here and
driven by [`bench/tdoku_bench.cc`](bench/tdoku_bench.cc), which mirrors
fastdoku's `bench` command exactly: same parsing, same in-memory puzzle
vector, same best-of-N protocol, same solution checksum. rust_sudoku is
measured by [`bench/rust_sudoku_bench.rs`](bench/rust_sudoku_bench.rs) under
the identical protocol. All are built `-march=native` /
`-C target-cpu=native` with LTO; fastdoku and tdoku were both offered PGO
(it helps fastdoku, it did nothing for tdoku).

Ryzen 7 5700X (Zen 3, 8C/16T, AVX2, no AVX-512), Windows 11, rustc 1.98 /
clang 22. Single thread, best-of-N, full corpora, time to first solution.
Corpora are ordered easy to hard; bold marks the fastest of the four.

| Corpus | puzzles | fastdoku | tdoku | tdoku+fastpath\* | rust_sudoku |
|--------|--------:|---------:|------:|-----------------:|------------:|
| puzzles0_kaggle            |   100,000 | **512 ns**   | 1100 ns | 989 ns | 834 ns |
| puzzles7_serg_benchmark    |    10,000 | **1.231 us** | 1.876   | 1.622  | 1.458  |
| puzzles1_unbiased          | 1,000,000 | **1.356 us** | 2.811   | 2.532  | 1.575  |
| puzzles2_17_clue           |    49,158 | **1.405 us** | 3.056   | 2.736  | 1.670  |
| puzzles3_magictour_top1465 |     1,465 | **4.781 us** | 5.796   | 5.257  | 5.350  |
| puzzles4_forum_hardest_1905| 2,135,371 | **16.90 us** | 21.24   | 19.53  | 21.38  |
| puzzles5_forum_hardest_11+ |    48,766 | **19.79 us** | 25.71   | 23.35  | 25.86  |
| puzzles6_forum_hardest_1106|       375 | **29.80 us** | 40.88   | 37.61  | 31.64  |

\* **`tdoku+fastpath` is tdoku with one of fastdoku's changes backported**
(a hot/cold split of its two propagation functions), not upstream tdoku. It
is there so the table shows how much of the lead comes from that one change.

**Verification.** On the seven uniquely-solvable corpora the solution
checksums match tdoku's exactly — 3.3M puzzles, bit-identical grids.
`puzzles7_serg_benchmark` is entirely multi-solution (`fastdoku check`
reports 10,000/10,000 multiple); there the first solution found is
engine-dependent, and fastdoku's checksum matches rust_sudoku's. Every
returned grid is additionally validated cell by cell, and the five engines
are cross-validated against each other on every solve in the test suite.

### Does it hold up off the development machine?

Two of the optimizations are Windows ABI artifacts, and the whole table
above is one Zen 3 machine, so the
[portability workflow](.github/workflows/portability.yml) re-runs the
comparison on stock GitHub runners. Below is fastdoku's speed as a multiple
of the *fastest other solver available on that platform* — above 1.00 means
fastdoku wins:

| Corpus | Zen 3 / Windows | linux-x86_64 | windows-x86_64 | macos-arm64 |
|--------|----------------:|-------------:|---------------:|------------:|
| kaggle          | 1.63x | 1.59x | 1.60x | **0.89x** |
| serg            | 1.18x | 1.17x | 1.17x | **0.95x** |
| unbiased        | 1.16x | 1.16x | 1.16x | **0.86x** |
| 17_clue         | 1.19x | 1.18x | 1.19x | **0.70x** |
| magictour       | 1.10x | 1.03x | 1.09x | **0.78x** |
| hardest_1905    | 1.16x |   —   |   —   |     —     |
| hardest_11+     | 1.18x | 1.08x | 1.13x | **0.75x** |
| hardest_1106    | 1.06x | 1.04x | 1.04x | **0.79x** |

The x86 result reproduces: fastest of the four on every corpus on both x86
runners. The hard-corpus margins compress — most of that is the Windows ABI
work, which is worth nothing under SysV — while the easy-corpus margins
barely move, because those belong to the scalar engine.

The ARM column is the honest caveat. There is no AVX2, so the SIMD engine is
absent and hard puzzles fall to the scalar engine, whose own fast paths
(`pext`, vector sweeps and extraction) degrade to portable fallbacks too.
tdoku cannot be built there at all — its SIMD layer is x86-only — so the
comparison is against rust_sudoku alone. NEON fallbacks are the obvious
unfinished work.

Runner CPUs are shared VMs and drift ~10% run to run, the corpora are
sampled to bound CI time, and nothing there is PGO-built, so those columns
are meaningful only as ratios within a job. Full tables and analysis are in
[INVESTIGATIONS.md](INVESTIGATIONS.md#portability-in-detail).

### Multithreaded

16 threads on the development machine; solving is embarrassingly parallel.

| Corpus | per puzzle | throughput |
|--------|-----------:|-----------:|
| puzzles0_kaggle            |  47 ns | 21.43M/s |
| puzzles1_unbiased          | 111 ns |  9.05M/s |
| puzzles2_17_clue           | 124 ns |  8.05M/s |
| puzzles5_forum_hardest_11+ | 1.68 us |  595K/s |

## How it works

Two engines, chosen per puzzle.

**`jcz`** is an original implementation of the JCZSolve architecture: a
subband is one `u32` — the 27-bit mask of cells in one band where one digit
can still go — and propagation is band-level locked candidates by table
lookup. Very little work per deduction, so it sprints through puzzles that
are long chains of easy deductions, which is most puzzles.

**`triad`** is a Rust port of tdoku's DPLL + triad + SIMD architecture: a box
is one 256-bit vector holding a 4x4 matrix of 9-bit candidate sets, with
negative triad literals in the margins, and a band is the six ways a digit's
triads can sit in it. Much more work per step, much stronger inference, so
it wins once puzzles are hard enough that the guesses it avoids pay for the
heavier steps. On the hard corpora it takes 2.4x fewer branch decisions than
jcz at 1.9x the cost per decision.

**`auto`** (the default) runs jcz's propagation to fixpoint — useful work
whichever engine finishes — and routes on how many cells remain unsolved at
the first guess point: solved outright or near it stays in jcz, stalled far
from a solution restarts in triad.

Three older engines (`band`, `simd`, `baseline`) are kept selectable. They
are slower, but they are independent implementations, and cross-validating
five solvers against each other on every test is what makes the unsafe SIMD
code maintainable.

[INVESTIGATIONS.md](INVESTIGATIONS.md) covers all of this properly: how each
engine works, every optimization and what it measured, the routing analysis
(including a rule that wins a benchmark and was rejected for overfitting),
and a table of the changes that looked obvious and lost.

## Usage

```
fastdoku solve <file|->    solve (81-char lines, . or 0 = blank, - reads stdin)
fastdoku check <file|->    classify: unique / multiple / unsolvable
fastdoku bench <file> [--rounds N] [--threads N] [--limit N] [--engine E]
fastdoku gen <count> [--seed N]    generate minimal unique puzzles
```

Engines for `bench --engine`: `auto` (default), `triad`, `jcz`, `band`,
`simd`, `baseline`. `#` comments and blank lines are skipped, so tdoku's
corpora and Norvig's lists work as-is.

```bash
cargo build --release
```

`.cargo/config.toml` sets `-C target-cpu=native`, enabling the AVX2 triad
engine and the BMI2 path in jcz; without them `auto` runs jcz with portable
fallbacks. On Windows `.\build.ps1` adds a two-stage PGO build (`-NoPgo` to
skip) that profiles both engines and the dispatch.

To reproduce the comparison: clone tdoku, unzip its `data.zip`, then with
clang on PATH run `.\bench\build_tdoku.ps1` and `.\bench\compare.ps1` (or
`bench/build_tdoku.sh` and `bench/ci_bench.sh` on POSIX, which is what the
portability workflow runs). Set `TDOKU_DIR` if the checkout isn't at
`C:\Claude\tdoku-ref`, and `RSBENCH` to a built rust_sudoku harness to
include it — [`bench/rust_sudoku_bench.rs`](bench/rust_sudoku_bench.rs)
documents how to build one. Only the comparison needs clang and tdoku; the
solver needs neither.

## Correctness

- `cargo test` cross-validates all five engines against each other on
  hundreds of random puzzles — valid, minimal, over-clued, contradictory and
  multi-solution — asserting identical solution counts and validating every
  grid returned.
- The jcz engine's closure tables are verified exhaustively against their
  definition (all 512 patterns x 6 permutation matrices).
- `bench` verifies every solution and asserts identical checksums across
  rounds, threads and engines; `ci_bench.sh` additionally checks that every
  solver present on the platform agrees, since a divergence elsewhere would
  be a portability bug rather than a timing.
- Checksums match tdoku bit-for-bit on the seven uniquely-solvable corpora
  (3.3M puzzles), and classification of a 500,000-puzzle generated set
  (763 unique / 12,786 multiple / 486,451 unsolvable) is unchanged across
  every optimization in this repository.

## Credits and license

The triad architecture, tables and update rules are **tdoku's**, by Tom
Dillon — https://github.com/t-dillon/tdoku. The jcz architecture is
**JCZSolve's**, by zhouyundong_2012, champagne and JasonLion (enjoysudoku
forum). This repository contributes a Rust port of the former, an original
implementation of the latter, the optimizations in
[INVESTIGATIONS.md](INVESTIGATIONS.md), the router, and the head-to-head
harness.

Those optimizations are substantial and are this repository's own work, but
they are optimizations *of* those architectures: `src/triad.rs` still carries
tdoku's state representation, its message tables and its update rules, and
would be recognisable to anyone reading the two side by side. The derivation
is stated plainly rather than softened.

| File | Relationship to prior work |
|---|---|
| `src/triad.rs` | Rust port of tdoku's `src/solver_dpll_triad_simd.cc` |
| `src/jcz.rs` | Original implementation of the JCZSolve architecture from its published description |
| `bench/tdoku_fastpath.cc` | Modified copy of tdoku's solver (hot/cold split only) |

BSD-2-Clause, matching tdoku so the derivation stays clean — see
[LICENSE](LICENSE) for terms and [NOTICE](NOTICE) for what derives from
where. rust_sudoku is AGPL and is neither a dependency nor a source of any
code here; it is benchmarked as an external binary.

Benchmark corpora belong to tdoku and are not redistributed here. `data/`
holds Peter Norvig's top95/hardest lists and generated minimal puzzles so the
tests and a non-PGO build work standalone.
