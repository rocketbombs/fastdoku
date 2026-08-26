# fastdoku

A fast, complete sudoku solver in Rust. Solves any valid 9x9 sudoku, proves
uniqueness (or counts solutions to a limit), detects unsolvable and
multi-solution puzzles, and generates random minimal puzzles.

On this machine it is faster than [tdoku](https://github.com/t-dillon/tdoku),
the fastest published sudoku solver, on all eight of tdoku's own benchmark
corpora. The margin and exactly where it comes from are broken out below --
including the part that is an optimization tdoku could simply adopt.

## Results

Benchmarked **on the same machine, the same data, and the same harness**.
tdoku is built from source here and driven by
[`bench/tdoku_bench.cc`](bench/tdoku_bench.cc), which mirrors fastdoku's own
`bench` command: same parsing, same in-memory puzzle vector, same best-of-N
protocol, same solution checksum. Both are built `-march=native` /
`-C target-cpu=native`, both with LTO, both with PGO (PGO helps fastdoku
1.5-4%; it did not help tdoku).

Machine: Ryzen 7 5700X (Zen 3, 8C/16T, AVX2, no AVX-512), Windows 11,
rustc 1.98 / clang 22. Single thread, best-of-N, full corpora, time per
puzzle to the first solution.

| Dataset | puzzles | **fastdoku** | tdoku | tdoku+fastpath\* | vs tdoku | vs +fastpath |
|---------|--------:|-------------:|------:|-----------------:|---------:|-------------:|
| puzzles0_kaggle             |   100,000 | **892 ns** | 1109 ns | 1000 ns | 1.24x | 1.12x |
| puzzles1_unbiased           | 1,000,000 | **2.268 us** | 2.814 | 2.533 | 1.24x | 1.12x |
| puzzles2_17_clue            |    49,158 | **2.374 us** | 3.056 | 2.731 | 1.29x | 1.15x |
| puzzles7_serg_benchmark     |    10,000 | **1.478 us** | 1.830 | 1.627 | 1.24x | 1.10x |
| puzzles3_magictour_top1465  |     1,465 | **4.781 us** | 5.771 | 5.252 | 1.21x | 1.10x |
| puzzles5_forum_hardest_11+  |    48,766 | **21.25 us** | 25.46 | 23.43 | 1.20x | 1.10x |
| puzzles6_forum_hardest_1106 |       375 | **34.18 us** | 40.87 | 37.61 | 1.20x | 1.10x |

\* **tdoku+fastpath is tdoku with fastdoku's one structural optimization
backported** (see below). It is not upstream tdoku; it exists to isolate how
much of the win is that single change. Honest summary: **most of the 12-22%
lead over stock tdoku is that one optimization**, which tdoku could adopt in
a few lines. Against a tdoku that has it, the remaining edge is a real but
modest **3-13%**.

Multithreaded (embarrassingly parallel, 16 threads, full corpora):

| Dataset | puzzles | per puzzle | throughput |
|---------|--------:|-----------:|-----------:|
| puzzles0_kaggle             |   100,000 | 87 ns | 11.51M/s |
| puzzles1_unbiased           | 1,000,000 | 186 ns | 5.37M/s |
| puzzles2_17_clue            |    49,158 | 208 ns | 4.80M/s |
| puzzles4_forum_hardest_1905 | 2,135,371 | 1.71 us | 586K/s |

### Verification

Solution checksums match tdoku's **exactly on all eight datasets** — over
3.4 million puzzles, bit-identical grids. That includes
`puzzles7_serg_benchmark`, which is composed entirely of puzzles with
multiple solutions (`fastdoku check` reports 10,000/10,000 multiple): because
this is the same algorithm with the same branching order, it selects the same
solution tdoku does. Every returned grid is also validated cell-by-cell.

## Usage

```
fastdoku solve <file|->            solve puzzles (81-char lines, . or 0 = blank)
fastdoku check <file|->            classify: unique / multiple / unsolvable
fastdoku bench <file> [--rounds N] [--threads N] [--limit N] [--engine E]
fastdoku gen <count> [--seed N]    generate random minimal unique puzzles
```

Puzzles are one per line, 81 characters, `.` or `0` for blanks; `#` comment
lines and blank lines are skipped, so tdoku's corpora and Norvig's lists work
as-is. `-` reads stdin.

### Building

```bash
cargo build --release
```

No dependencies. `.cargo/config.toml` sets `-C target-cpu=native`, which
enables the AVX2 `triad` engine; without AVX2 the build falls back to the
portable scalar `band` engine automatically.

On Windows, `.\build.ps1` does a two-stage PGO build (worth 1.5-4%);
`-NoPgo` skips it.

### Reproducing the benchmarks

```bash
git clone https://github.com/t-dillon/tdoku      # then unzip its data.zip
```

Then, with LLVM/clang on PATH, `.\bench\build_tdoku.ps1` builds the two
reference binaries and `.\bench\compare.ps1` runs the three-way comparison.
Set `TDOKU_DIR` if the checkout is not at `C:\Claude\tdoku-ref`.

Only the comparison needs clang and a tdoku checkout; the solver itself needs
neither.

## Design

The default `triad` engine is a Rust port of tdoku's `solver_dpll_triad_simd`
(BSD-2-Clause; notice in [`src/triad.rs`](src/triad.rs)). Credit for the
architecture belongs to Tom Dillon. The mechanism:

- **Box state is one 256-bit vector.** Each box is a 4x4 matrix of 9-bit
  candidate sets: the 3x3 corner is the box's cells, the right column and
  bottom row hold *negative triad* literals ("this digit is not in this
  minirow/minicol"). Two constraint families then fall out uniformly —
  exactly-one along each matrix row/column, and per-lane cardinality minimums
  (a cell keeps >= 1 candidate; a negative triad keeps >= 6, because exactly
  3 of 9 digits live in a triad). Popcount equality with the minimum asserts
  everything left in the lane.
- **Band state is configurations, not triads.** For each digit there are only
  6 ways its triads can sit in a band (the 3x3 permutation matrices), so a
  band is 6 lanes of 9-bit digit masks. Boxes and bands exchange elimination
  messages through byte-shuffle tables until mutual fixpoint.
- **Branching is on (band, digit)**: commit the lowest remaining
  configuration versus rule it out, choosing the band with fewest total
  configurations and a digit with fewest configurations in it.

### The one thing that is not tdoku's

Splitting each of `box_restrict` and `band_eliminate` into an always-inlined
fast path (the early-return test, which is overwhelmingly the common case)
and a `noinline` cold body. Under the Windows x64 ABI `xmm6`-`xmm15` are
callee-saved, so the un-split functions execute a prologue spilling ten
vector registers *before* discovering they have nothing to do. Backporting
this to tdoku recovers most of its deficit, which is why it is broken out as
its own column above. It is worth ~11% here and would be worth less on
SysV (Linux/macOS), where no vector registers are callee-saved.

Three other measured wins:

- **`extern "sysv64"` on the recursive propagation function** (~2%). Windows
  x64 makes `xmm6`-`xmm15` callee-saved, so every entry to a function that
  uses the full vector file spills and reloads ten of them. SysV treats all
  sixteen as volatile. This cut the prologue from 19 instructions (8 pushes,
  a 248-byte frame, 10 vector spills) to 7, leaving only 2 genuine spills.
- **Branchless clue scanning** at initialization: locate clue cells with
  three `vpcmpeqb` + bit-scan rather than a per-cell branch (~5% on easy
  puzzles).
- **PGO** (1.5-4%).

### Scheduling the propagation loop

Disassembling the fixpoint loop and working through it instruction by
instruction produced one more win, and it is the opposite of what counting
instructions suggests. The loop is **latency bound on its loop-carried
dependency chain**, not throughput bound, so the payoff is in shortening that
chain even at the cost of extra work:

- **Broadcasting an assertion across its column** was a running `x |= rot(x)`
  reduction: two cross-lane permutes, but chained as permute, or, permute, or
  — 8 cycles, because a cross-lane permute costs 3. Issuing three independent
  permutes off the same source and OR-ing them as a tree is one *more*
  instruction and 5 cycles. **Worth 2-4%**, and most on the hardest puzzles,
  where the loop iterates more.
- The same treatment on the row direction, where in-lane rotations cost 1
  cycle instead of 3, is worth almost nothing — kept, but it is noise.
- **Building the band elimination message** used shuffle + two
  `vextracti128` + or + `vinserti128` per value. The high half already holds
  the vertical triads in the right places, and only one of the three
  horizontal triads lives in the other 128-bit lane, so a half-swap plus two
  in-lane shuffles reaches everything: 4 shuffle-port ops instead of 6.
  Worth ~0.5%.

Three measured *losses*, all recorded because they look like obvious wins:

- Rewriting the contradiction check from `movemask`+`cmp` to the shorter
  `vptest` form removed an instruction and ran **~1% slower**. The loop is
  vector-port bound for throughput; moving the test to the integer ports is
  worth more than the instruction it costs.
- Building `two_or_more` as a balanced tree rather than a running
  accumulation is one level shallower for identical op count, but keeps three
  rotations and four partials live at once. It pushed the loop from 77 to 81
  instructions through rematerialization and ran **~0.7% slower**.
- Pinning the popcount nibble table in a register with inline `asm!` (to stop
  it being re-broadcast every iteration) worked — the `vbroadcasti128` left
  the loop — but spending one of sixteen vector registers cost three
  instructions elsewhere, for **no measurable change**. The broadcast issues
  on the load port, which this loop has to spare. LLVM's choice was correct.

The pattern: instructions that sit on an idle port are free, and the compiler
already knows it. The loop ends up at 80 instructions — two *more* than
before this pass — and 3.5% faster.

### Other engines

Three earlier engines are kept, selectable via `--engine`, because four
independent implementations cross-validating each other is what makes
aggressive unsafe optimization survivable:

- **`band`** — bitboards per digit and horizontal band; exact
  permutation-support reduction by table lookup; lazy column inference; AVX2
  in the assignment hot loop. The fastest of the pre-port engines and the
  portable fallback when AVX2 is unavailable.
- **`simd`** — dual-orientation bitboards, each digit's board held twice in
  one register. Correct, cross-validated, and slower.
- **`baseline`** — classic 9-bit cell masks with 20-peer elimination.

## Engineering log

The path here was not monotonic, and the rejected experiments are as
informative as the accepted ones. Measured on magictour / kaggle:

| Engine | magictour | kaggle |
|---|---|---|
| `baseline` (cell masks, hidden singles, MRV) | ~29 us\* | — |
| `band` (bitboards + permutation-support tables) | 11.97 | 2.02 |
| `band` + AVX2 assign hot path | 13.05† | 1.94 |
| `simd` (dual-orientation vector state) | 16.04 | 3.35 |
| `triad` (ported architecture) | 5.96 | 1.39 |
| `triad` + hot/cold split | 5.14 | 0.96 |
| `triad` + branchless init + PGO | **5.06** | **0.92** |

\* on the older top95 set. † magictour absorbed a regression from
restructuring the stall pass; kaggle and unbiased improved.

Rejected by measurement — each looked good on paper:

- **Cross-digit cardinality inference.** A band's 9 minirows hold exactly 3
  digits each, constraining the digit-by-minirow incidence matrix by column
  as well as row. Shrinks the tree (magictour 21.3 -> 16.9 guesses/puzzle)
  but the scan costs more than the nodes it saves, scalar *and* vectorised.
- **Full dual-orientation SIMD state.** Removes all transposition, but a
  propagation event dirties one of six lanes, so filling the vector does 6x
  the work; `vpgatherdd` is also slow on Zen 3.
- **Configuration branching in the `band` engine** — the right idea in the
  wrong representation. It only pays once triads make it cheap, which is
  exactly what the port provides.
- **Naked pairs** (cut guesses 21 -> 13 on top95, still a net loss),
  **box hidden singles** (found nothing the other scans missed), and
  **`panic=abort`** (LLVM had already proven the hot paths nounwind).

The through-line: at ~0.5 us per search node, extra inference must be nearly
free to pay for itself, and vector width only helps where all nine digits are
genuinely involved. That is precisely what the triad representation achieves
and what the earlier engines could not.

## Correctness

- `cargo test` cross-validates all four engines against each other on
  hundreds of random puzzles — valid, minimal, over-clued, contradictory, and
  multi-solution — asserting identical solution counts and validating every
  returned grid. It also proves the 512-entry permutation-support table
  agrees with an independently written iterative reduction on all inputs.
- `bench` verifies every solution and asserts identical checksums across
  rounds, threads, and engines.
- Checksums match tdoku exactly on 3.4M+ benchmark puzzles.

## Credits and license

The `triad` engine's architecture, tables, and update rules are **tdoku's**,
by Tom Dillon — https://github.com/t-dillon/tdoku. Credit for the design
belongs to him; this repository contributes a Rust port, the hot/cold split
described above, and the head-to-head harness.

Derived files, both carrying the upstream notice:

| File | Relationship to tdoku |
|---|---|
| `src/triad.rs` | Rust port of `src/solver_dpll_triad_simd.cc` |
| `bench/tdoku_fastpath.cc` | Modified copy of that file (hot/cold split only) |

Licensed **BSD-2-Clause**, matching upstream so the derivation stays clean —
see [LICENSE](LICENSE) for terms and [NOTICE](NOTICE) for exactly what is
derived from where.

Benchmark corpora are tdoku's and are not redistributed here. `data/` holds
Peter Norvig's top95/hardest lists (norvig.com) plus generated minimal
puzzles, so the tests and a `-NoPgo`-free build work standalone.
