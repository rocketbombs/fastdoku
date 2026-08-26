# fastdoku

A complete sudoku solver in Rust. Solves any valid 9x9 grid, proves
uniqueness, counts solutions to a limit, and generates minimal puzzles.

Two engines share the work. The `triad` engine is a port of
[tdoku](https://github.com/t-dillon/tdoku)'s DPLL + triad + SIMD
architecture — the design is Tom Dillon's — plus scheduling and ABI changes
that make it faster than tdoku on every one of tdoku's own benchmark
corpora. The `jcz` engine is an original implementation of the JCZSolve
architecture (bands stored by digit, locked candidates through lookup
tables — the design behind [rust_sudoku](https://github.com/Emerentius/sudoku))
with an exact-closure strengthening and a batched initializer. The default
`auto` engine runs jcz's propagation first and routes each puzzle by how far
that got: near-solved puzzles stay in jcz, the rest restart in triad.

The result is the fastest solver in this comparison on seven of the eight
benchmark corpora, against both tdoku (the strongest on hard puzzles) and
rust_sudoku (the strongest on easy ones). The eighth — 375 extreme puzzles —
still belongs to rust_sudoku, by about 5%; the routing gate cannot separate
those from the adjacent corpus that triad wins, and the honest fix is noted
below rather than a knob tuned to one benchmark.

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
| puzzles0_kaggle            |   100,000 | **622 ns**   | 1106 ns | 989 ns | 826 ns |
| puzzles7_serg_benchmark    |    10,000 | **1.392 us** | 1.827   | 1.628  | 1.458  |
| puzzles1_unbiased          | 1,000,000 | **1.484 us** | 2.812   | 2.537  | 1.574  |
| puzzles2_17_clue           |    49,158 | **1.517 us** | 3.051   | 2.718  | 1.665  |
| puzzles3_magictour_top1465 |     1,465 | **5.063 us** | 5.771   | 5.256  | 5.343  |
| puzzles4_forum_hardest_1905| 2,135,371 | **17.78 us** | 21.26   | 19.49  | 21.28  |
| puzzles5_forum_hardest_11+ |    48,766 | **21.16 us** | 25.39   | 23.33  | 25.81  |
| puzzles6_forum_hardest_1106|       375 | 33.49 us     | 40.83   | 37.60  | **31.66**  |

**Against tdoku:** faster on all eight corpora, by 1.14-2.01x (and by
1.06-1.79x against the fastpath build carrying fastdoku's own ABI fix).

**Against rust_sudoku:** faster on seven of eight — including every corpus
rust_sudoku used to win — and behind only on `puzzles6`, the 375-puzzle
extreme set.

\* **`tdoku+fastpath` is tdoku with one of fastdoku's changes backported**
(the hot/cold split described below), not upstream tdoku. It exists so the
table shows how much of the triad engine's lead over tdoku comes from that
single change.

### Where the time goes, and why a hybrid

The two architectures fail in opposite regimes. JCZSolve-family solvers do
very little work per deduction — one `u32`, a few table lookups — so they
sprint through puzzles that are long chains of easy deductions, which is
most puzzles. tdoku's triad architecture does much more work per step (a
256-bit fixpoint over a whole box), buys much stronger inference with it,
and pulls ahead when puzzles get hard enough that avoided guesses pay for
the heavier steps.

The `auto` engine exploits the fact that the crossover is observable
mid-solve. It runs jcz's propagation to fixpoint — useful work no matter
which engine finishes — and looks at how many cells remain unsolved at the
first guess point:

- **0 unsolved:** solved outright; jcz was the right engine (100% of
  kaggle, 40% of unbiased, 73% of 17-clue).
- **1–50 unsolved:** stay in jcz and search, with a 16-guess budget as a
  safety valve (covers essentially all of the easy corpora and all of
  serg).
- **more than 50 unsolved:** propagation stalled far from a solution — this
  is the triad engine's regime; restart there, having spent ~0.3–0.6 us.
  (89% of magictour, 100% of the forum-hardest corpora.)

The gate was tuned on the corpora above and is honest about its blind spot:
`puzzles5` (48,766 hard puzzles, triad wins by 25%) and `puzzles6` (375
extreme puzzles, jcz ties rust_sudoku and beats triad) look identical to it
— both stall at 51–60 unsolved cells. It routes both to triad, which is
right 130 times more often than it is wrong. That is why `puzzles6` stays
lost: any escalation scheme that could recover it costs more than the 5% it
would win. Run `--engine jcz` if your puzzles are all extreme.

Multithreaded, 16 threads (solving is embarrassingly parallel):

| Corpus | per puzzle | throughput |
|--------|-----------:|-----------:|
| puzzles0_kaggle            |  59 ns | 16.91M/s |
| puzzles1_unbiased          | 128 ns |  7.84M/s |
| puzzles2_17_clue           | 142 ns |  7.06M/s |
| puzzles5_forum_hardest_11+ | 1.80 us |  554K/s |

**Verification:** on the six uniquely-solvable corpora the solution
checksums match tdoku's exactly — 3.3M puzzles, bit-identical grids — and
match the previous fastdoku release on all of them. `puzzles7_serg_benchmark`
is composed entirely of multi-solution puzzles (`fastdoku check` reports
10,000/10,000 multiple); there the first solution found is engine-dependent,
and fastdoku's checksum matches rust_sudoku's, since the jcz engine makes
the same branching choices. Every returned grid is additionally validated
cell by cell, and the five engines are cross-validated against each other on
every solve in the test suite.

## Usage

```
fastdoku solve <file|->    solve (81-char lines, . or 0 = blank, - reads stdin)
fastdoku check <file|->    classify: unique / multiple / unsolvable
fastdoku bench <file> [--rounds N] [--threads N] [--limit N] [--engine E]
fastdoku gen <count> [--seed N]    generate minimal unique puzzles
```

Engines for `bench --engine`: `auto` (default; the jcz/triad hybrid),
`triad`, `jcz`, `band`, `simd`, `baseline`.

`#` comments and blank lines are skipped, so tdoku's corpora and Norvig's
lists work as-is.

```bash
cargo build --release
```

No dependencies. `.cargo/config.toml` sets `-C target-cpu=native`, enabling
the AVX2 triad engine; without AVX2 the `auto` engine runs jcz (scalar)
unbounded. On Windows `.\build.ps1` adds a two-stage PGO build (`-NoPgo` to
skip) that profiles both engines and the dispatch.

To reproduce the comparison: clone tdoku, unzip its `data.zip`, then with
clang on PATH run `.\bench\build_tdoku.ps1` and `.\bench\compare.ps1`. Set
`TDOKU_DIR` if the checkout isn't at `C:\Claude\tdoku-ref`. Only the
comparison needs clang and tdoku; the solver needs neither.
[`bench/rust_sudoku_bench.rs`](bench/rust_sudoku_bench.rs) documents how to
build the rust_sudoku harness (it is AGPL, so it is not a dependency of this
crate and none of its code is used here).

## How the triad engine works

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
unrolled, so one large function holds three copies of the propagation
fixpoint loop; recursion re-enters it by call. Essentially all triad runtime
is in that loop.

## How the jcz engine works

Credit for the architecture belongs to JCZSolve, by zhouyundong_2012 with
refinements by champagne and JasonLion (enjoysudoku forum). This is an
original implementation from the published design, not a port; see
[NOTICE](NOTICE).

**A subband is one u32** — the 27-bit mask of cells in one band where one
digit can still go. The whole state is 27 words plus a few masks, so a
search node copies in a handful of cache lines.

**Propagation is band-level locked candidates by table lookup.** A subband
shrinks to a 9-bit minirow-occupancy pattern; one lookup returns the closure
of that pattern expanded back to a cell mask, another returns the pointing
eliminations for the two neighbor bands, and a third pair detects solved
cells. Naked singles are swept between rounds with three saturating
accumulator masks per band.

**The closure tables are exact.** A digit's placements in a band restricted
to any full solution form a 3x3 permutation matrix over minirows, so the
true closure of a pattern is the union of the permutation matrices it
contains, and the forced minirows are their intersection — six subset tests
per entry at table-build time, the same single lookup at run time. Canonical
JCZSolve approximates this with a pointing/claiming fixpoint, which is sound
but weaker. The tables are verified exhaustively against the definition in
the test suite.

**A dirty worklist replaces the change-scan.** Canonical JCZSolve re-scans
all 27 subbands against a shadow copy each round. Here every write maintains
a 27-bit dirty mask instead, and the fixpoint loop drains it in captured
batches — no shadow array (halving the per-guess state copy), no rescans,
and batch processing keeps each iteration's control flow independent of the
loads the updates are computing.

**Initialization is batched and branchless.** Clue positions come from a
bitmask (three AVX2 compares); a single pass accumulates per-digit
row/column/box masks; then all 27 subbands are constructed directly from
those masks. Replacing the per-clue scan (a data-dependent branch per cell,
a dozen read-modify-writes per clue) more than halved init time. Solution
extraction is likewise branchless: each subband becomes a 27-lane byte mask
multiplied by its digit, instead of 81 iterations of an unpredictable
bit-scan loop.

## What's different from tdoku (triad engine)

Two of these are Windows ABI artifacts that would not help on Linux. The
rest are scheduling changes that apply anywhere.

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

### Exit the box fixpoint on assertions, not eliminations — ~6%

Everything downstream of the box fixpoint loop — the elimination closure and
both band messages — is a function of the newly asserted literals alone, and
distributes over union. So an iteration that asserts nothing new can only
re-derive consequences already accumulated, and testing *that* (against a
per-box record of what has already been asserted) moves the loop exit above
the closure: the terminating iteration skips about half the loop body, on
70% of iterations. The record also lets a re-entered box exit immediately.
Replacing the old exit test mattered; keeping both was slower than either.

### Const-specialized peer dispatch — ~2%

"Visit the inbound peer last" makes the three peer-triad vectors a runtime
permutation, which the compiler materializes on the stack and reloads
through a variable index — a store-forwarding stall on the critical path of
every `band_eliminate`. Making the inbound peer a const generic parameter
turns the permutation into compile-time register naming. (Dropping the
ordering heuristic instead costs 3–6%.)

### Seeded peer eliminations at initialization — ~5% on easy puzzles

Entering a clue only touched the clue's own box, so the most elementary
deduction — this digit is already placed in your row/column — had to travel
box → band configurations → triads → box before it landed. Seeding those
eliminations directly (nine vector updates built from row/column digit
masks accumulated during the clue scan) removes a fifth of easy-puzzle
propagation. The seeded boxes are then drained once before the band cascade
so their deductions still reach the branching heuristic; without that, the
heuristic goes in partially blind and 17-clue guesses tripled.

### Parallel column broadcast — 2-4%

Broadcasting an asserted digit across its column is a 4-way OR reduction.
Written the obvious way — `x |= rot(x)` twice — it uses only two cross-lane
permutes, but chains them: permute, or, permute, or. A cross-lane permute
costs 3 cycles on Zen 3, so that path is 8 cycles. Issuing three *independent*
permutes off the same source and combining them as a balanced tree is one
more instruction and 5 cycles. The loop is latency-bound on its loop-carried
dependency chain, not throughput-bound, so trading an instruction for three
cycles wins. Upstream tdoku uses the chained form.

### Fused band elimination message — ~0.5%

Building the message from a box vector took shuffle + two `vextracti128` +
or + `vinserti128` per value. But the high half already holds the vertical
triads in the right positions, and of the three horizontal triads only one
lives in the other 128-bit lane. A half-swap plus two in-lane shuffles
reaches everything: 4 shuffle-port operations instead of 6.

### Smaller

- **Branchless clue scanning** at initialization: locate clue cells with
  three `vpcmpeqb` and a bit-scan rather than a per-cell branch.
- **The box is carried in a register** across fixpoint iterations and written
  back once on exit; the intermediate stores were dead.
- **PGO** by default on Windows.

## What didn't work

Recorded because each looks like an obvious win. Measured against a ±0.3%
run-to-run noise floor.

| Change | Expected | Measured |
|---|---|---|
| `vptest` instead of `movemask`+`cmp` for the triad contradiction check | −1 instruction | **~1% slower** — the loop is vector-port bound for throughput; `movemask` hands the test to idle integer ports |
| Balanced tree for `two_or_more` instead of running accumulation | 1 level shallower, same op count | **~0.7% slower** — keeps 3 rotations and 4 partials live, forcing rematerialization |
| Pinning the popcount nibble table in a register via inline `asm!` | −1 instruction per iteration | **no change** — the evicted broadcast issues on the load port, which the loop has to spare |
| jcz: unconditional solved-cell sweep (canonical JCZSolve advice) | fewer mispredicts | **slower on every corpus** — the exact closure tables make the no-solve case strongly biased, and the branch predicts |
| jcz: serialized dirty-worklist drain | fewer redundant updates | **slower** — each iteration's branch waited on the previous update's loads; draining captured batches restored the ILP |
| jcz: pairwise combine tree for the naked-single scan | 9-deep chain → 4-deep | **no change** — out-of-order execution across the three bands already hides the chain |
| Defining the auto dispatch in the library crate | no reason it should matter | **~12% slower triad** on hard corpora — with fat LTO, compiling the dispatch beside the triad hot path degrades its codegen; the dispatch lives in the binary crate instead, and `box_restrict_full` carries `inline(always)` after LLVM started outlining it |

The recurring pattern: an instruction sitting on an idle port is free, a
predictable branch is nearly free, and latency on the critical path is
neither.

Earlier, before the port to tdoku's architecture, three original designs were
built and measured: bitboards per digit and band with a permutation-support
table, a dual-orientation variant holding each digit's board twice in one
register, and a classic cell-mask solver. All three are slower and all three
survive as `--engine band|simd|baseline`.

## Correctness

- `cargo test` cross-validates all five engines against each other on
  hundreds of random puzzles — valid, minimal, over-clued, contradictory and
  multi-solution — asserting identical solution counts and validating every
  grid returned.
- The jcz engine's closure tables are verified exhaustively against their
  definition (all 512 patterns × 6 permutation matrices).
- `bench` verifies every solution and asserts identical checksums across
  rounds, threads and engines.
- Checksums match tdoku bit-for-bit on the six uniquely-solvable benchmark
  corpora (3.3M puzzles).

Five independent implementations agreeing is what makes the unsafe SIMD code
maintainable; any change that breaks one engine's semantics gets caught.

## Credits and license

The triad architecture, tables and update rules are **tdoku's**, by Tom
Dillon — https://github.com/t-dillon/tdoku. The jcz architecture is
**JCZSolve's**, by zhouyundong_2012, champagne and JasonLion (enjoysudoku
forum). This repository contributes a Rust port of the former, an original
implementation of the latter, the changes above, and the head-to-head
harness.

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
