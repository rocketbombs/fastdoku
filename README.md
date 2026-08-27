# fastdoku

A complete sudoku solver in Rust. Solves any valid 9x9 grid, proves
uniqueness, counts solutions to a limit, and generates minimal puzzles.

On x86-64 it is **the fastest solver in this comparison on all eight of
tdoku's benchmark corpora**, against both tdoku — the strongest of the field
on hard puzzles — and [rust_sudoku](https://github.com/Emerentius/sudoku),
the strongest on easy ones. Against whichever of the two is faster on a
given corpus the margin runs 1.06x to 1.63x; against tdoku alone, 1.21x to
2.18x. Every number below is measured on one machine with one harness.

That lead is x86-specific, and partly Windows-specific. On Apple Silicon,
where there is no AVX2 and the SIMD engine does not exist, rust_sudoku is
the faster solver — see [Portability](#portability-other-machines-other-abis),
which measures all of this on stock CI runners rather than asserting it.

Two engines share the work:

- **`triad`** — a Rust port of [tdoku](https://github.com/t-dillon/tdoku)'s
  DPLL + triad + SIMD architecture (the design is Tom Dillon's), plus a set
  of scheduling, algebraic and ABI changes worth 1.3x-1.4x over upstream on
  the hard corpora.
- **`jcz`** — an original implementation of the JCZSolve architecture
  (bands stored by digit, locked candidates through lookup tables), written
  from its published description with an exact-closure strengthening, a
  worklist-driven propagator and a branchless initializer.

The default **`auto`** engine runs jcz's propagation first and routes each
puzzle by how far that got: near-solved puzzles stay in jcz, the rest
restart in triad.

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
(the hot/cold split described below), not upstream tdoku. It exists so the
table shows how much of the triad engine's lead over tdoku comes from that
single change.

Multithreaded, 16 threads (solving is embarrassingly parallel):

| Corpus | per puzzle | throughput |
|--------|-----------:|-----------:|
| puzzles0_kaggle            |  47 ns | 21.43M/s |
| puzzles1_unbiased          | 111 ns |  9.05M/s |
| puzzles2_17_clue           | 124 ns |  8.05M/s |
| puzzles5_forum_hardest_11+ | 1.68 us |  595K/s |

**Verification:** on the seven uniquely-solvable corpora the solution
checksums match tdoku's exactly — 3.3M puzzles, bit-identical grids.
`puzzles7_serg_benchmark` is composed entirely of multi-solution puzzles
(`fastdoku check` reports 10,000/10,000 multiple); there the first solution
found is engine-dependent, and fastdoku's checksum matches rust_sudoku's,
since the jcz engine makes the same branching choices. Every returned grid
is additionally validated cell by cell, and the five engines are
cross-validated against each other on every solve in the test suite.

### Portability: other machines, other ABIs

Everything above is one Zen 3 Windows machine, and two of the triad engine's
optimizations are Windows ABI artifacts. The
[portability workflow](.github/workflows/portability.yml) re-runs the same
comparison on stock GitHub runners so the machine-specific part of the lead
is measured rather than guessed.

Read these as ratios between solvers measured in the same job, never as
absolute throughput: runners are shared VMs and drift ~10% run to run, the
corpora are sampled to bound CI time, and — unlike the table above — nothing
is PGO-built. fastdoku is built twice per runner, tuned (`target-cpu=native`)
and generic, to separate "lost Zen 3 tuning" from "lost AVX2 entirely".

**linux-x86_64** — all five solvers, all checksums agreeing:

| corpus | fastdoku | portable | tdoku | +fastpath | rust_sudoku |
|--------|---------:|---------:|------:|----------:|------------:|
| kaggle          |   **746** | 1187 | 1360 | 1331 | 1190 |
| serg            |  **1791** | 2233 | 2252 | 2192 | 2094 |
| unbiased        |  **1945** | 2540 | 3492 | 3422 | 2264 |
| 17_clue         |  **1947** | 2629 | 3725 | 3614 | 2294 |
| magictour       |  **6941** | 8598 | 7251 | 7119 | 7656 |
| hardest_11+     | **29319** | 40569 | 31871 | 31664 | 35335 |
| hardest_1106    | **43498** | 52389 | 51296 | 51018 | 45206 |

**windows-x86_64** — all five solvers, all checksums agreeing:

| corpus | fastdoku | portable | tdoku | +fastpath | rust_sudoku |
|--------|---------:|---------:|------:|----------:|------------:|
| kaggle          |   **811** | 1398 | 1763 | 1529 | 1297 |
| serg            |  **1920** | 2657 | 2819 | 2428 | 2249 |
| unbiased        |  **2134** | 3041 | 4568 | 3949 | 2473 |
| 17_clue         |  **2133** | 3095 | 4841 | 4143 | 2537 |
| magictour       |  **7573** | 9917 | 9285 | 8220 | 8339 |
| hardest_11+     | **31998** | 47669 | 40167 | 36254 | 38327 |
| hardest_1106    | **46972** | 61184 | 64200 | 58166 | 48894 |

**macos-arm64** — tdoku's SIMD layer is x86-only (SSE/AVX/AVX-512, no NEON
path), so it cannot be built here at all; fastdoku's triad engine likewise
is not compiled, and `auto` runs jcz with every fallback taken — scalar
condense, scalar sweep, scalar extraction, no vector clue scan:

| corpus | fastdoku | portable | tdoku | +fastpath | rust_sudoku |
|--------|---------:|---------:|------:|----------:|------------:|
| kaggle          | 1187 | 1183 | n/a | n/a |  **1061** |
| serg            | 2244 | 2579 | n/a | n/a |  **2143** |
| unbiased        | 2740 | 2942 | n/a | n/a |  **2350** |
| 17_clue         | 3204 | 2870 | n/a | n/a |  **2238** |
| magictour       | 9565 | 9954 | n/a | n/a |  **7474** |
| hardest_11+     | 43828 | 42873 | n/a | n/a | **32924** |
| hardest_1106    | 56098 | 57977 | n/a | n/a | **44268** |

Three things fall out, one of them unwelcome:

**The Windows-specific claims check out.** The hot/cold split is worth 10-16%
on the Windows runner (tdoku 1763 vs fastpath 1529 on kaggle, 4568 vs 3949
on unbiased) and about 2% on Linux (1360 vs 1331, 3492 vs 3422) — which is
exactly what "Windows x64 makes xmm6-xmm15 callee-saved" predicts, measured
on tdoku's own source rather than ours.

**The lead survives on x86 but compresses.** fastdoku is still fastest of
the four on every corpus on both x86 runners, but against the best of the
other three the hard-corpus margins fall from 1.10x/1.18x/1.06x on the
development machine to 1.03x/1.08x/1.04x on Linux. Losing the Windows ABI
work costs most of it; the rest is that these runners are not Zen 3. The
easy-corpus margins barely move (1.63x -> 1.59x on kaggle), because those
are the jcz engine's, and jcz is ordinary scalar code.

**On Apple Silicon, rust_sudoku is the better solver** — by 4% to 43%. Two
compounding reasons: the triad engine is gone entirely, so hard puzzles fall
to jcz, which is 25-35% slower there even on x86; and jcz's own fast paths
(`pext` condense, AVX2 band sweep and extraction) all degrade to scalar
fallbacks. Native and generic builds are within noise of each other on ARM,
confirming there is no machine-specific tuning left to lose. Porting the
fallbacks to NEON is the obvious next step and has not been done.

### Where the time goes, and why a hybrid

The two architectures fail in opposite regimes. JCZSolve-family solvers do
very little work per deduction — one `u32`, a few table lookups — so they
sprint through puzzles that are long chains of easy deductions, which is
most puzzles. tdoku's triad architecture does much more work per step (a
256-bit fixpoint over a whole box), buys much stronger inference with it,
and pulls ahead when puzzles get hard enough that the guesses it avoids pay
for its heavier steps. On the hard corpora triad takes 2.4x fewer branch
decisions than jcz, at 1.9x the cost per decision.

The `auto` engine exploits the fact that the crossover is observable
mid-solve. It runs jcz's propagation to fixpoint — useful work no matter
which engine finishes — and looks at how many cells remain unsolved at the
first guess point:

- **0 unsolved:** solved outright; jcz was the right engine (100% of
  kaggle, 40% of unbiased, 73% of 17-clue).
- **1–50 unsolved:** stay in jcz and search, with a 16-guess budget as a
  safety valve (covers essentially all of the easy corpora, and all of
  serg).
- **more than 50 unsolved:** propagation stalled far from a solution — this
  is the triad engine's regime; restart there, having spent ~0.3 us.
  (89% of magictour, 100% of the forum-hardest corpora.)

**Why one threshold and not two.** The extreme corpus (`puzzles6`) is the
one place jcz beats triad outright, so a second gate sending the very
hardest puzzles back to jcz looks attractive, and a rule like
`unsolved >= 58` does win that benchmark. It was rejected as overfitting.
Correlating every observable at the first guess point — unsolved cells,
bivalue cells, total candidates, and triad-side band-configuration counts —
against which engine actually wins gives |r| <= 0.064 for all of them:
nothing available before search predicts the winner. The rule "works" only
because `puzzles6` sits slightly higher on that axis than its neighbours,
i.e. it detects the corpus rather than the difficulty. Weighted by real
corpus sizes, triad wins every stratum above 53 unsolved cells, and the
rule costs 0.59 us across `puzzles5`'s 48,766 puzzles to gain 2.5 us on
`puzzles6`'s 375 — a 30x net loss. So the gate stays a single monotone
threshold, and `puzzles6` was won on engine speed instead.

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
the AVX2 triad engine and the BMI2 path in jcz; without them the `auto`
engine runs jcz with portable fallbacks. On Windows `.\build.ps1` adds a
two-stage PGO build (`-NoPgo` to skip) that profiles both engines and the
dispatch.

To reproduce the comparison: clone tdoku, unzip its `data.zip`, then with
clang on PATH run `.\bench\build_tdoku.ps1` and `.\bench\compare.ps1`. Set
`TDOKU_DIR` if the checkout isn't at `C:\Claude\tdoku-ref`, and `RSBENCH` to
a built rust_sudoku harness to include it. Only the comparison needs clang
and tdoku; the solver needs neither.

On POSIX systems `bench/build_tdoku.sh` and `bench/ci_bench.sh` do the same
job and are what the portability workflow runs; `ci_bench.sh` also checks
that every solver present agrees on the solution checksums, since a
divergence on another platform would be a portability bug rather than a
timing.
[`bench/rust_sudoku_bench.rs`](bench/rust_sudoku_bench.rs) documents how to
build that harness (it is AGPL, so it is not a dependency of this crate and
none of its code is used here).

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
condenses to a 9-bit minirow-occupancy pattern; one lookup returns the
closure of that pattern expanded back to a cell mask, another returns the
pointing eliminations for the two neighbour bands, and a third pair detects
solved cells. Naked singles are swept between rounds with three saturating
accumulator masks per band.

Four things differ from the canonical design:

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
a 27-bit worklist instead, kept in a register across the drain and processed
in captured batches — no shadow array (halving the per-guess state copy), no
rescans, and batching keeps each iteration's control flow independent of the
loads the updates are computing.

**The minirow condense is a bit-extract, not a table.** `m | m>>1 | m>>2`
puts each minirow's or at the minirow's first cell, so the nine bits wanted
are exactly those at positions 0, 3, 6, ... On Zen 3 `pext` is 3-cycle
hardware, which beats the three dependent table loads it replaces: it takes
a load off the critical path of every subband update — the result feeds
straight into the closure lookup — and frees 512 bytes of L1 besides. Worth
3% on the deep-search corpora, with a portable fallback.

**Initialization is batched and branchless.** Clue positions come from a
bitmask (three AVX2 compares); one pass accumulates a single `unit_mask` per
digit holding its rows, boxes and columns, plus the clue cells per band; then
all 27 subbands are built directly from those. Duplicate clues fall out of a
count rather than a test — every clue contributes exactly three unit bits, so
if two share a row, box or column the or loses one and the total falls short.
Solution extraction is likewise branchless: each subband becomes a 27-lane
byte mask multiplied by its digit, instead of 81 iterations of an
unpredictable bit-scan loop. Together these took initialization from 141 ns
to 69 ns — it had been a quarter of an easy-puzzle solve, and is paid again
on every route to triad.

## What's different from tdoku (triad engine)

Two of these are Windows ABI artifacts that would not help on Linux. The
rest apply anywhere. Measured on the triad engine alone, together they are
worth 1.32x to 1.39x over upstream tdoku on the hard corpora, and 1.19x to
1.28x against the fastpath build that already carries the first of them.

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
tdoku's deficit.

The CI runs confirm the mechanism on tdoku itself: backporting the split is
worth 10-16% on the Windows runner and about 2% on Linux, where SysV leaves
no vector registers callee-saved and there is no prologue to avoid.

### SysV calling convention for the recursive body — ~2%, Windows-specific

Even split, the cold body uses the whole vector file and pays to preserve ten
registers at every level of the propagation recursion. `extern "sysv64"`
makes all sixteen volatile. The prologue drops from 19 instructions (8
pushes, a 248-byte frame, 10 vector spills) to 7, leaving 2 genuine spills.
A no-op on platforms where SysV is already the default.

### The hidden-single scan, algebraically — ~2.5%

The row/column scan built, for each lane, the candidates appearing in two or
more lanes of its 4-group, then took `cells & ~that`. That is a strictly
stronger quantity than the result needs: a candidate the lane does not itself
hold cannot survive the final `cells &` anyway. So the whole thing collapses
to "the candidates no *other* lane of the group holds" — `cells & ~(r1|r2|r3)`
— and fusing both axes with the cardinality trigger gives the entire step as

```
cells & (triggered | ~R | ~C)  ==  cells & ~(R & C & ~triggered)
```

Nine fewer vector operations, and a six-deep serial accumulation becomes a
two-level OR tree off the rotations. Identical assertions: guess counts are
unchanged to the puzzle.

### Broadcasts by folding — ~2.5%

Two of the closure's broadcasts stop being rotations.

A matrix row is exactly one 64-bit lane, so two shift-and-or steps carry each
row's union up to its top element — the horizontal triad, the only lane that
needs it, since inside the 3x3 the row union is a subset of the box union the
closure already takes. The lower elements are left holding prefix unions,
subsets of that same box union, which change nothing. Four operations instead
of three shuffles and three ors, on ports the shuffles were competing for.

A matrix column is one element position repeated in each 64-bit lane, so
folding the register in half twice — once across the 128-bit halves, once
within each half — unions all four rows into every row. Two shuffles and two
ors rather than three cross-lane permutes and three ors, with the second fold
an in-lane `vpshufd`.

### Branch order by configuration count — ~5.5% on extreme puzzles

When the chosen digit has exactly two remaining configurations, both children
are commitments and their order is a free heuristic; exploring the negation
child first cuts branch decisions on the extreme corpus from 59.7 to 58.4 per
puzzle. With three or more configurations the commit child is the stronger
constraint and trying it first stays better. Conditioning on that distinction
is worth 5.5% on `puzzles6` and 1.1% on `puzzles5`, and is what took the last
corpus without any routing special case.

### Const-specialized peer dispatch — ~2%

"Visit the inbound peer last" makes the three peer-triad vectors a runtime
permutation, which the compiler materializes on the stack and reloads through
a variable index — a store-forwarding stall on the critical path of every
`band_eliminate`. Making the inbound peer a const generic parameter turns the
permutation into compile-time register naming. (Dropping the ordering
heuristic instead costs 3-6%.)

### Exit the box fixpoint on assertions, not eliminations — ~6%

Everything downstream of the fixpoint loop — the elimination closure and both
band messages — is a function of the newly asserted literals alone, and
distributes over union. So an iteration that asserts nothing new can only
re-derive consequences already accumulated, and testing *that* moves the loop
exit above the closure: the terminating iteration skips about half the loop
body. A per-box record of what has already been asserted lets a re-entered
box exit immediately. Replacing the old exit test mattered; keeping both was
slower than either alone.

### Seeded peer eliminations at initialization — ~5% on easy puzzles

Entering a clue only touched the clue's own box, so the most elementary
deduction — this digit is already placed in your row or column — had to
travel box -> band configurations -> triads -> box before it landed. Seeding
those eliminations directly (nine vector updates built from row/column digit
masks accumulated during the clue scan) removes a fifth of easy-puzzle
propagation. The seeded boxes are then drained once before the band cascade
so their deductions still reach the branching heuristic; without that the
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
- **PGO** by default on Windows, profiling both engines and the dispatch.

## What didn't work

Recorded because each looks like an obvious win. Measured against a ±0.3%
run-to-run noise floor.

| Change | Expected | Measured |
|---|---|---|
| `vptest` instead of `movemask`+`cmp` for the triad contradiction check | −1 instruction | **~1% slower** — the loop is vector-port bound for throughput; `movemask` hands the test to idle integer ports |
| A second, cheaper loop-exit test in the box fixpoint | skips a whole 26-op iteration | **~1% slower**, twice, at both positions tried — the extra branch costs more than the iteration it saves |
| Hoisting the band accumulators into registers across the fixpoint | removes a store-to-load forward per iteration | **~1% slower** — register pressure; the store port had slack, the register file did not |
| Hoisting the band message out of the fixpoint (valid: it is monotone in the assertion set, so only the last one matters) | one construction per call instead of per iteration | **~2% slower** — the two extra vectors held live across the loop cost more than the hoisted work |
| `vpermd`-based `triad_message`: one cross-lane gather instead of a permute, two shuffles and an or | −2 ops per call | **~1% slower** — `vpermd`'s control must live in a register, and this loop is register-bound |
| Fold-based *peer* unions (as used for the closure broadcasts) | −1 op per axis | **~3% slower** — unlike the closure, the peer union sits on the loop-carried chain, and folding lengthens it |
| Dropping the per-box assertion memo to shrink search state 768 -> 480 bytes | less copying per guess | **~0.5% slower** — the memo still earns its cache traffic even now that easy puzzles route to jcz |
| Balanced tree for `two_or_more` instead of running accumulation | 1 level shallower, same op count | **~0.7% slower** — keeps 3 rotations and 4 partials live, forcing rematerialization (since obsoleted: the quantity itself was unnecessary) |
| Pinning the popcount nibble table in a register via inline `asm!` | −1 instruction per iteration | **no change** — the evicted broadcast issues on the load port, which the loop has to spare |
| jcz: pairing the two `s`-indexed and two `cols`-indexed lookup tables into one `u64` load each | half the loads, half the cache lines | **~2% slower** — the paired tables double to 8 KB of L1, which costs more than the saved base pointers |
| jcz: unconditional solved-cell sweep (canonical JCZSolve advice) | fewer mispredicts | **slower on every corpus** — the exact closure tables make the no-solve case strongly biased, and the branch predicts |
| jcz: serialized dirty-worklist drain | fewer redundant updates | **slower** — each iteration's branch waited on the previous update's loads; draining captured batches restored the ILP |
| jcz: pairwise combine tree for the naked-single scan | 9-deep chain → 4-deep | **no change** — out-of-order execution across the three bands already hides the chain |
| Defining the auto dispatch in the library crate | no reason it should matter | **~12% slower triad** on hard corpora — with fat LTO, compiling the dispatch beside the triad hot path degrades its codegen; the dispatch lives in the binary crate instead, and `box_restrict_full` carries `inline(always)` after LLVM started outlining it |

The recurring pattern in the triad loop: it is register- and latency-bound,
not op-count-bound. An instruction on an idle port is free and a predictable
branch is nearly free, but anything that adds a live value or lengthens the
`cells -> peers -> assertions` chain loses — even when it removes
instructions. Only changes that cut both won.

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
  definition (all 512 patterns x 6 permutation matrices).
- `bench` verifies every solution and asserts identical checksums across
  rounds, threads and engines.
- Checksums match tdoku bit-for-bit on the seven uniquely-solvable benchmark
  corpora (3.3M puzzles), and classification on a 500,000-puzzle generated
  set (763 unique / 12,786 multiple / 486,451 unsolvable) is unchanged across
  every optimization here.

Five independent implementations agreeing is what makes the unsafe SIMD code
maintainable; any change that breaks one engine's semantics gets caught.

## Credits and license

The triad architecture, tables and update rules are **tdoku's**, by Tom
Dillon — https://github.com/t-dillon/tdoku. The jcz architecture is
**JCZSolve's**, by zhouyundong_2012, champagne and JasonLion (enjoysudoku
forum). This repository contributes a Rust port of the former, an original
implementation of the latter, the optimizations above, the router, and the
head-to-head harness.

The optimizations are substantial and are this repository's own work, but
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
