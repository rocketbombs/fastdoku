# Investigations

Working notes behind [fastdoku](README.md): how the two engines work, every
optimization that earned its place, every one that looked obvious and lost,
and the measurements that decided each.

Two conventions throughout. Percentages are against a ±0.3% run-to-run noise
floor on the development machine (Ryzen 7 5700X, Zen 3, Windows 11), taken
best-of-N with the corpora in memory. And a change only counts as a win if
solution checksums are unchanged — an optimization that alters which grid
comes back is a semantic change, not a speedup, and is called out as such.

## Contents

- [Where the time goes, and why a hybrid](#where-the-time-goes-and-why-a-hybrid)
- [How the triad engine works](#how-the-triad-engine-works)
- [What's different from tdoku](#whats-different-from-tdoku-triad-engine)
- [How the jcz engine works](#how-the-jcz-engine-works)
- [The NEON backend](#the-neon-backend)
- [Portability in detail](#portability-in-detail)
- [What didn't work](#what-didnt-work)


## Where the time goes, and why a hybrid

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
and it is why the README's results table carries a `tdoku+fastpath` column:
it applies
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

**Finding the digit of a naked single is branchless too.** Placing a single
needs to know *which* digit it is, and the obvious answer -- scan the band's
nine subbands for the one holding that cell -- runs 1..9 iterations with the
trip count set by the digit itself. That is as good as random, so it
mispredicted on nearly every single placed. The digit's index is instead
bit-sliced across four masks (`slice[k]` carries, at each cell, bit k of the
index of every digit still possible there), so at a cell where only one digit
survives the four masks spell out its index and reading it back is four
shifts and three ors. The masks cost twelve ors, built only when a band
actually has a single to place -- on the hard corpora most calls place
nothing and would never use them. Worth 1.0% on `unbiased` and 0.2-0.6% on
`17_clue` and `magictour`, against 0.25% on `kaggle`, which always has
singles and so always pays for the masks.

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


## The NEON backend

For most of this project's life the triad engine was x86-only, and the
[portability run](#portability-in-detail) showed what that cost: on the
macOS ARM runner every hard puzzle fell through to jcz, and jcz's own fast
paths degraded to scalar fallbacks besides. rust_sudoku won every corpus.

The fix is not a rewrite. The engine's architecture — 4x4 matrices of 9-bit
candidate sets, band configurations, the box/band message fixpoint — has
nothing x86 about it; only the two dozen vector primitives underneath it
did. So `src/triad.rs` now names those primitives and nothing else, and
[`src/tvec.rs`](src/tvec.rs) picks a backend: [AVX2](src/tvec_x86.rs) or
[NEON](src/tvec_neon.rs). The AVX2 backend is the old code moved, byte for
byte in every hot primitive; the x86 benchmarks are unchanged on every
corpus and every checksum is identical.

**Where the vocabulary line falls matters.** It is drawn semantically, not
mechanically. `row_peers` and `col_peers` name *"the union of a lane's three
peers along this axis"* rather than the three rotations x86 reaches it with,
and `triad_message` names the box-to-band permutation rather than the
half-swap-plus-two-shuffles that computes it there. Both were originally
written out at the call site, and leaving them there would have silently
fixed the x86 answer for ARM: the whole margin below comes from the two
architectures answering those three questions differently.

### Two 128-bit registers instead of one 256-bit one

A box is 256 bits, so `C16` becomes a pair of `uint16x8_t` — `lo` holding
matrix rows 0 and 1, `hi` rows 2 and 3. That keeps a matrix row equal to one
64-bit lane, exactly as under AVX2, so every shuffle-control table in the
engine carries over unchanged.

It is also less of a loss than it looks. Apple's cores issue four 128-bit
vector operations per cycle, the same width per cycle as two 256-bit ports,
and aarch64 has 32 vector registers against AVX2's 16 — which matters for a
loop whose failed optimizations are recorded below almost entirely as
register pressure. Three primitives then come out strictly ahead:

**`triad_message` is one instruction.** The permutation gathers the box's
three horizontal triads into positions 4..6 of the low half and leaves the
high half — whose vertical triads already sit there — alone. Under AVX2
neither half is free: a 256-bit shuffle has to cover both, and the third
horizontal triad lives in the other 128-bit lane, so reaching it costs a
half-swap and a second shuffle to OR back in. On NEON `vqtbl2q_u8` indexes
all 32 bytes of a register pair, which no AVX2 shuffle can do, so the
awkward triad is just another index — and the identity half is a register
that is already there. Four table lookups and two ORs collapse to one `tbl`,
twice per assertion round. LLVM allocates the register pair in place, with
no moves to make it consecutive.

**`col_peers` is five operations, not six.** Writing the matrix rows
`r0..r3`, the rotate-by-2 the union needs is `(hi, lo)` — a rename, since
half-swaps are free in a split representation, where AVX2 pays a 3-cycle
`vpermq`. The rotate-by-1 is `(ext(lo,hi), ext(hi,lo))`, and the rotate-by-3
is that same pair swapped, so two `ext`s supply both; their union is common
to the two halves and is computed once. Two `ext`s and three ORs, two levels
deep — and this sits on the fixpoint loop's carried dependency chain, which
is where the engine's latency budget actually goes.

**`popcounts9` is two instructions.** NEON has per-byte popcount in
hardware, and a 16-bit lane holding a 9-bit set is exactly two bytes, so
`cnt` plus `uaddlp` is the whole thing — and exact, with no assumption about
the high bits. AVX2 needs a nibble table, two shuffles, a mask, a shift and
two adds.

**The branch heuristic's six band counts fold in vector.** Choosing a band
to branch on needs the total set bits of each of six configuration vectors,
packed into the eight lanes `minpos_after_sub` then minimizes over. Written
the way x86 wants it — a horizontal popcount per band, assembled through an
array — that is NEON's worst case: six `addv` reductions, six
vector-to-general-register moves and a round trip through memory, all to
build a vector. But `addp` adds *adjacent pairs across two vectors*, so
three levels of it fold six 8-lane count vectors into exactly the layout
wanted, with no scalar in sight: 19 instructions, entirely in registers.
This runs once per search node, so it is paid on every branch decision of
every puzzle that guesses at all. (The x86 backend keeps the scalar form,
where the integer unit is idle and there is no horizontal-popcount
instruction to use instead.)

**The box-wide broadcast folds instead of rotating.** Turning an
asserted-cell vector into the eliminations it implies positionally needs the
whole box's union in the nine cell lanes and each column's union in the
vertical-triad row. AVX2 reaches that by folding the four matrix rows and
then ORing two row rotations, which on NEON is four table lookups and four
control loads. Both reductions fall out of reversals instead: the rows fold
with one OR (the half swap being a rename) and one `ext`, and the four
columns then fold with `rev64`/`rev32`, which need no control register at
all. The two are blended by the cell mask -- bitwise rather than lane-wise,
which is safe because both inputs are 9-bit sets and the mask's set lanes
are exactly nine bits wide. Same depth as the AVX2 form, which matters more
than the two instructions saved, because this sits on the fixpoint loop's
carried dependency chain.

What NEON lacks is a cheap `movemask` or `ptest`: every "is anything set"
question ends in a `umaxv` reduction and a vector-to-general-register move.
Those sit on branch conditions rather than on the carried chain, so
speculation absorbs the latency, but it is why `subset_of` and
`any_less_than` are written to reduce *once* over both halves rather than
testing each — and why the x86 note about preferring `movemask` to `ptest`
has no analogue here.

Two smaller primitives have no ARM instruction at all and are rebuilt:
`clear_low_bit` (a 128-bit `x & (x-1)`) observes that the borrow out of the
low half is exactly "the low half was zero", so the subtrahend is
`[1, lo == 0]` and no 128-bit adder is needed; and `minpos_after_sub`, which
is `phminposuw` on x86, rides the lane index in the low three bits of the
value being minimized so that a plain `vminvq_u16` returns both — with a
*saturating* shift, so the caller's sentinel lanes stay above every genuine
count instead of wrapping back under it.

### Where the ARM floor is

After the second optimization pass the hot function, `band_eliminate_full`,
is 1948 aarch64 instructions against 1393 on x86-64. That ratio is not fat,
and it is worth saying where it does come from, because it bounds what
further work on this engine can return. Broken down by kind:

| | bitwise logic | scalar & addressing | other vector | loads/stores | table lookups | constant loads | stack |
|---|---:|---:|---:|---:|---:|---:|---:|
| share | 25% | 23% | 15% | 16% | 8% | 5% | **2.8%** |

The last column is the informative one: 46 stack loads and 8 stack stores in
a 1948-instruction function is not a register-allocation problem, and 96
constant loads is not a rematerialization problem either. What is left is
that a 256-bit operation is two 128-bit ones, and that x86 folds an address
into the operand where aarch64 computes it. Neither is addressable in this
source.

The same conclusion falls out of the runtimes from the other direction. Take
the ratio each solver's ARM time bears to its own x86 time: fastdoku is 1.07x
to 1.16x, rust_sudoku is 0.76x to 0.94x — that is, rust_sudoku is *faster*
on the ARM runner than on the x86 one. The swing that turned a 1.16x lead on
`unbiased` into a 0.95x deficit is mostly them gaining, not us losing. A
solver that is IPC-limited on Zen 3 has room for Apple's wider core to give
back; one already close to the machine's limit on both does not.

### The jcz engine's fallbacks

jcz runs on every puzzle by both routes, and three of its fast paths were
scalar off x86. All three are now vectorized or rebuilt:

- **The band condense** (`shrink_band`, on the critical path of every
  subband update, feeding straight into the `CLOSED_CELLS` lookup) has no
  `pext` outside x86. It was nine shift-mask-or steps; it is now three
  *independent* byte lookups, one per row of the band, shifted into place —
  and the tables absorb the `m | m>>1 | m>>2` folding as well, since an entry
  is indexed by a raw 9-bit row rather than a pre-folded one. One shift plus
  one load plus the combine, against canonical JCZSolve's three *dependent*
  loads. A table-free alternative exists and was rejected: two chained
  multiplies gather bits at stride 3 without collisions (`* 0x15`, mask,
  `* 0x1041`) in seven instructions, but two dependent 3-cycle multiplies is
  longer than the load latency it would replace.
- **The solved-cell sweep** and **grid extraction** are the AVX2 constructions
  over two 16-byte registers instead of one 32-byte one. The sweep's mask
  needs five operations where x86 has `movemask`'s one — the eight
  comparisons are narrowed to 16-bit lanes, masked against bit weights and
  summed with one `addv` — but that is still against roughly seven scalar
  instructions per subband.
- **The clue scan** is now shared between the two engines
  ([`src/clue_scan.rs`](src/clue_scan.rs)), since both do it and `auto` pays
  it twice whenever it defers. NEON has no `movemask` here either; one `and`
  against per-lane bit weights and two `addv` reductions per 16 bytes give
  the byte mask directly. (The narrowing-`shrn` idiom is shorter but leaves
  one *nibble* per byte, and compacting 80 nibbles back down costs more than
  the reductions it saves.)


## Portability in detail

The README's results table is one Zen 3 Windows machine, and two of the
triad engine's optimizations are Windows ABI artifacts. The
[portability workflow](.github/workflows/portability.yml) re-runs the same
comparison on stock GitHub runners so the machine-specific part of the lead
is measured rather than guessed.

Read these as ratios between solvers measured in the same job, never as
absolute throughput: runners are shared VMs and drift ~10% run to run, the
corpora are sampled to bound CI time, and — unlike the README's table —
nothing is PGO-built. fastdoku is built twice per runner, tuned (`target-cpu=native`)
and generic, to separate "lost Zen 3 tuning" from "lost AVX2 entirely" (on
aarch64 that distinction is empty — NEON is architectural, so the generic
build has the same vector engine).

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
path), so it cannot be built here at all and the comparison is against
rust_sudoku alone:

| corpus | fastdoku | portable | tdoku | +fastpath | rust_sudoku |
|--------|---------:|---------:|------:|----------:|------------:|
| kaggle          |   **785** |   783 | n/a | n/a |  1026 |
| serg            |  **1824** |  1808 | n/a | n/a |  1828 |
| unbiased        |      2108 |  2079 | n/a | n/a |  **2055** |
| 17_clue         |      2126 |  2149 | n/a | n/a |  **2098** |
| magictour       |  **6902** |  7032 | n/a | n/a |  7070 |
| hardest_11+     | **31222** | 31302 | n/a | n/a | 36130 |
| hardest_1106    |     44845 | 46578 | n/a | n/a | **41832** |

These are medians of four runs, not one. The ARM runner's variance
swamped the differences being measured — the same binary on the same corpus
moved 20% between two steps of a single job — so the branch was run four
times and `main` three times on the same runner population, and both columns
below are per-corpus medians of the fastdoku/rust_sudoku ratio:

| corpus | before (jcz only) | after (NEON triad) | fastdoku faster by |
|--------|------------------:|-------------------:|-------------------:|
| kaggle          | 0.90x | **1.30x** | 1.36x |
| serg            | 0.83x | **1.01x** | 1.12x |
| unbiased        | 0.76x |     0.97x | 1.29x |
| 17_clue         | 0.74x |     0.99x | 1.30x |
| magictour       | 0.75x | **1.02x** | 1.37x |
| hardest_11+     | 0.74x | **1.08x** | 1.54x |
| hardest_1106    | 0.77x |     0.92x | 1.20x |

**The deficit is gone.** fastdoku was behind on all seven corpora, by 10% to
35%; it is now ahead on four and within 1-3% on `unbiased` and `17_clue`, and
is 1.12x to 1.54x faster than it was. The largest single gain is
`hardest_11+` at 1.54x, which is the triad engine's own regime and exactly
what having the engine at all is worth.

`hardest_1106` is the one corpus this table cannot really resolve: it is 375
puzzles on a shared VM, and its four runs spanned 0.90x to 1.04x. fastdoku's
own median moved by 0.1% between the two measurement passes while
rust_sudoku's moved by 3%, so read that row as "about even", not as a 8%
deficit.

**Two different mechanisms, visible in the split.** `kaggle` and `serg` never
reach the triad engine — both route entirely to jcz — so their 1.32x and
1.05x are the jcz fallbacks alone, and the gap between those two figures says
where jcz's ARM cost was: kaggle solves at the first fixpoint, so it is
almost all initialization and extraction, which is where the vector clue scan
and the NEON extraction land. serg searches, so it gains only the condense,
and gains 5%.

**Where it still loses, it loses the way it already did on x86.** The three
remaining corpora are ones where `auto`'s single threshold is a compromise
rather than a win, and running the engines separately on the ARM runner
showed the *same* compromise there as on Zen 3: jcz alone beats `auto` on
`unbiased` (2.06 vs 2.11 us) and triad alone beats it on `magictour` (6.54 vs
6.88 us), which is what those two engines do on x86 as well. NEON did not
move the crossover, so there is nothing architecture-specific to re-tune —
see [why one threshold and not two](#where-the-time-goes-and-why-a-hybrid)
for why the compromise stands.

**Native and generic builds are still within noise of each other**, as they
were before: NEON is architectural on aarch64, so `target-cpu=native` has no
vector width to unlock, and the four-run medians differ by 1-3% in both
directions.


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
| Sharing the clue scan as a function taking the per-clue work as a closure | one implementation instead of two, fully inlined | **~1% slower** on kaggle — the callers' accumulators are small arrays, and handing them to a closure across even a fully-inlined call boundary was enough to stop LLVM keeping them where it had. The masks are shared; the bit-scan loops stayed in each engine |
| A table-free portable band condense: two chained multiplies gather bits at stride 3 without collisions (`* 0x15`, mask, `* 0x1041`) | 7 instructions, no L1 footprint | **rejected on latency, not measured** — two dependent 3-cycle multiplies is longer than the three *independent* byte loads it would replace, and this feeds the `CLOSED_CELLS` lookup directly |
| Re-siting the `auto` difficulty gate for aarch64, on the theory that NEON changes the two engines' relative cost | the mid-difficulty corpora, where `auto` is below the better single engine | **no change indicated** — measured per-engine on the ARM runner, `auto` trails jcz on `unbiased` and triad on `magictour` by the same margins it does on x86. The compromise is the same one on both architectures, so there is nothing architecture-specific to re-tune |
| Pre-shifted condense tables, to take the two shifts off `shrink_band`'s critical path | −2 ops per subband update, on jcz's hottest chain | **nothing to take off** — aarch64 folds a shift into a second operand, so `orr x13, x13, x14, lsl #3` was already one instruction. Read the disassembly before optimizing an addressing pattern |
| Building the naked-single digit slices in the accumulation pass, reusing the nine loads already in flight | −9 loads | **no different**, and strictly worse where a band has no single to place, which is most calls on the hard corpora. Built on demand instead |
| `panic = abort` (the `fast` profile) for the benchmark build | no unwind landing pads in the recursive hot paths | **0.5-0.8% on the triad corpora, 0.8% worse on kaggle** — and it would put a differently-built fastdoku against identically-built rivals. Left as a profile you can select, not the default |
| `vqtbl2q` for `positive_triads_to_box_candidates`, as it paid for `triad_message` | one 32-byte gather instead of two lookups and an or | **1 to 3 ops** — unlike `triad_message` the result is a genuine OR of two source lanes per output, which no single gather expresses, and the `others` vector it would gather from costs as much as it saves |

The recurring pattern in the triad loop: it is register- and latency-bound,
not op-count-bound. An instruction on an idle port is free and a predictable
branch is nearly free, but anything that adds a live value or lengthens the
`cells -> peers -> assertions` chain loses — even when it removes
instructions. Only changes that cut both won.

Earlier, before the port to tdoku's architecture, three original designs were
built and measured: bitboards per digit and band with a permutation-support
table, a dual-orientation AVX2 variant holding each digit's board twice in
one register, and a classic cell-mask solver. All three are slower than both
shipped engines — the first two by 2x to 4x — and the first two have since
been deleted: they were kept for cross-validation, but they only ever
validated each other, and the vocabulary tests in
[`src/tvec.rs`](src/tvec.rs) now check the SIMD primitives directly and far
more sharply than "a second AVX2 engine agrees on this puzzle" ever did. The
cell-mask solver stays as `--engine baseline`, because a reference with no
`unsafe` in it is the one that earns its keep.

