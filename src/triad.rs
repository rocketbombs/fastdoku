//! DPLL + triad + SIMD solver core.
//!
//! This is a Rust port of tdoku's `solver_dpll_triad_simd.cc` by Tom Dillon
//! (https://github.com/t-dillon/tdoku, BSD-2-Clause; see comment at the end
//! of this file for the license notice). The architecture, tables, and update
//! rules follow that solver closely; comments here summarize the mechanism,
//! and the original file documents it in depth.
//!
//! State per box: a 4x4 matrix of 9-bit candidate sets in one 256-bit vector.
//! The 3x3 top-left corner holds the box's cells; the right column holds
//! *negative horizontal triad* literals (digit is NOT in that minirow) and
//! the bottom row negative vertical triads. Two constraint families fall out
//! uniformly:
//!
//! - exactly-one per matrix row/column (a digit is in one of 3 cells or
//!   absent from the minirow/minicol), detected as hidden singles by
//!   rotate-and-accumulate;
//! - cardinality by popcount minimums: a cell keeps >= 1 candidate, a
//!   negative triad keeps >= 6 (because exactly 3 digits live in a triad).
//!   Equality triggers assertion of everything remaining in the lane.
//!
//! State per band: for each digit, a mask over the 6 possible *configurations*
//! (3x3 permutation matrices) of that digit's triads in the band, stored as
//! 6 lanes of 9-bit digit masks. Bands and boxes exchange elimination
//! messages through byte-shuffle tables until mutual fixpoint.
//!
//! Branching is binary on (band, digit): commit the lowest remaining
//! configuration versus rule it out, choosing the band with the fewest total
//! configurations and a digit with the fewest configurations in it.

use crate::tvec::{band_config_counts, c16, c16_bytes, c8, ALL, CELLS_3X3, C16, C8, S0, S1, S2, S3, S4, S5, S6, XX};


// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

// Masks eliminating band configurations inconsistent with placing a digit in
// element e of box peer p. A configuration places peer p's triad at exactly
// one element, so placement at e kills the 4 configurations placing it
// elsewhere. Row 3 is padding.
static PEER_X_ELEM_TO_CONFIG_MASK: [[[u16; 8]; 4]; 3] = [
    [
        [0, ALL, ALL, ALL, 0, ALL, 0, 0],
        [ALL, 0, ALL, ALL, ALL, 0, 0, 0],
        [ALL, ALL, 0, 0, ALL, ALL, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [ALL, ALL, 0, ALL, ALL, 0, 0, 0],
        [0, ALL, ALL, 0, ALL, ALL, 0, 0],
        [ALL, 0, ALL, ALL, 0, ALL, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [ALL, 0, ALL, 0, ALL, ALL, 0, 0],
        [ALL, ALL, 0, ALL, 0, ALL, 0, 0],
        [0, ALL, ALL, ALL, ALL, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
];

// Shuffle controls turning a vector holding a box's (positive or negative)
// triads in lanes 4..6 into a configuration-elimination message for the
// band, per box peer index, at each of the three "shifts" (a negative triad
// kills the configurations placing the digit there = shift 0; an asserted
// positive triad kills the configurations placing the digit at the other two
// elements = shifts 1 and 2).
static TRIADS_SHIFT0_TO_CONFIG_ELIMS: [[u16; 8]; 3] = [
    [S4, S5, S6, S6, S4, S5, XX, XX],
    [S5, S6, S4, S5, S6, S4, XX, XX],
    [S6, S4, S5, S4, S5, S6, XX, XX],
];
static TRIADS_SHIFT1_TO_CONFIG_ELIMS: [[u16; 8]; 3] = [
    [S5, S6, S4, S4, S5, S6, XX, XX],
    [S6, S4, S5, S6, S4, S5, XX, XX],
    [S4, S5, S6, S5, S6, S4, XX, XX],
];
static TRIADS_SHIFT2_TO_CONFIG_ELIMS: [[u16; 8]; 3] = [
    [S6, S4, S5, S5, S6, S4, XX, XX],
    [S4, S5, S6, S4, S5, S6, XX, XX],
    [S5, S6, S4, S6, S4, S5, XX, XX],
];

// The 16-lane pairings [box_j * 3 + box_i] = {table[box_j] (lo, for the
// horizontal band), table[box_i] (hi, for the vertical band)}.
const fn pair16(t: &[[u16; 8]; 3]) -> [[u16; 16]; 9] {
    let mut out = [[0u16; 16]; 9];
    let mut i = 0;
    while i < 3 {
        let mut j = 0;
        while j < 3 {
            let mut k = 0;
            while k < 8 {
                out[i * 3 + j][k] = t[i][k];
                out[i * 3 + j][8 + k] = t[j][k];
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    out
}

static TRIADS_SHIFT0_TO_CONFIG_ELIMS16: [[u16; 16]; 9] = pair16(&TRIADS_SHIFT0_TO_CONFIG_ELIMS);
static TRIADS_SHIFT1_TO_CONFIG_ELIMS16: [[u16; 16]; 9] = pair16(&TRIADS_SHIFT1_TO_CONFIG_ELIMS);
static TRIADS_SHIFT2_TO_CONFIG_ELIMS16: [[u16; 16]; 9] = pair16(&TRIADS_SHIFT2_TO_CONFIG_ELIMS);

// Two shuffles OR'd convert a configuration vector (duplicated across both
// 128-bit halves) into the 3x3 matrix of positive triads: each (peer, elem)
// is possible under exactly two configurations.
static SHUFFLE_CONFIGS_TO_TRIADS: [[u16; 16]; 2] = [
    [S0, S1, S2, XX, S2, S0, S1, XX, S1, S2, S0, XX, XX, XX, XX, XX],
    [S4, S5, S3, XX, S5, S3, S4, XX, S3, S4, S5, XX, XX, XX, XX, XX],
];

// Two shuffles OR'd convert one box peer's positive triads (lanes 0..2, with
// kAll planted in lane 3 as a no-op selector) into a box restriction mask:
// cells limited to their triad, negative triads to the union of the other
// two triads in the same (mini)line.
static POS_TRIADS_TO_CANDIDATES: [[[u16; 16]; 2]; 2] = [
    // horizontal
    [
        [S0, S0, S0, S1, S1, S1, S1, S2, S2, S2, S2, S0, S3, S3, S3, S3],
        [S0, S0, S0, S2, S1, S1, S1, S0, S2, S2, S2, S1, S3, S3, S3, S3],
    ],
    // vertical
    [
        [S0, S1, S2, S3, S0, S1, S2, S3, S0, S1, S2, S3, S1, S2, S0, S3],
        [S0, S1, S2, S3, S0, S1, S2, S3, S0, S1, S2, S3, S2, S0, S1, S3],
    ],
];



// Extracts horizontal triad literals (matrix lanes 3, 7, 11) into lanes 4..6.
// Controls for the fused triad-message build (see `triad_message`). A takes
// the horizontal triads that live in the low 128-bit lane (matrix lanes 3
// and 7) to positions 4 and 5, and passes the high lane through unchanged
// because the vertical triads already sit at positions 4..6 there. B picks
// the third horizontal triad (matrix lane 11) out of the *other* 128-bit
// lane, which is why its input is the half-swapped vector.
// Minimum candidates per lane: cells >= 1; negative triads >= 6, because
// exactly 3 of 9 digits live in a triad. Popcount equality with the minimum
// triggers assertion of everything left in the lane.
static BOX_MINIMUMS: [u16; 16] = [1, 1, 1, 6, 1, 1, 1, 6, 1, 1, 1, 6, 6, 6, 6, 0];

// Rotates band configuration lanes 0..5 cyclically (used to count per-digit
// configurations without extraction).
static CONFIG_ROTATE: [u16; 8] = [S1, S2, S3, S4, S5, S0, XX, XX];

// Plant kAll into lane 3 before shuffling positive triads into a box mask.
static LANE3_ALL: [u16; 8] = [0, 0, 0, ALL, 0, 0, 0, 0];

/// Per (digit, element) elimination mask for asserting a clue in its box:
/// clears other digits from the cell, the digit from all other cells, and
/// the digit from the two negative triads covering the cell.
const fn build_cell_assignment_elims() -> [[[u16; 16]; 16]; 9] {
    let mut t = [[[0u16; 16]; 16]; 9];
    let cells = [0usize, 1, 2, 4, 5, 6, 8, 9, 10];
    let mut ci = 0;
    while ci < 9 {
        let i = cells[ci];
        let mut v = 0;
        while v < 9 {
            let mut j = 0;
            while j < 15 {
                t[v][i][j] = if j == i {
                    ALL ^ (1 << v)
                } else if j / 4 < 3 && j % 4 < 3 {
                    1 << v
                } else if j / 4 == i / 4 || j % 4 == i % 4 {
                    1 << v
                } else {
                    0
                };
                j += 1;
            }
            v += 1;
        }
        ci += 1;
    }
    t
}

static CELL_ASSIGNMENT_ELIMS: [[[u16; 16]; 16]; 9] = build_cell_assignment_elims();

const DIV3: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];
const MOD3: [usize; 9] = [0, 1, 2, 0, 1, 2, 0, 1, 2];
const BOX_PEERS: [[[usize; 3]; 3]; 2] = [
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    [[0, 3, 6], [1, 4, 7], [2, 5, 8]],
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct Band {
    /// Lanes 0..5: per-configuration masks of digits still allowed to take
    /// that configuration in this band.
    configurations: C8,
    /// Pending configuration eliminations, applied at the next BandEliminate.
    eliminations: C8,
}

#[derive(Copy, Clone)]
struct TState {
    /// bands[0] horizontal, bands[1] vertical.
    bands: [[Band; 2]; 3],
    boxen: [C16; 9],
    /// Literals each box has already asserted and drawn the consequences of.
    /// Grows monotonically within a search node; the branch copy restores it.
    asserted: [C16; 9],
}

impl TState {
    #[inline]
    unsafe fn new() -> TState {
        let init = [ALL, ALL, ALL, ALL, ALL, ALL, 0, 0];
        TState {
            bands: [[Band { configurations: c8(&init), eliminations: C8::zero() }; 2]; 3],
            boxen: [C16::all(ALL); 9],
            asserted: [C16::all(0); 9],
        }
    }

    #[inline(always)]
    unsafe fn band(&mut self, vertical: usize, idx: usize) -> &mut Band {
        self.bands.get_unchecked_mut(idx).get_unchecked_mut(vertical)
    }
}

// ---------------------------------------------------------------------------
// Message passing between boxes and bands
// ---------------------------------------------------------------------------

/// Convert a band's configurations into its 3x3 matrix of positive triads
/// (rows = box peers, columns = elements).
#[inline(always)]
unsafe fn configurations_to_positive_triads(configurations: C8) -> C16 {
    let tmp = C16::from_parts(configurations, configurations);
    tmp.shuffle(c16(&SHUFFLE_CONFIGS_TO_TRIADS[0]))
        .or(tmp.shuffle(c16(&SHUFFLE_CONFIGS_TO_TRIADS[1])))
}

/// Convert one box peer's positive triads (lanes 0..2) into the box
/// restriction mask for the given orientation.
#[inline(always)]
unsafe fn positive_triads_to_box_candidates(triads: C8, orientation: usize) -> C16 {
    let with_all = triads.or(c8(&LANE3_ALL));
    let tmp = C16::from_parts(with_all, with_all);
    tmp.shuffle(c16(&POS_TRIADS_TO_CANDIDATES[orientation][0]))
        .or(tmp.shuffle(c16(&POS_TRIADS_TO_CANDIDATES[orientation][1])))
}


/// Turn newly asserted literals into further box eliminations plus band
/// elimination messages. See the port source for the full derivation.
#[inline(always)]
unsafe fn assertions_to_eliminations(
    assertions: C16,
    box_i: usize,
    box_j: usize,
    box_eliminations: &mut C16,
    h_band_eliminations: &mut C8,
    v_band_eliminations: &mut C8,
) {
    let cell_assertions_only = assertions.and(c16(&CELLS_3X3));
    // Row broadcast. A matrix row is exactly one 64-bit lane, so a two-step
    // shift-and-or carries each row's union up to its top element -- the
    // horizontal triad, the only lane that needs it, since inside the 3x3 the
    // row union is a subset of the box union that `new_box_elims` already
    // takes. The lower elements are left holding prefix unions, which are
    // subsets of that same box union and so change nothing.
    //
    // Four operations instead of three shuffles and three ors, and shifts
    // issue on ports the shuffles are competing for.
    let t = cell_assertions_only.or(cell_assertions_only.shift_rows_up2());
    let across_rows = t.or(t.shift_rows_up1());
    // Rotate three ways off the same source and OR as a balanced tree. The
    // obvious `x |= rot(x)` log-reduction is one shuffle cheaper, but it
    // chains permute -> or -> permute -> or, and a cross-lane permute costs 3
    // cycles: 8 cycles of latency against 5 for three independent permutes
    // plus a two-level OR. This sits on the loop-carried critical path, where
    // latency is worth more than the extra op.
    // Column broadcast. A matrix column is one element position repeated in
    // each of the four 64-bit lanes, so folding the register in half twice --
    // once across the 128-bit halves, once across the two lanes inside each
    // half -- unions all four rows into every row. Two shuffles and two ors
    // rather than three cross-lane permutes and three ors, and the second
    // fold is an in-lane `vpshufd` rather than another `vpermq`.
    // The 3x3 submatrix eliminates an asserted digit everywhere in the box;
    // margins pick up row/col broadcasts; asserted cells eliminate all bits.
    let new_box_elims = cell_assertions_only
        .box_and_column_unions()
        .or(across_rows)
        .or(cell_assertions_only.which_nonzero());
    // Keep the asserted candidate itself in its own cell. This replaces the
    // caller's mask rather than accumulating into it: the box has already had
    // every earlier elimination applied, so re-ORing them in only lengthens
    // the value the next `and_not` waits on.
    *box_eliminations = new_box_elims.xor(cell_assertions_only);

    // Negative triad assertions kill the configurations placing the digit
    // there (shift 0); asserted cells imply positive triads, killing the
    // configurations placing the digit at the other elements (shifts 1, 2).
    let hv_neg = assertions.triad_message();
    let hv_pos = new_box_elims.triad_message();
    let idx = box_j * 3 + box_i;
    let new_elims = hv_neg
        .shuffle(c16(&TRIADS_SHIFT0_TO_CONFIG_ELIMS16[idx]))
        .or(hv_pos.shuffle(c16(&TRIADS_SHIFT1_TO_CONFIG_ELIMS16[idx])))
        .or(hv_pos.shuffle(c16(&TRIADS_SHIFT2_TO_CONFIG_ELIMS16[idx])));
    *h_band_eliminations = h_band_eliminations.or(new_elims.get_lo());
    *v_band_eliminations = v_band_eliminations.or(new_elims.get_hi());
}

/// Restrict a box to the given candidates, propagate its internal clauses to
/// fixpoint, and forward resulting messages to its two bands.
/// Fast-path wrapper: most restriction messages are no-ops, and inlining
/// just the subset test into callers keeps that path free of the callee-save
/// spills the full body needs (LLVM does not shrink-wrap the vector saves).
#[inline(always)]
unsafe fn box_restrict<const FROM_VERTICAL: usize>(
    state: &mut TState,
    box_idx: usize,
    candidates: C16,
) -> bool {
    if state.boxen.get_unchecked(box_idx).subset_of(candidates) {
        return true;
    }
    box_restrict_full::<FROM_VERTICAL>(state, box_idx, candidates)
}

/// `inline(always)`: the design keeps three copies of the fixpoint loop
/// inlined inside each `band_eliminate_full` instantiation; without the
/// attribute the inlining is at the mercy of module-wide heuristics, and
/// adding the jcz engine to the crate was enough to flip LLVM into
/// outlining this (measured ~7-10% slower on the hard corpora).
#[inline(always)]
unsafe fn box_restrict_full<const FROM_VERTICAL: usize>(
    state: &mut TState,
    box_idx: usize,
    candidates: C16,
) -> bool {
    if !box_fixpoint(state, box_idx, candidates) {
        return false;
    }
    let box_i = *DIV3.get_unchecked(box_idx);
    let box_j = *MOD3.get_unchecked(box_idx);
    // Forward to band peers, visiting the opposite orientation first.
    if FROM_VERTICAL != 0 {
        band_eliminate::<0>(state, box_i, box_j) && band_eliminate::<1>(state, box_j, box_i)
    } else {
        band_eliminate::<1>(state, box_j, box_i) && band_eliminate::<0>(state, box_i, box_j)
    }
}

/// Restrict a box and run its internal clauses to fixpoint, leaving the
/// resulting messages queued on its two bands but not delivering them.
#[inline(always)]
unsafe fn box_fixpoint(state: &mut TState, box_idx: usize, candidates: C16) -> bool {
    // SAFETY: box_idx < 9 (from BOX_PEERS), so DIV3/MOD3/boxen indexing is in
    // bounds; box_i/box_j < 3 index the bands array.
    let mut eliminating = state.boxen.get_unchecked(box_idx).and_not(candidates);

    let box_i = *DIV3.get_unchecked(box_idx);
    let box_j = *MOD3.get_unchecked(box_idx);
    let box_minimums = c16(&BOX_MINIMUMS);

    // Carry the box in a register across iterations and write it back once on
    // exit. Nothing inside the loop reads it, and on the contradiction exit
    // the caller discards this state, so the intermediate stores were dead.
    let mut cells = *state.boxen.get_unchecked(box_idx);
    let mut asserted = *state.asserted.get_unchecked(box_idx);
    loop {
        cells = cells.and_not(eliminating);
        let counts = cells.popcounts9();
        if counts.any_less_than(box_minimums) {
            return false;
        }
        // Literal assertions: lanes at their minimum assert everything left,
        // plus hidden singles along the exactly-one row/column clauses.
        //
        // A lane asserts candidate `d` when it is at its cardinality minimum,
        // or when no other lane of its matrix row still holds `d`, or none of
        // its matrix column does. Writing `R` and `C` for the union of a
        // lane's three row-peers and three column-peers, that is
        //
        //     cells & (triggered | ~R | ~C)  ==  cells & ~(R & C & ~triggered)
        //
        // which is the whole step in three logic operations on top of the six
        // rotations. The peer union is also all the row/column scan needs:
        // the previous form built, per lane, the candidates appearing in two
        // or more lanes of the group -- a strictly stronger quantity, since a
        // candidate the lane itself does not hold cannot survive the final
        // `cells &` anyway. Same assertions, nine fewer vector operations,
        // and the dependency chain drops from a six-deep serial accumulation
        // to a two-level OR tree off the rotations.
        let triggered = counts.which_equal(box_minimums);
        let row_peers = cells.row_peers();
        let col_peers = cells.col_peers();
        let assertions = cells.and_not(row_peers.and(col_peers).and_not(triggered));

        // Loop exit. Everything downstream -- the box's own elimination
        // closure and both band messages -- is a function of `assertions`
        // alone, and each of those functions distributes over union, so an
        // iteration that asserts nothing new can only re-derive consequences
        // already accumulated. Testing that instead of testing whether
        // `eliminating` grew moves the exit *above* the closure, so the
        // terminating iteration skips it: about 40 of the loop's 80
        // instructions, on 70% of iterations.
        //
        // It is the weaker test -- assertions can grow while every consequence
        // is already eliminated -- which costs an extra iteration in that case
        // and reaches the same fixpoint. Keeping both tests measured worse
        // than either alone: the second branch costs more than the iteration.
        //
        // `assertions` only grows within a search node: a lane at its minimum
        // either keeps its contents or falls below the minimum, and the
        // cardinality check above rejects the latter first. So the subset test
        // is an equality test, and carrying the previous value per box is what
        // lets a re-entered box (~2.5 entries per easy puzzle) exit at once.
        if assertions.subset_of(asserted) {
            break;
        }
        asserted = assertions;

        // The band elimination accumulators are one register each; copy out,
        // update, copy back.
        let mut h_elims = state.bands.get_unchecked(box_i).get_unchecked(0).eliminations;
        let mut v_elims = state.bands.get_unchecked(box_j).get_unchecked(1).eliminations;
        assertions_to_eliminations(
            assertions,
            box_i,
            box_j,
            &mut eliminating,
            &mut h_elims,
            &mut v_elims,
        );
        state.bands.get_unchecked_mut(box_i).get_unchecked_mut(0).eliminations = h_elims;
        state.bands.get_unchecked_mut(box_j).get_unchecked_mut(1).eliminations = v_elims;
    }
    *state.boxen.get_unchecked_mut(box_idx) = cells;
    *state.asserted.get_unchecked_mut(box_idx) = asserted;
    true
}

/// One box's fixpoint, out of line, for the initialization pass: nine copies
/// of the inlined loop would be a lot of code for a path taken once.
#[inline(never)]
unsafe fn seed_box_assertions(state: &mut TState, box_idx: usize) -> bool {
    box_fixpoint(state, box_idx, C16::all(ALL))
}

/// Fast-path wrapper: a band with no pending eliminations intersecting its
/// configurations is a no-op, which is the common case; see `box_restrict`.
#[inline(always)]
unsafe fn band_eliminate<const VERTICAL: usize>(
    state: &mut TState,
    band_idx: usize,
    from_peer: usize,
) -> bool {
    let band = state.band(VERTICAL, band_idx);
    if !band.configurations.intersects(band.eliminations) {
        return true;
    }
    band_eliminate_full::<VERTICAL>(state, band_idx, from_peer)
}

/// Apply a band's pending configuration eliminations, run the triad-count
/// clause (a triad with exactly 3 remaining digits pins all three), and
/// forward the restriction messages to the band's three box peers.
///
/// `sysv64`: under the Windows x64 ABI xmm6-xmm15 are callee-saved, so every
/// entry to this recursive function spills and reloads ten vector registers.
/// The SysV convention treats all of them as volatile. Rust has no way to
/// make an ABI conditional, and `sysv64` does not exist off x86-64, so the
/// body lives in an always-inlined inner function and this is the one place
/// the two architectures need separate declarations. On aarch64 the default
/// ABI already leaves only the low 64 bits of v8-v15 callee-saved, and the
/// register file is twice as large, so there is nothing to recover here.
#[cfg(target_arch = "x86_64")]
unsafe extern "sysv64" fn band_eliminate_full<const VERTICAL: usize>(
    state: &mut TState,
    band_idx: usize,
    from_peer: usize,
) -> bool {
    band_eliminate_body::<VERTICAL>(state, band_idx, from_peer)
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn band_eliminate_full<const VERTICAL: usize>(
    state: &mut TState,
    band_idx: usize,
    from_peer: usize,
) -> bool {
    band_eliminate_body::<VERTICAL>(state, band_idx, from_peer)
}

#[inline(always)]
unsafe fn band_eliminate_body<const VERTICAL: usize>(
    state: &mut TState,
    band_idx: usize,
    from_peer: usize,
) -> bool {
    let band = state.band(VERTICAL, band_idx);
    band.configurations = band.configurations.and_not(band.eliminations);
    let configurations = band.configurations;

    let triads = configurations_to_positive_triads(configurations);
    let counts = triads.popcounts9();

    // Triads with exactly 3 candidates assert all three digits; kill the
    // configurations placing those digits elsewhere. Once is enough for
    // nearly all of the benefit.
    let asserting = triads.and(counts.which_equal(C16::all(3)));
    let lo = asserting.get_lo();
    let hi = asserting.get_hi();
    let band = state.band(VERTICAL, band_idx);
    band.configurations = band.configurations.and_not(
        lo.rotate_cols()
            .shuffle(c8(&TRIADS_SHIFT1_TO_CONFIG_ELIMS[0]))
            .or(lo.rotate_cols().shuffle(c8(&TRIADS_SHIFT2_TO_CONFIG_ELIMS[0])))
            .or(lo.shuffle(c8(&TRIADS_SHIFT1_TO_CONFIG_ELIMS[1]))),
    );
    band.configurations = band.configurations.and_not(
        lo.shuffle(c8(&TRIADS_SHIFT2_TO_CONFIG_ELIMS[1]))
            .or(hi.rotate_cols().shuffle(c8(&TRIADS_SHIFT1_TO_CONFIG_ELIMS[2])))
            .or(hi.rotate_cols().shuffle(c8(&TRIADS_SHIFT2_TO_CONFIG_ELIMS[2]))),
    );
    let triads = configurations_to_positive_triads(band.configurations);

    // Send box restriction messages, returning to the inbound peer last.
    // SAFETY: from_peer < 3 at every call site.
    match from_peer {
        0 => band_forward::<VERTICAL, 0>(state, band_idx, triads),
        1 => band_forward::<VERTICAL, 1>(state, band_idx, triads),
        _ => band_forward::<VERTICAL, 2>(state, band_idx, triads),
    }
}

/// Forward a band's restriction messages to its three box peers, visiting the
/// inbound peer (`FROM`) last.
///
/// `FROM` is a const parameter rather than a value because the visit order is
/// a runtime permutation of three vectors: with a value the compiler has to
/// spill all three to the stack and reload through a variable index, putting a
/// store-forwarding stall on the critical path of every call. Specializing
/// turns the selection into compile-time register naming.
#[inline(always)]
unsafe fn band_forward<const VERTICAL: usize, const FROM: usize>(
    state: &mut TState,
    band_idx: usize,
    triads: C16,
) -> bool {
    // SAFETY: band_idx < 3 at every call site; FROM < 3 by construction.
    let box_peers = BOX_PEERS.get_unchecked(VERTICAL).get_unchecked(band_idx);
    const fn nxt<const F: usize>(k: usize) -> usize {
        (F + k) % 3
    }
    let (p0, p1, p2) = (nxt::<FROM>(1), nxt::<FROM>(2), FROM);
    box_restrict::<VERTICAL>(
        state,
        *box_peers.get_unchecked(p0),
        positive_triads_to_box_candidates(peer_triad(triads, p0), VERTICAL),
    ) && box_restrict::<VERTICAL>(
        state,
        *box_peers.get_unchecked(p1),
        positive_triads_to_box_candidates(peer_triad(triads, p1), VERTICAL),
    ) && box_restrict::<VERTICAL>(
        state,
        *box_peers.get_unchecked(p2),
        positive_triads_to_box_candidates(peer_triad(triads, p2), VERTICAL),
    )
}

/// One band peer's positive triads. `p` is always a constant after inlining,
/// so the selection folds away entirely.
#[inline(always)]
unsafe fn peer_triad(triads: C16, p: usize) -> C8 {
    if p == 0 {
        triads.get_lo()
    } else if p == 1 {
        triads.get_lo().rotate_cols()
    } else {
        triads.get_hi()
    }
}

// ---------------------------------------------------------------------------
// Branching
// ---------------------------------------------------------------------------

const NONE: u32 = u32::MAX;

/// Choose the unfixed band with the fewest configurations, then a digit in it
/// with the fewest configurations, preferring 2, then 3, then more. Returns
/// (band 0..6 or NONE, digit mask replicated across config lanes).
#[inline]
unsafe fn choose_band_and_value(state: &TState) -> (u32, C8, bool) {
    // A fixed band has exactly 9 configuration bits (one per digit), so
    // subtracting 10 puts every fixed band above every unfixed one.
    let counts = band_config_counts([
        state.bands[0][0].configurations,
        state.bands[1][0].configurations,
        state.bands[2][0].configurations,
        state.bands[0][1].configurations,
        state.bands[1][1].configurations,
        state.bands[2][1].configurations,
    ]);
    let config_minpos = counts.minpos_after_sub(10);
    if config_minpos & 0xff00 != 0 {
        return (NONE, C8::zero(), false);
    }
    let best_band = config_minpos >> 16;
    // SAFETY: best_band < 6 -- fixed bands and padding lanes carry huge
    // adjusted counts, so minpos lands on an unfixed band lane.
    let configurations = state
        .bands
        .get_unchecked(*MOD3.get_unchecked(best_band as usize) as usize)
        .get_unchecked(*DIV3.get_unchecked(best_band as usize) as usize)
        .configurations;

    // Count per-digit configurations across lanes by rotate-accumulate.
    let ctrl = c8(&CONFIG_ROTATE);
    let mut one = configurations;
    let mut rotated = one.shuffle(ctrl); // 1
    let mut two = one.and(rotated);
    one = one.or(rotated);
    rotated = rotated.shuffle(ctrl); // 2
    let mut three = two.and(rotated);
    two = two.or(one.and(rotated));
    one = one.or(rotated);
    rotated = rotated.shuffle(ctrl); // 3
    let mut four = three.and(rotated);
    three = three.or(two.and(rotated));
    two = two.or(one.and(rotated));
    one = one.or(rotated);
    rotated = rotated.shuffle(ctrl); // 4
    four = four.or(three.and(rotated));
    three = three.or(two.and(rotated));
    two = two.or(one.and(rotated));
    one = one.or(rotated);
    rotated = rotated.shuffle(ctrl); // 5
    four = four.or(three.and(rotated));
    three = three.or(two.and(rotated));
    two = two.or(one.and(rotated));

    let only_two = two.and_not(three);
    if !only_two.all_zero() {
        (best_band, only_two.low_bit_per_lane(), true)
    } else {
        let only_three = three.and_not(four);
        if !only_three.all_zero() {
            (best_band, only_three.low_bit_per_lane(), false)
        } else {
            (best_band, four.low_bit_per_lane(), false)
        }
    }
}

struct Solver {
    limit: u64,
    num_solutions: u64,
    /// Written only when the solution count reaches the limit.
    solution: core::mem::MaybeUninit<TState>,
    #[cfg(feature = "stats")]
    guesses: u64,
}

impl Solver {
    unsafe fn branch_on_band_and_value<const VERTICAL: usize>(
        &mut self,
        band_idx: usize,
        value_mask: C8,
        pair: bool,
        state: &mut TState,
    ) {
        #[cfg(feature = "stats")]
        {
            self.guesses += 1;
        }
        let value_configurations = state.band(VERTICAL, band_idx).configurations.and(value_mask);
        let assignment_elims = value_configurations.clear_low_bit();
        let negation_elims = value_configurations.xor(assignment_elims);
        // Child order. When the digit has exactly two configurations both
        // children are commitments and the order is a pure heuristic:
        // exploring the higher configuration first measured ~5% faster on the
        // deep-search corpora. With three or more, the commit child is the
        // stronger constraint and trying it first stays better.
        let (first, second) = if pair {
            (negation_elims, assignment_elims)
        } else {
            (assignment_elims, negation_elims)
        };
        let mut copy = *state;
        {
            let b = copy.band(VERTICAL, band_idx);
            b.eliminations = b.eliminations.or(first);
        }
        if band_eliminate::<VERTICAL>(&mut copy, band_idx, 0) {
            self.count_solutions(&mut copy);
            if self.num_solutions == self.limit {
                return;
            }
        }
        {
            let b = state.band(VERTICAL, band_idx);
            b.eliminations = b.eliminations.or(second);
        }
        if band_eliminate::<VERTICAL>(state, band_idx, 0) {
            self.count_solutions(state);
        }
    }

    unsafe fn count_solutions(&mut self, state: &mut TState) {
        let (band, value_mask, pair) = choose_band_and_value(state);
        if band == NONE {
            // All bands fixed: this is a solution.
            self.num_solutions += 1;
            if self.num_solutions == self.limit {
                self.solution.write(*state);
            }
        } else if band < 3 {
            self.branch_on_band_and_value::<0>(band as usize, value_mask, pair, state);
        } else {
            self.branch_on_band_and_value::<1>(band as usize - 3, value_mask, pair, state);
        }
    }
}

// ---------------------------------------------------------------------------
// Initialization and extraction
// ---------------------------------------------------------------------------

/// Per-cell indexing: [box_i, box_j, box, elem_i, elem_j, elem].
const BOX_INDEXING: [[u8; 6]; 81] = build_box_indexing();

/// Per-cell [row, column], for accumulating the initial digit masks.
const ROW_COL_OF: [[u8; 2]; 81] = build_row_col();

const fn build_row_col() -> [[u8; 2]; 81] {
    let mut t = [[0u8; 2]; 81];
    let mut cell = 0;
    while cell < 81 {
        t[cell] = [(cell / 9) as u8, (cell % 9) as u8];
        cell += 1;
    }
    t
}

/// Byte shuffles spreading three packed u16 digit masks over the 4x4 matrix:
/// ROW_BCAST copies entry i into every lane of matrix row i, COL_BCAST copies
/// entry j into every lane of matrix column j. The source is the quadword of
/// three masks broadcast to all four 64-bit lanes, so both 128-bit halves can
/// reach every entry.
static ROW_BCAST: [u8; 32] = build_bcast(true);
static COL_BCAST: [u8; 32] = build_bcast(false);

const fn build_bcast(by_row: bool) -> [u8; 32] {
    let mut t = [0u8; 32];
    let mut b = 0;
    while b < 32 {
        let lane = b / 2;
        let idx = if by_row { lane / 4 } else { lane % 4 };
        t[b] = (2 * idx + b % 2) as u8;
        b += 1;
    }
    t
}

const fn build_box_indexing() -> [[u8; 6]; 81] {
    let mut t = [[0u8; 6]; 81];
    let mut cell = 0;
    while cell < 81 {
        let box_i = cell / 27;
        let box_j = (cell % 9) / 3;
        let elem_i = (cell / 9) % 3;
        let elem_j = cell % 3;
        t[cell] = [
            box_i as u8,
            box_j as u8,
            (box_i * 3 + box_j) as u8,
            elem_i as u8,
            elem_j as u8,
            (elem_i * 4 + elem_j) as u8,
        ];
        cell += 1;
    }
    t
}

unsafe fn init_clue(state: &mut TState, cell: usize, digit: u8) {
    let value = (digit - 1) as usize;
    let ix = BOX_INDEXING.get_unchecked(cell);
    let (box_i, box_j) = (ix[0] as usize, ix[1] as usize);
    let bx = ix[2] as usize;
    let (elem_i, elem_j) = (ix[3] as usize, ix[4] as usize);
    let elem = ix[5] as usize;
    // SAFETY: all indices are bounded by the BOX_INDEXING table contents
    // (box < 9, peers/elems < 3, elem < 12) and value < 9.
    // Restrict the clue's own box without propagating yet.
    *state.boxen.get_unchecked_mut(bx) = state.boxen.get_unchecked(bx).and_not(c16(
        CELL_ASSIGNMENT_ELIMS.get_unchecked(value).get_unchecked(elem),
    ));
    // Merge configuration eliminations into both bands.
    let candidate = C8::all(1 << value);
    {
        let b = state.band(0, box_i);
        b.eliminations = c8(PEER_X_ELEM_TO_CONFIG_MASK.get_unchecked(box_j).get_unchecked(elem_i))
            .and(candidate)
            .or(b.eliminations);
    }
    {
        let b = state.band(1, box_j);
        b.eliminations = c8(PEER_X_ELEM_TO_CONFIG_MASK.get_unchecked(box_i).get_unchecked(elem_j))
            .and(candidate)
            .or(b.eliminations);
    }
}

/// Remove from every box the digits already placed in its rows and columns
/// by a clue in one of its band peers.
///
/// `init_clue` only ever touches the clue's own box and the two bands' pending
/// configuration eliminations, so this deduction -- the most elementary one
/// there is -- otherwise has to travel box -> band configurations -> positive
/// triads -> box before it lands. Doing it directly costs nine vector updates
/// and removes a fifth of the propagation on easy puzzles, where that first
/// cascade is most of the work: box entries drop 22.4 -> 14.1 per puzzle and
/// fixpoint iterations 47.4 -> 37.4.
///
/// Lanes already down to a single candidate are left alone. Those are the clue
/// cells (whose own digit is in `row_digits` by definition, so eliminating it
/// would empty the cell) plus any cell the clue eliminations happened to
/// reduce to one. Skipping the latter costs nothing: a genuine conflict there
/// is still found by propagation, exactly as it is today.
#[inline]
unsafe fn seed_peer_eliminations(
    state: &mut TState,
    row_digits: &[u16; 12],
    col_digits: &[u16; 12],
) {
    let row_ctrl = c16_bytes(&ROW_BCAST);
    let col_ctrl = c16_bytes(&COL_BCAST);
    let cells_only = c16(&CELLS_3X3);
    let one = C16::all(1);
    for bx in 0..9 {
        // SAFETY: bx < 9, so the band offsets are at most 6 and the quadword
        // loads read entries 6..9 of arrays padded to 12.
        let rp = row_digits.as_ptr().add(*DIV3.get_unchecked(bx) * 3);
        let cp = col_digits.as_ptr().add(*MOD3.get_unchecked(bx) * 3);
        let r = (rp as *const u64).read_unaligned();
        let c = (cp as *const u64).read_unaligned();
        let spread = C16::splat_u64(r)
            .shuffle(row_ctrl)
            .or(C16::splat_u64(c).shuffle(col_ctrl));
        let cells = *state.boxen.get_unchecked(bx);
        let settled = cells.popcounts9().which_equal(one);
        *state.boxen.get_unchecked_mut(bx) =
            cells.and_not(spread.and(cells_only).and_not(settled));
    }
}

unsafe fn init_and_propagate(state: &mut TState, clues: &[u8; 81]) -> bool {
    // Digits already placed in each row and column, for `seed_peer_eliminations`.
    // Padded to 12 so the quadword loads below stay in bounds.
    let mut row_digits = [0u16; 12];
    let mut col_digits = [0u16; 12];
    // Clue positions as a bitmask, then a bit-scan, so the random clue
    // pattern costs no branch mispredictions.
    // SAFETY: every bit set in the masks is a cell index below 81, keeping
    // `clues`, `ROW_COL_OF` and the padded digit arrays all in range.
    let (mut lo, mut hi) = crate::clue_scan::clue_masks(clues);
    let mut enter = |cell: usize| {
        let digit = *clues.get_unchecked(cell);
        init_clue(state, cell, digit);
        let rc = ROW_COL_OF.get_unchecked(cell);
        let bit = 1u16 << (digit - 1);
        *row_digits.get_unchecked_mut(rc[0] as usize) |= bit;
        *col_digits.get_unchecked_mut(rc[1] as usize) |= bit;
    };
    while lo != 0 {
        enter(lo.trailing_zeros() as usize);
        lo &= lo - 1;
    }
    while hi != 0 {
        enter(64 + hi.trailing_zeros() as usize);
        hi &= hi - 1;
    }
    seed_peer_eliminations(state, &row_digits, &col_digits);
    // Drain each box before the bands run. The seeded eliminations can leave
    // a box strictly inside what its bands are about to ask for, and
    // `box_restrict`'s subset fast path would then skip it -- so what the
    // seeding let the box deduce would never reach the band configurations,
    // and the branching heuristic, which reads only those, would go in blind
    // (17-clue guesses measured 0.38 -> 1.39 per puzzle without this). The
    // drain sits before the cascade, where it costs one pass over the boxes
    // and the cascade below then runs exactly once; in the hybrid, puzzles
    // that never branch are the jcz engine's and rarely arrive here.
    for bx in 0..9 {
        if !seed_box_assertions(state, bx) {
            return false;
        }
    }
    // One batched eliminate per band nearly always completes initialization.
    band_eliminate::<0>(state, 0, 1)
        && band_eliminate::<1>(state, 0, 1)
        && band_eliminate::<0>(state, 1, 2)
        && band_eliminate::<1>(state, 1, 2)
        && band_eliminate::<0>(state, 2, 0)
        && band_eliminate::<1>(state, 2, 0)
}

unsafe fn extract_solution(state: &TState, out: &mut [u8; 81]) {
    for bx in 0..9 {
        let rows = state.boxen[bx].extract_rows_u64();
        let base = DIV3[bx] * 27 + MOD3[bx] * 3;
        for r in 0..3 {
            let row = rows[r];
            out[base + 9 * r] = (row & 0xffff).trailing_zeros() as u8 + 1;
            out[base + 9 * r + 1] = ((row >> 16) & 0xffff).trailing_zeros() as u8 + 1;
            out[base + 9 * r + 2] = ((row >> 32) & 0xffff).trailing_zeros() as u8 + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[cfg(feature = "stats")]
use crate::GUESSES;

fn run(clues: &[u8; 81], limit: u64) -> (u64, Option<[u8; 81]>) {
    unsafe {
        let mut state = TState::new();
        if !init_and_propagate(&mut state, clues) {
            return (0, None);
        }
        let mut solver = Solver {
            limit,
            num_solutions: 0,
            solution: core::mem::MaybeUninit::uninit(),
            #[cfg(feature = "stats")]
            guesses: 0,
        };
        solver.count_solutions(&mut state);
        #[cfg(feature = "stats")]
        GUESSES.fetch_add(solver.guesses, core::sync::atomic::Ordering::Relaxed);
        if solver.num_solutions == 0 {
            return (0, None);
        }
        let mut out = [0u8; 81];
        if solver.num_solutions == limit {
            // SAFETY: written when the count reached the limit.
            extract_solution(solver.solution.assume_init_ref(), &mut out);
        } else {
            return (solver.num_solutions, None);
        }
        (solver.num_solutions, Some(out))
    }
}

/// Solve to the first solution.
pub fn solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    let (n, sol) = run(clues, 1);
    if n > 0 { sol } else { None }
}

/// Count solutions up to `limit`.
pub fn count_solutions(clues: &[u8; 81], limit: u64) -> u64 {
    run(clues, limit).0
}

// ---------------------------------------------------------------------------
// BSD-2-Clause notice for the ported design (t-dillon/tdoku):
//
// Copyright (c) 2019 Tom Dillon
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------------
