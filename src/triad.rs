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

use core::arch::x86_64::*;

const ALL: u16 = 0x1ff;
const XX: u16 = 0xffff;

// Byte-pair shuffle selectors for 16-bit lanes 0..7 (pshufb operates on
// bytes; 0xffff's high bits make pshufb emit zero).
const S0: u16 = 0x0100;
const S1: u16 = 0x0302;
const S2: u16 = 0x0504;
const S3: u16 = 0x0706;
const S4: u16 = 0x0908;
const S5: u16 = 0x0b0a;
const S6: u16 = 0x0d0c;
const S7: u16 = 0x0f0e;

// ---------------------------------------------------------------------------
// Vector wrappers: C8 = 8 x u16 (one band / half a box), C16 = 16 x u16 (a
// box as a 4x4 matrix; lanes 0..15 row-major).
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct C8(__m128i);

#[derive(Copy, Clone)]
struct C16(__m256i);

#[inline(always)]
unsafe fn c8(a: &[u16; 8]) -> C8 {
    C8(_mm_loadu_si128(a.as_ptr() as *const __m128i))
}

#[inline(always)]
unsafe fn c16(a: &[u16; 16]) -> C16 {
    C16(_mm256_loadu_si256(a.as_ptr() as *const __m256i))
}

impl C8 {
    #[inline(always)]
    unsafe fn all(v: u16) -> C8 {
        C8(_mm_set1_epi16(v as i16))
    }
    #[inline(always)]
    unsafe fn zero() -> C8 {
        C8(_mm_setzero_si128())
    }
    #[inline(always)]
    unsafe fn and(self, o: C8) -> C8 {
        C8(_mm_and_si128(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn or(self, o: C8) -> C8 {
        C8(_mm_or_si128(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn xor(self, o: C8) -> C8 {
        C8(_mm_xor_si128(self.0, o.0))
    }
    /// self & !o
    #[inline(always)]
    unsafe fn and_not(self, o: C8) -> C8 {
        C8(_mm_andnot_si128(o.0, self.0))
    }
    #[inline(always)]
    unsafe fn shuffle(self, ctrl: C8) -> C8 {
        C8(_mm_shuffle_epi8(self.0, ctrl.0))
    }
    /// Swap the two rows of a 2x4 view (64-bit halves).
    #[inline(always)]
    unsafe fn rotate_cols(self) -> C8 {
        C8(_mm_shuffle_epi32::<0b01001110>(self.0))
    }
    #[inline(always)]
    unsafe fn all_zero(self) -> bool {
        _mm_testz_si128(self.0, self.0) != 0
    }
    #[inline(always)]
    unsafe fn intersects(self, o: C8) -> bool {
        _mm_testz_si128(self.0, o.0) == 0
    }
    /// Total set bits across the vector.
    #[inline(always)]
    unsafe fn popcount(self) -> u32 {
        let lo = _mm_cvtsi128_si64(self.0) as u64;
        let hi = _mm_extract_epi64::<1>(self.0) as u64;
        lo.count_ones() + hi.count_ones()
    }
    /// Lowest set bit of each 16-bit lane.
    #[inline(always)]
    unsafe fn low_bit_per_lane(self) -> C8 {
        let neg = _mm_sub_epi16(_mm_setzero_si128(), self.0);
        C8(_mm_and_si128(self.0, neg))
    }
    /// Clear the lowest set bit of the vector viewed as one long integer.
    #[inline(always)]
    unsafe fn clear_low_bit(self) -> C8 {
        let cmp = _mm_cmpgt_epi64(self.0, _mm_setzero_si128());
        let one = _mm_andnot_si128(_mm_slli_si128::<1>(cmp), _mm_srli_epi64::<63>(cmp));
        C8(_mm_and_si128(self.0, _mm_sub_epi64(self.0, one)))
    }
    /// (min value, lane) over lanes after subtracting `floor`; packed as
    /// value in bits 0..16, lane in bits 16..19 (via phminposuw).
    #[inline(always)]
    unsafe fn minpos_after_sub(self, floor: u16) -> u32 {
        let adj = _mm_sub_epi16(self.0, _mm_set1_epi16(floor as i16));
        _mm_cvtsi128_si32(_mm_minpos_epu16(adj)) as u32
    }
}

impl C16 {
    #[inline(always)]
    unsafe fn all(v: u16) -> C16 {
        C16(_mm256_set1_epi16(v as i16))
    }
    #[inline(always)]
    unsafe fn from_parts(lo: C8, hi: C8) -> C16 {
        C16(_mm256_set_m128i(hi.0, lo.0))
    }
    #[inline(always)]
    unsafe fn get_lo(self) -> C8 {
        C8(_mm256_castsi256_si128(self.0))
    }
    #[inline(always)]
    unsafe fn get_hi(self) -> C8 {
        C8(_mm256_extracti128_si256::<1>(self.0))
    }
    #[inline(always)]
    unsafe fn and(self, o: C16) -> C16 {
        C16(_mm256_and_si256(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn or(self, o: C16) -> C16 {
        C16(_mm256_or_si256(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn xor(self, o: C16) -> C16 {
        C16(_mm256_xor_si256(self.0, o.0))
    }
    /// self & !o
    #[inline(always)]
    unsafe fn and_not(self, o: C16) -> C16 {
        C16(_mm256_andnot_si256(o.0, self.0))
    }
    #[inline(always)]
    unsafe fn shuffle(self, ctrl: C16) -> C16 {
        C16(_mm256_shuffle_epi8(self.0, ctrl.0))
    }
    #[inline(always)]
    unsafe fn subset_of(self, o: C16) -> bool {
        _mm256_testc_si256(o.0, self.0) != 0
    }
    #[inline(always)]
    unsafe fn intersects(self, o: C16) -> bool {
        _mm256_testz_si256(self.0, o.0) == 0
    }
    #[inline(always)]
    unsafe fn which_equal(self, o: C16) -> C16 {
        C16(_mm256_cmpeq_epi16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn which_nonzero(self) -> C16 {
        C16(_mm256_cmpgt_epi16(self.0, _mm256_setzero_si256()))
    }
    #[inline(always)]
    unsafe fn any_less_than(self, o: C16) -> bool {
        let lt = _mm256_cmpgt_epi16(o.0, self.0);
        _mm256_movemask_epi8(lt) != 0
    }
    /// Per-lane popcount, assuming the 7 high bits of every lane are zero.
    #[inline(always)]
    unsafe fn popcounts9(self) -> C16 {
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
            2, 3, 3, 4,
        );
        let mask4 = _mm256_set1_epi16(0x0f);
        let sum_0_3 = _mm256_shuffle_epi8(lookup, _mm256_and_si256(self.0, mask4));
        let sum_4_7 = _mm256_shuffle_epi8(lookup, _mm256_srli_epi16::<4>(self.0));
        let sum_0_7 = _mm256_add_epi16(sum_0_3, sum_4_7);
        C16(_mm256_add_epi16(sum_0_7, _mm256_srli_epi16::<8>(self.0)))
    }
    /// Rotate the elements of each matrix row left by one.
    #[inline(always)]
    unsafe fn rotate_rows(self) -> C16 {
        let ctrl = _mm256_setr_epi8(
            2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 2, 3, 4, 5, 6, 7, 0, 1, 10,
            11, 12, 13, 14, 15, 8, 9,
        );
        C16(_mm256_shuffle_epi8(self.0, ctrl))
    }
    /// Rotate the elements of each matrix row left by two.
    #[inline(always)]
    unsafe fn rotate_rows2(self) -> C16 {
        C16(_mm256_shuffle_epi32::<0b10110001>(self.0))
    }
    /// Rotate the matrix rows up by one (element (r,c) <- (r+1,c)).
    #[inline(always)]
    unsafe fn rotate_cols(self) -> C16 {
        C16(_mm256_permute4x64_epi64::<0b00111001>(self.0))
    }
    /// Rotate the matrix rows up by two.
    #[inline(always)]
    unsafe fn rotate_cols2(self) -> C16 {
        C16(_mm256_permute4x64_epi64::<0b01001110>(self.0))
    }
    #[inline(always)]
    unsafe fn extract_rows_u64(self) -> [u64; 4] {
        [
            _mm256_extract_epi64::<0>(self.0) as u64,
            _mm256_extract_epi64::<1>(self.0) as u64,
            _mm256_extract_epi64::<2>(self.0) as u64,
            _mm256_extract_epi64::<3>(self.0) as u64,
        ]
    }
}

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

static CELL3X3_MASK: [u16; 16] = [
    ALL, ALL, ALL, 0, ALL, ALL, ALL, 0, ALL, ALL, ALL, 0, 0, 0, 0, 0,
];

// Row rotations restricted to the 3x3 submatrix (margins kept in place).
static ROW_ROTATE_3X3_1: [u16; 16] = [
    S1, S2, S0, S3, S5, S6, S4, S7, S1, S2, S0, S3, S4, S5, S6, S7,
];
static ROW_ROTATE_3X3_2: [u16; 16] = [
    S2, S0, S1, S3, S6, S4, S5, S7, S2, S0, S1, S3, S4, S5, S6, S7,
];

// Extracts horizontal triad literals (matrix lanes 3, 7, 11) into lanes 4..6.
static H_TRIADS_CTRL: [u16; 16] = [XX, XX, XX, XX, S3, S7, XX, XX, XX, XX, XX, XX, XX, XX, S3, XX];

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
}

impl TState {
    #[inline]
    unsafe fn new() -> TState {
        let init = [ALL, ALL, ALL, ALL, ALL, ALL, 0, 0];
        TState {
            bands: [[Band { configurations: c8(&init), eliminations: C8::zero() }; 2]; 3],
            boxen: [C16::all(ALL); 9],
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

/// Extract horizontal triad literals into lanes 4..6 of a C8.
#[inline(always)]
unsafe fn horizontal_triads(cells: C16) -> C8 {
    let split = cells.shuffle(c16(&H_TRIADS_CTRL));
    split.get_lo().or(split.get_hi())
}

/// Vertical triad literals already sit in lanes 4..6 of the high half.
#[inline(always)]
unsafe fn vertical_triads(cells: C16) -> C8 {
    cells.get_hi()
}

/// Hidden singles over the exactly-one clauses along matrix rows or columns
/// (depending on the rotate), OR'd into `assertions`.
#[inline(always)]
unsafe fn gather_triad_clause_assertions<F: Fn(C16) -> C16>(
    cells: C16,
    rotate: F,
    assertions: C16,
) -> C16 {
    let mut one_or_more = cells;
    let mut rotated = rotate(cells);
    let mut two_or_more = one_or_more.and(rotated);
    one_or_more = one_or_more.or(rotated);
    rotated = rotate(rotated);
    two_or_more = one_or_more.and(rotated).or(two_or_more);
    one_or_more = one_or_more.or(rotated);
    rotated = rotate(rotated);
    two_or_more = one_or_more.and(rotated).or(two_or_more);
    cells.and_not(two_or_more).or(assertions)
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
    let cell_assertions_only = assertions.and(c16(&CELL3X3_MASK));
    // Broadcast assertions across their rows and columns.
    let mut across_rows = cell_assertions_only;
    across_rows = across_rows.or(across_rows.rotate_rows());
    across_rows = across_rows.or(across_rows.rotate_rows2());
    let mut across_cols = cell_assertions_only;
    across_cols = across_cols.or(across_cols.rotate_cols());
    across_cols = across_cols.or(across_cols.rotate_cols2());
    // The 3x3 submatrix eliminates an asserted digit everywhere in the box;
    // margins pick up row/col broadcasts; asserted cells eliminate all bits.
    let new_box_elims = across_cols
        .or(across_cols.shuffle(c16(&ROW_ROTATE_3X3_1)))
        .or(across_cols.shuffle(c16(&ROW_ROTATE_3X3_2)))
        .or(across_rows)
        .or(cell_assertions_only.which_nonzero());
    // Keep the asserted candidate itself in its own cell.
    *box_eliminations = new_box_elims.xor(cell_assertions_only).or(*box_eliminations);

    // Negative triad assertions kill the configurations placing the digit
    // there (shift 0); asserted cells imply positive triads, killing the
    // configurations placing the digit at the other elements (shifts 1, 2).
    let hv_neg = C16::from_parts(horizontal_triads(assertions), vertical_triads(assertions));
    let hv_pos = C16::from_parts(
        horizontal_triads(new_box_elims),
        vertical_triads(new_box_elims),
    );
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

unsafe fn box_restrict_full<const FROM_VERTICAL: usize>(
    state: &mut TState,
    box_idx: usize,
    candidates: C16,
) -> bool {
    // SAFETY: box_idx < 9 (from BOX_PEERS), so DIV3/MOD3/boxen indexing is in
    // bounds; box_i/box_j < 3 index the bands array.
    let mut eliminating = state.boxen.get_unchecked(box_idx).and_not(candidates);

    let box_i = *DIV3.get_unchecked(box_idx);
    let box_j = *MOD3.get_unchecked(box_idx);
    let box_minimums = c16(&BOX_MINIMUMS);

    loop {
        let cells = state.boxen.get_unchecked(box_idx).and_not(eliminating);
        *state.boxen.get_unchecked_mut(box_idx) = cells;
        let counts = cells.popcounts9();
        if counts.any_less_than(box_minimums) {
            return false;
        }
        // Literal assertions: lanes at their minimum assert everything left,
        // plus hidden singles along the exactly-one row/column clauses.
        let triggered = counts.which_equal(box_minimums);
        let mut assertions = cells.and(triggered);
        assertions = gather_triad_clause_assertions(cells, |v| v.rotate_rows(), assertions);
        assertions = gather_triad_clause_assertions(cells, |v| v.rotate_cols(), assertions);

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

        if !eliminating.intersects(*state.boxen.get_unchecked(box_idx)) {
            break;
        }
    }
    // Forward to band peers, visiting the opposite orientation first.
    if FROM_VERTICAL != 0 {
        band_eliminate::<0>(state, box_i, box_j) && band_eliminate::<1>(state, box_j, box_i)
    } else {
        band_eliminate::<1>(state, box_j, box_i) && band_eliminate::<0>(state, box_i, box_j)
    }
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
unsafe fn band_eliminate_full<const VERTICAL: usize>(
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
    // SAFETY: from_peer < 3 and band_idx < 3 at every call site.
    let peer = [
        *MOD3.get_unchecked(from_peer + 1),
        *MOD3.get_unchecked(from_peer + 2),
        from_peer,
    ];
    let box_peers = BOX_PEERS.get_unchecked(VERTICAL).get_unchecked(band_idx);
    let peer_triads = [triads.get_lo(), triads.get_lo().rotate_cols(), triads.get_hi()];
    // Unrolled: a 3-iteration loop materializes the triads on the stack.
    let (p0, p1, p2) = (peer[0], peer[1], peer[2]);
    box_restrict::<VERTICAL>(
        state,
        *box_peers.get_unchecked(p0),
        positive_triads_to_box_candidates(*peer_triads.get_unchecked(p0), VERTICAL),
    ) && box_restrict::<VERTICAL>(
        state,
        *box_peers.get_unchecked(p1),
        positive_triads_to_box_candidates(*peer_triads.get_unchecked(p1), VERTICAL),
    ) && box_restrict::<VERTICAL>(
        state,
        *box_peers.get_unchecked(p2),
        positive_triads_to_box_candidates(*peer_triads.get_unchecked(p2), VERTICAL),
    )
}

// ---------------------------------------------------------------------------
// Branching
// ---------------------------------------------------------------------------

const NONE: u32 = u32::MAX;

/// Choose the unfixed band with the fewest configurations, then a digit in it
/// with the fewest configurations, preferring 2, then 3, then more. Returns
/// (band 0..6 or NONE, digit mask replicated across config lanes).
#[inline]
unsafe fn choose_band_and_value(state: &TState) -> (u32, C8) {
    // A fixed band has exactly 9 configuration bits (one per digit).
    let counts = [
        state.bands[0][0].configurations.popcount() as u16,
        state.bands[1][0].configurations.popcount() as u16,
        state.bands[2][0].configurations.popcount() as u16,
        state.bands[0][1].configurations.popcount() as u16,
        state.bands[1][1].configurations.popcount() as u16,
        state.bands[2][1].configurations.popcount() as u16,
        0xffff,
        0xffff,
    ];
    let config_minpos = c8(&counts).minpos_after_sub(10);
    if config_minpos & 0xff00 != 0 {
        return (NONE, C8::zero());
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
        (best_band, only_two.low_bit_per_lane())
    } else {
        let only_three = three.and_not(four);
        if !only_three.all_zero() {
            (best_band, only_three.low_bit_per_lane())
        } else {
            (best_band, four.low_bit_per_lane())
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
        state: &mut TState,
    ) {
        #[cfg(feature = "stats")]
        {
            self.guesses += 1;
        }
        let value_configurations = state.band(VERTICAL, band_idx).configurations.and(value_mask);
        // Try the lowest configuration by eliminating the others...
        let mut copy = *state;
        let assignment_elims = value_configurations.clear_low_bit();
        {
            let b = copy.band(VERTICAL, band_idx);
            b.eliminations = b.eliminations.or(assignment_elims);
        }
        if band_eliminate::<VERTICAL>(&mut copy, band_idx, 0) {
            self.count_solutions(&mut copy);
            if self.num_solutions == self.limit {
                return;
            }
        }
        // ...then rule it out.
        let negation_elims = value_configurations.xor(assignment_elims);
        {
            let b = state.band(VERTICAL, band_idx);
            b.eliminations = b.eliminations.or(negation_elims);
        }
        if band_eliminate::<VERTICAL>(state, band_idx, 0) {
            self.count_solutions(state);
        }
    }

    unsafe fn count_solutions(&mut self, state: &mut TState) {
        let (band, value_mask) = choose_band_and_value(state);
        if band == NONE {
            // All bands fixed: this is a solution.
            self.num_solutions += 1;
            if self.num_solutions == self.limit {
                self.solution.write(*state);
            }
        } else if band < 3 {
            self.branch_on_band_and_value::<0>(band as usize, value_mask, state);
        } else {
            self.branch_on_band_and_value::<1>(band as usize - 3, value_mask, state);
        }
    }
}

// ---------------------------------------------------------------------------
// Initialization and extraction
// ---------------------------------------------------------------------------

/// Per-cell indexing: [box_i, box_j, box, elem_i, elem_j, elem].
const BOX_INDEXING: [[u8; 6]; 81] = build_box_indexing();

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

unsafe fn init_and_propagate(state: &mut TState, clues: &[u8; 81]) -> bool {
    // Build a bitmask of clue positions branchlessly (three overlapping
    // 32-byte loads; the third covers cells 49..81), then bit-scan it, so
    // the random clue pattern costs no branch mispredictions.
    let zero = _mm256_setzero_si256();
    let p = clues.as_ptr();
    let m_a = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
        _mm256_loadu_si256(p as *const __m256i),
        zero,
    )) as u32;
    let m_b = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
        _mm256_loadu_si256(p.add(32) as *const __m256i),
        zero,
    )) as u32;
    let m_c = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
        _mm256_loadu_si256(p.add(49) as *const __m256i),
        zero,
    )) as u32;
    let mut lo = !(m_a as u64 | (m_b as u64) << 32); // clue cells 0..64
    let mut hi = (!m_c >> 15) & 0x1ffff; // clue cells 64..81
    while lo != 0 {
        let cell = lo.trailing_zeros() as usize;
        lo &= lo - 1;
        init_clue(state, cell, *clues.get_unchecked(cell));
    }
    while hi != 0 {
        let cell = 64 + hi.trailing_zeros() as usize;
        hi &= hi - 1;
        init_clue(state, cell, *clues.get_unchecked(cell));
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
