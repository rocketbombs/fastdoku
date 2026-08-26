//! AVX2 solver core: dual-orientation bitboards, one register per digit.
//!
//! The band engine in `band.rs` keeps only the horizontal orientation and has
//! to transpose on demand to reason about columns; that transposition
//! dominated its stall-time inference. Here each digit's whole 81-cell board
//! is held **twice** in a single 256-bit register:
//!
//! ```text
//!   lane:   0     1     2      3      4      5     6  7
//!         band0 band1 band2 stack0 stack1 stack2  -  -
//! ```
//!
//! Each lane is a 27-bit mask of cells where that digit is still possible.
//! Horizontal lane `b` uses bit `9*(r%3) + c`; vertical lane `s` uses bit
//! `9*(c%3) + r`. The two halves are redundant, and keeping them in sync is
//! free: a cell's peers are a *fixed* set, so assignment is one AND against a
//! precomputed 256-bit mask that clears the peers in both orientations at
//! once. Nothing is ever transposed at run time, and every band scan sees all
//! six bands in one instruction.
//!
//! On top of that the engine adds cross-digit reasoning the scalar engine
//! could not afford: a band's 9 minirows each hold exactly 3 digits, a
//! cardinality constraint over the 9x9 digit-by-minirow incidence matrix. In
//! this layout that matrix is just the 9 digit registers, so the whole rule
//! costs a handful of vector ops (see `band_inference`).

use core::arch::x86_64::*;

#[cfg(feature = "stats")]
use crate::GUESSES;

const BAND_ALL: u32 = 0x07FF_FFFF;

/// Cross-digit cardinality inference (see `band_inference`). Toggle to A/B
/// its value against its cost.
const CARDINALITY: bool = true;

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

const fn h_of(cell: usize) -> (usize, usize) {
    let (r, c) = (cell / 9, cell % 9);
    (r / 3, 9 * (r % 3) + c)
}

const fn v_of(cell: usize) -> (usize, usize) {
    let (r, c) = (cell / 9, cell % 9);
    (c / 3, 9 * (c % 3) + r)
}

/// Assigning digit d at a cell: clears d from the cell's 20 peers in both
/// orientations while keeping the cell itself.
static PEER: [[u32; 8]; 81] = build_peer();

const fn build_peer() -> [[u32; 8]; 81] {
    let mut t = [[0u32; 8]; 81];
    let mut cell = 0;
    while cell < 81 {
        let (r0, c0) = (cell / 9, cell % 9);
        let mut lanes = [0u32; 8];
        let mut i = 0;
        while i < 6 {
            lanes[i] = BAND_ALL;
            i += 1;
        }
        let mut p = 0;
        while p < 81 {
            let (r, c) = (p / 9, p % 9);
            if p != cell && (r == r0 || c == c0 || (r / 3 == r0 / 3 && c / 3 == c0 / 3)) {
                let (hb, hi) = h_of(p);
                lanes[hb] &= !(1u32 << hi);
                let (vs, vi) = v_of(p);
                lanes[3 + vs] &= !(1u32 << vi);
            }
            p += 1;
        }
        t[cell] = lanes;
        cell += 1;
    }
    t
}

/// (lane << 5) | bit for each cell in the horizontal orientation.
const CELL_H: [u8; 81] = build_cell_h();

const fn build_cell_h() -> [u8; 81] {
    let mut t = [0u8; 81];
    let mut cell = 0;
    while cell < 81 {
        let (b, i) = h_of(cell);
        t[cell] = ((b << 5) | i) as u8;
        cell += 1;
    }
    t
}

/// (lane << 5) | bit for each cell in the vertical orientation.
const CELL_V: [u8; 81] = build_cell_v();

const fn build_cell_v() -> [u8; 81] {
    let mut t = [0u8; 81];
    let mut cell = 0;
    while cell < 81 {
        let (s, i) = v_of(cell);
        t[cell] = (((3 + s) << 5) | i) as u8;
        cell += 1;
    }
    t
}

/// Cell index for (line, position). Lines 0..9 are grid rows (the horizontal
/// lanes' lines); lines 9..18 are grid columns (the vertical lanes' lines).
const LINE_CELL: [[u8; 9]; 18] = build_line_cell();

const fn build_line_cell() -> [[u8; 9]; 18] {
    let mut t = [[0u8; 9]; 18];
    let mut line = 0;
    while line < 18 {
        let mut p = 0;
        while p < 9 {
            t[line][p] = if line < 9 {
                (line * 9 + p) as u8
            } else {
                (p * 9 + (line - 9)) as u8
            };
            p += 1;
        }
        line += 1;
    }
    t
}

// A digit occupies exactly 3 minirows of a band -- one per row, one per box --
// so its occupancy must contain a 3x3 permutation matrix.
const PERMS: [u16; 6] = build_perms();

const fn build_perms() -> [u16; 6] {
    let p = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let mut out = [0u16; 6];
    let mut k = 0;
    while k < 6 {
        let mut m = 0u16;
        let mut r = 0;
        while r < 3 {
            m |= 1 << (3 * r + p[k][r]);
            r += 1;
        }
        out[k] = m;
        k += 1;
    }
    out
}

const fn support_of(v: u16) -> u16 {
    let mut s = 0u16;
    let mut k = 0;
    while k < 6 {
        if PERMS[k] & !v == 0 {
            s |= PERMS[k];
        }
        k += 1;
    }
    s
}

const fn expand_of(m: u16) -> u32 {
    let mut mask = 0u32;
    let mut t = 0;
    while t < 9 {
        if m & (1 << t) != 0 {
            mask |= 0b111 << ((t / 3) * 9 + 3 * (t % 3));
        }
        t += 1;
    }
    mask
}

/// Occupancy -> 27-bit cell mask of the exact (permutation-support)
/// reduction. Subsumes locked candidates in both directions. 0 means the
/// digit has no valid placement in the band.
static SUP_EXPAND: [u32; 512] = build_sup_expand();

const fn build_sup_expand() -> [u32; 512] {
    let mut t = [0u32; 512];
    let mut v = 0;
    while v < 512 {
        t[v] = expand_of(support_of(v as u16));
        v += 1;
    }
    t
}

/// Occupancy -> 27-bit cell mask, with no reduction applied.
static MINIROW_EXPAND: [u32; 512] = build_minirow_expand();

const fn build_minirow_expand() -> [u32; 512] {
    let mut t = [0u32; 512];
    let mut v = 0;
    while v < 512 {
        t[v] = expand_of(v as u16);
        v += 1;
    }
    t
}

/// 9-bit line -> 3-bit box occupancy, for the scalar per-lane shrink.
const SHRINK: [u16; 512] = build_shrink();

const fn build_shrink() -> [u16; 512] {
    let mut t = [0u16; 512];
    let mut v = 0;
    while v < 512 {
        let mut s = 0u16;
        if v & 0o007 != 0 {
            s |= 1;
        }
        if v & 0o070 != 0 {
            s |= 2;
        }
        if v & 0o700 != 0 {
            s |= 4;
        }
        t[v] = s;
        v += 1;
    }
    t
}

/// (lane << 5) | bit of the same cell in the other orientation. Assignment
/// keeps both orientations in sync via precomputed masks, but a reduction
/// discovered while scanning one lane has to be mirrored to its partner.
const MIRROR: [[u8; 27]; 6] = build_mirror();

const fn build_mirror() -> [[u8; 27]; 6] {
    let mut t = [[0u8; 27]; 6];
    let mut lane = 0;
    while lane < 6 {
        let mut i = 0;
        while i < 27 {
            let (ol, ob) = if lane < 3 {
                // horizontal band `lane`, bit i -> vertical
                let (r, c) = (3 * lane + i / 9, i % 9);
                (3 + c / 3, 9 * (c % 3) + r)
            } else {
                // vertical stack `lane-3`, bit i -> horizontal
                let (r, c) = (i % 9, 3 * (lane - 3) + i / 9);
                (r / 3, 9 * (r % 3) + c)
            };
            t[lane][i] = ((ol << 5) | ob) as u8;
            i += 1;
        }
        lane += 1;
    }
    t
}

/// Dirty-unit masks selecting the horizontal and vertical lanes of every
/// digit. Rows propagate eagerly; columns are deferred to the stall phase,
/// which keeps easy puzzles from paying for column inference they never need.
const H_UNITS: u64 = unit_mask(0b000111);
const V_UNITS: u64 = unit_mask(0b111000);

const fn unit_mask(lanes: u64) -> u64 {
    let mut m = 0u64;
    let mut d = 0;
    while d < 9 {
        m |= lanes << (d * 6);
        d += 1;
    }
    m
}

/// Companion minirows of each minirow: those sharing its row or its box.
const MINIROW_LINKED: [u32; 9] = build_minirow_linked();

const fn build_minirow_linked() -> [u32; 9] {
    let mut t = [0u32; 9];
    let mut i = 0;
    while i < 9 {
        t[i] = ((0b111u32 << (3 * (i / 3))) | (0o111u32 << (i % 3))) & !(1 << i);
        i += 1;
    }
    t
}

// ---------------------------------------------------------------------------
// Vector helpers
//
// Loads/stores are unaligned: on Zen 3 there is no penalty when the address
// happens to be aligned, and it keeps plain `[u32; 8]` arrays usable as
// vector operands without relying on struct field placement.
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn vload(p: &[u32; 8]) -> __m256i {
    _mm256_loadu_si256(p.as_ptr() as *const __m256i)
}

#[inline(always)]
unsafe fn vstore(p: &mut [u32; 8], v: __m256i) {
    _mm256_storeu_si256(p.as_mut_ptr() as *mut __m256i, v)
}

/// Reduce each lane's 27-bit cell mask to its 9-bit minirow occupancy:
/// bit 3r+j is set iff row r of box j holds a candidate.
#[inline(always)]
unsafe fn shrink6(v: __m256i) -> __m256i {
    // OR each aligned group of 3 bits into its low bit.
    let t = _mm256_or_si256(
        _mm256_or_si256(v, _mm256_srli_epi32(v, 1)),
        _mm256_srli_epi32(v, 2),
    );
    let a = _mm256_and_si256(t, _mm256_set1_epi32(0x1249249));
    // Compact the stride-3 bits: {9r+0,9r+3,9r+6} -> {9r+0,9r+1,9r+2}.
    let b = _mm256_and_si256(
        _mm256_or_si256(a, _mm256_srli_epi32(a, 2)),
        _mm256_set1_epi32(0x10C8643),
    );
    let c = _mm256_and_si256(
        _mm256_or_si256(b, _mm256_srli_epi32(b, 4)),
        _mm256_set1_epi32(0x1C0E07),
    );
    // Pack the three lines down to bits 0..8.
    _mm256_and_si256(
        _mm256_or_si256(
            _mm256_or_si256(c, _mm256_srli_epi32(c, 6)),
            _mm256_srli_epi32(c, 12),
        ),
        _mm256_set1_epi32(0x1FF),
    )
}

#[inline(always)]
unsafe fn gather(table: &[u32; 512], idx: __m256i) -> __m256i {
    _mm256_i32gather_epi32::<4>(table.as_ptr() as *const i32, idx)
}

/// Bitmask of lanes that are entirely zero.
#[inline(always)]
unsafe fn zero_lanes(v: __m256i) -> u32 {
    let c = _mm256_cmpeq_epi32(v, _mm256_setzero_si256());
    _mm256_movemask_ps(_mm256_castsi256_ps(c)) as u32
}

/// Bitmask of lanes where a and b differ.
#[inline(always)]
unsafe fn diff_lanes(a: __m256i, b: __m256i) -> u32 {
    let c = _mm256_cmpeq_epi32(a, b);
    (!(_mm256_movemask_ps(_mm256_castsi256_ps(c)) as u32)) & 0xFF
}

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------

struct Queue {
    buf: [u16; 256],
    len: usize,
}

impl Queue {
    #[inline]
    fn new() -> Self {
        Queue { buf: [0; 256], len: 0 }
    }
    #[inline]
    fn push(&mut self, d: usize, cell: usize) {
        // Scans can queue the same cell from several directions, so the queue
        // can exceed 81 entries; 256 is far beyond anything reachable, but
        // clamp rather than risk a write past the end.
        if self.len < 256 {
            self.buf[self.len] = ((d << 7) | cell) as u16;
            self.len += 1;
        }
    }
    #[inline]
    fn pop(&mut self) -> Option<(usize, usize)> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            let e = self.buf[self.len] as usize;
            Some((e >> 7, e & 127))
        }
    }
    #[inline]
    fn clear(&mut self) {
        self.len = 0;
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct State {
    /// Per digit: the whole board in both orientations (see module docs).
    b: [[u32; 8]; 9],
    /// Unsolved cells per horizontal band.
    unsolved: [u32; 3],
    /// Per digit: 18 bits marking lines (9 rows, then 9 columns) that already
    /// hold that digit, so solved lines are not rescanned as singles.
    placed: [u32; 9],
    /// One bit per (digit, lane) unit needing a band reduction, at index
    /// `d * 6 + lane`. Fine-grained so propagation only revisits what
    /// actually changed.
    dirty: u64,
    /// A board changed; a naked-single scan is due.
    dirty_cells: bool,
    /// A board changed; a cardinality pass is due.
    tdirty: bool,
    n_unsolved: u8,
}

impl State {
    fn new() -> Self {
        let mut b = [[0u32; 8]; 9];
        let mut d = 0;
        while d < 9 {
            let mut l = 0;
            while l < 6 {
                b[d][l] = BAND_ALL;
                l += 1;
            }
            d += 1;
        }
        State {
            b,
            unsolved: [BAND_ALL; 3],
            placed: [0; 9],
            // An untouched lane holds every candidate and reduces to itself,
            // so nothing needs scanning until a clue is placed.
            dirty: 0,
            dirty_cells: true,
            tdirty: true,
            n_unsolved: 81,
        }
    }

    /// Place digit `d` in `cell`. Returns false on contradiction.
    #[inline]
    unsafe fn assign(&mut self, d: usize, cell: usize) -> bool {
        let h = *CELL_H.get_unchecked(cell) as usize;
        let (hb, hi) = (h >> 5, h & 31);
        let bit = 1u32 << hi;
        if *self.b.get_unchecked(d).get_unchecked(hb) & bit == 0 {
            return false; // digit already eliminated here
        }
        if *self.unsolved.get_unchecked(hb) & bit == 0 {
            return true; // already solved, and solved as d
        }
        *self.unsolved.get_unchecked_mut(hb) &= !bit;
        self.n_unsolved -= 1;

        // d loses its 20 peers across all six of its lanes: one vector AND
        // against a precomputed mask, both orientations at once.
        let peer = vload(PEER.get_unchecked(cell));
        let nd = _mm256_and_si256(vload(self.b.get_unchecked(d)), peer);
        vstore(self.b.get_unchecked_mut(d), nd);
        self.dirty |= 0x3F << (d * 6);

        // Every other digit only loses this one cell, which lives in exactly
        // two lanes -- far cheaper to clear scalar than to rewrite all nine
        // registers.
        let vv = *CELL_V.get_unchecked(cell) as usize;
        let (vs, vi) = (vv >> 5, vv & 31);
        let unit = (1u64 << hb) | (1u64 << vs);
        for e in 0..9 {
            if e == d {
                continue;
            }
            let row = self.b.get_unchecked_mut(e);
            if *row.get_unchecked(hb) & bit != 0 {
                *row.get_unchecked_mut(hb) &= !bit;
                *row.get_unchecked_mut(vs) &= !(1u32 << vi);
                self.dirty |= unit << (e * 6);
            }
        }

        let (r, c) = (cell / 9, cell % 9);
        *self.placed.get_unchecked_mut(d) |= (1 << r) | (1 << (9 + c));
        self.dirty_cells = true;
        self.tdirty = true;
        true
    }

    /// Exact band reduction for one (digit, lane) unit, plus hidden singles
    /// along that lane's three lines.
    ///
    /// Scalar on purpose. A propagation event usually dirties one or two of
    /// the 54 units, so vectorising across a digit's six lanes would do six
    /// times the work to fill the register. The vector width is spent on the
    /// cross-digit passes instead, where all nine digits are always involved.
    #[inline]
    unsafe fn scan_unit(&mut self, d: usize, lane: usize, q: &mut Queue) -> bool {
        let a = *self.b.get_unchecked(d).get_unchecked(lane);
        let s = *SHRINK.get_unchecked((a & 511) as usize)
            | *SHRINK.get_unchecked(((a >> 9) & 511) as usize) << 3
            | *SHRINK.get_unchecked((a >> 18) as usize) << 6;
        let na = a & *SUP_EXPAND.get_unchecked(s as usize);
        if na == 0 {
            return false; // no valid placement for this digit in this band
        }
        if na != a {
            *self.b.get_unchecked_mut(d).get_unchecked_mut(lane) = na;
            // Mirror the reduction into the other orientation.
            let mut gone = a & !na;
            while gone != 0 {
                let i = gone.trailing_zeros() as usize;
                gone &= gone - 1;
                let m = *MIRROR.get_unchecked(lane).get_unchecked(i) as usize;
                let (ol, ob) = (m >> 5, m & 31);
                *self.b.get_unchecked_mut(d).get_unchecked_mut(ol) &= !(1u32 << ob);
                self.dirty |= 1u64 << (d * 6 + ol);
            }
            self.dirty_cells = true;
            self.tdirty = true;
        }
        // Hidden singles on this lane's three lines.
        let placed = *self.placed.get_unchecked(d);
        for r in 0..3 {
            let line = lane * 3 + r; // grid row, or 9 + grid column
            if placed & (1 << line) != 0 {
                continue;
            }
            let bits = (na >> (9 * r)) & 511;
            if bits & bits.wrapping_sub(1) == 0 {
                // bits == 0 cannot occur: the reduction above rejects it.
                let p = bits.trailing_zeros() as usize;
                let cell = *LINE_CELL.get_unchecked(line).get_unchecked(p) as usize;
                q.push(d, cell);
            }
        }
        true
    }

    /// Naked singles, and cells left with no candidate at all.
    #[inline]
    unsafe fn scan_naked(&mut self, q: &mut Queue) -> bool {
        let mut one = _mm256_setzero_si256();
        let mut two = _mm256_setzero_si256();
        for d in 0..9 {
            let m = vload(self.b.get_unchecked(d));
            two = _mm256_or_si256(two, _mm256_and_si256(one, m));
            one = _mm256_or_si256(one, m);
        }
        let mut ones = [0u32; 8];
        let mut twos = [0u32; 8];
        vstore(&mut ones, one);
        vstore(&mut twos, two);
        for b in 0..3 {
            let uns = *self.unsolved.get_unchecked(b);
            let o = *ones.get_unchecked(b);
            if uns & !o != 0 {
                return false; // a cell has no candidates left
            }
            let mut naked = o & !*twos.get_unchecked(b) & uns;
            while naked != 0 {
                let i = naked.trailing_zeros() as usize;
                naked &= naked - 1;
                let cell = (b * 3 + i / 9) * 9 + i % 9;
                for d in 0..9 {
                    if *self.b.get_unchecked(d).get_unchecked(b) >> i & 1 != 0 {
                        q.push(d, cell);
                        break;
                    }
                }
            }
        }
        true
    }

    /// Cross-digit cardinality inference over all six bands.
    ///
    /// Each of a band's 9 minirows spans 3 cells and so holds exactly 3
    /// distinct digits. Viewing a band as a 9x9 digit-by-minirow incidence
    /// matrix, the per-digit permutation structure constrains its rows (done
    /// in `scan_digit`) while this constrains its columns:
    ///
    /// - fewer than 3 digits can reach a minirow -> contradiction;
    /// - exactly 3 can -> all three are pinned there, so none of them may use
    ///   another minirow sharing that row or box.
    ///
    /// In this layout the matrix columns are lanes across the 9 digit
    /// registers, so counting how many digits reach each minirow of every
    /// band is a single pass of vector ops.
    #[inline]
    unsafe fn band_inference(&mut self) -> bool {
        let mut occ = [[0u32; 8]; 9];
        let mut one = _mm256_setzero_si256();
        let mut two = _mm256_setzero_si256();
        let mut three = _mm256_setzero_si256();
        let mut four = _mm256_setzero_si256();
        for d in 0..9 {
            // Boards are already band-reduced here, so the raw occupancy is
            // its own support and needs no table lookup.
            let s = shrink6(vload(self.b.get_unchecked(d)));
            vstore(occ.get_unchecked_mut(d), s);
            four = _mm256_or_si256(four, _mm256_and_si256(three, s));
            three = _mm256_or_si256(three, _mm256_and_si256(two, s));
            two = _mm256_or_si256(two, _mm256_and_si256(one, s));
            one = _mm256_or_si256(one, s);
        }
        // Every minirow of every real band must be reachable by >= 3 digits.
        if diff_lanes(three, _mm256_set1_epi32(0x1FF)) & 0x3F != 0 {
            return false;
        }
        let exact = _mm256_andnot_si256(four, three);
        if zero_lanes(exact) == 0xFF {
            return true; // nothing pinned anywhere
        }

        let mut ex = [0u32; 8];
        vstore(&mut ex, exact);
        let mut changed = false;
        for lane in 0..6 {
            let mut e = *ex.get_unchecked(lane);
            while e != 0 {
                let t = e.trailing_zeros() as usize;
                e &= e - 1;
                let tb = 1u32 << t;
                let linked = *MINIROW_LINKED.get_unchecked(t);
                // The 3 digits that can reach this minirow all must, so none
                // of them may also use a minirow in its row or box.
                for d in 0..9 {
                    let o = *occ.get_unchecked(d).get_unchecked(lane);
                    if o & tb != 0 && o & linked != 0 {
                        *occ.get_unchecked_mut(d).get_unchecked_mut(lane) = o & !linked;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return true;
        }

        // Push the refined occupancies back onto the cell masks.
        for d in 0..9 {
            let keep = gather(&MINIROW_EXPAND, vload(occ.get_unchecked(d)));
            let v = vload(self.b.get_unchecked(d));
            let nv = _mm256_and_si256(v, keep);
            let changed_lanes = diff_lanes(nv, v);
            if changed_lanes != 0 {
                if zero_lanes(nv) & 0x3F != 0 {
                    return false;
                }
                vstore(self.b.get_unchecked_mut(d), nv);
                // Mirror the pruned cells into the other orientation.
                let mut old = [0u32; 8];
                let mut new = [0u32; 8];
                vstore(&mut old, v);
                vstore(&mut new, nv);
                for lane in 0..6 {
                    if changed_lanes >> lane & 1 == 0 {
                        continue;
                    }
                    let mut gone = old.get_unchecked(lane) & !new.get_unchecked(lane);
                    while gone != 0 {
                        let i = gone.trailing_zeros() as usize;
                        gone &= gone - 1;
                        let m = *MIRROR.get_unchecked(lane).get_unchecked(i) as usize;
                        let (ol, ob) = (m >> 5, m & 31);
                        *self.b.get_unchecked_mut(d).get_unchecked_mut(ol) &= !(1u32 << ob);
                        self.dirty |= 1u64 << (d * 6 + ol);
                    }
                    self.dirty |= 1u64 << (d * 6 + lane);
                }
                self.dirty_cells = true;
                self.tdirty = true;
            }
        }
        true
    }

    fn propagate(&mut self, q: &mut Queue) -> bool {
        unsafe {
            loop {
                while let Some((d, cell)) = q.pop() {
                    if !self.assign(d, cell) {
                        q.clear();
                        return false;
                    }
                }
                let hot = self.dirty & H_UNITS;
                if hot != 0 {
                    let u = hot.trailing_zeros() as usize;
                    self.dirty &= !(1u64 << u);
                    if !self.scan_unit(u / 6, u % 6, q) {
                        q.clear();
                        return false;
                    }
                    continue;
                }
                if self.dirty_cells {
                    self.dirty_cells = false;
                    if !self.scan_naked(q) {
                        q.clear();
                        return false;
                    }
                    continue;
                }
                if self.n_unsolved == 0 {
                    return true;
                }
                // Stalled on rows: now pay for the column direction.
                let cold = self.dirty & V_UNITS;
                if cold != 0 {
                    let u = cold.trailing_zeros() as usize;
                    self.dirty &= !(1u64 << u);
                    if !self.scan_unit(u / 6, u % 6, q) {
                        q.clear();
                        return false;
                    }
                    continue;
                }
                if !CARDINALITY || !self.tdirty {
                    return true;
                }
                self.tdirty = false;
                if !self.band_inference() {
                    q.clear();
                    return false;
                }
                if self.dirty == 0 && !self.dirty_cells {
                    return true;
                }
            }
        }
    }

    /// Branch point: a bivalue cell if one exists, else fewest candidates.
    #[inline]
    unsafe fn pick(&self) -> (usize, [u8; 9], usize) {
        let mut one = _mm256_setzero_si256();
        let mut two = _mm256_setzero_si256();
        let mut three = _mm256_setzero_si256();
        for d in 0..9 {
            let m = vload(self.b.get_unchecked(d));
            three = _mm256_or_si256(three, _mm256_and_si256(two, m));
            two = _mm256_or_si256(two, _mm256_and_si256(one, m));
            one = _mm256_or_si256(one, m);
        }
        let mut bivs = [0u32; 8];
        vstore(&mut bivs, _mm256_andnot_si256(three, two));
        for b in 0..3 {
            let m = *bivs.get_unchecked(b) & *self.unsolved.get_unchecked(b);
            if m != 0 {
                let i = m.trailing_zeros() as usize;
                return self.cell_digits((b * 3 + i / 9) * 9 + i % 9);
            }
        }
        // No bivalue cell: take the fewest-candidate cell.
        let mut best = (0usize, [0u8; 9], 10usize);
        for b in 0..3 {
            let mut uns = *self.unsolved.get_unchecked(b);
            while uns != 0 {
                let i = uns.trailing_zeros() as usize;
                uns &= uns - 1;
                let g = self.cell_digits((b * 3 + i / 9) * 9 + i % 9);
                if g.2 < best.2 {
                    best = g;
                    if best.2 == 3 {
                        return best;
                    }
                }
            }
        }
        best
    }

    #[inline]
    unsafe fn cell_digits(&self, cell: usize) -> (usize, [u8; 9], usize) {
        let h = *CELL_H.get_unchecked(cell) as usize;
        let (hb, hi) = (h >> 5, h & 31);
        let mut digits = [0u8; 9];
        let mut n = 0;
        for d in 0..9 {
            if *self.b.get_unchecked(d).get_unchecked(hb) >> hi & 1 != 0 {
                *digits.get_unchecked_mut(n) = d as u8;
                n += 1;
            }
        }
        (cell, digits, n)
    }

    unsafe fn write_grid(&self, out: &mut [u8; 81]) {
        for cell in 0..81 {
            let h = *CELL_H.get_unchecked(cell) as usize;
            let (hb, hi) = (h >> 5, h & 31);
            for d in 0..9 {
                if *self.b.get_unchecked(d).get_unchecked(hb) >> hi & 1 != 0 {
                    *out.get_unchecked_mut(cell) = d as u8 + 1;
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn search(st: &State, q: &mut Queue, limit: u64, count: &mut u64, out: &mut [u8; 81]) {
    if st.n_unsolved == 0 {
        if *count == 0 {
            unsafe { st.write_grid(out) };
        }
        *count += 1;
        return;
    }
    let (cell, digits, n) = unsafe { st.pick() };
    for i in 0..n {
        let d = digits[i] as usize;
        #[cfg(feature = "stats")]
        GUESSES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let mut s2 = *st;
        q.push(d, cell);
        if s2.propagate(q) {
            search(&s2, q, limit, count, out);
            if *count >= limit {
                return;
            }
        }
    }
}

fn load(q: &mut Queue, clues: &[u8; 81]) {
    for cell in 0..81 {
        let d = clues[cell];
        if d != 0 {
            q.push(d as usize - 1, cell);
        }
    }
}

pub fn solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    let mut st = State::new();
    let mut q = Queue::new();
    load(&mut q, clues);
    if !st.propagate(&mut q) {
        return None;
    }
    let mut out = [0u8; 81];
    let mut count = 0;
    search(&st, &mut q, 1, &mut count, &mut out);
    if count > 0 { Some(out) } else { None }
}

pub fn count_solutions(clues: &[u8; 81], limit: u64) -> u64 {
    let mut st = State::new();
    let mut q = Queue::new();
    load(&mut q, clues);
    if !st.propagate(&mut q) {
        return 0;
    }
    let mut out = [0u8; 81];
    let mut count = 0;
    search(&st, &mut q, limit, &mut count, &mut out);
    count
}
