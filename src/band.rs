//! Band-bitboard solver core.
//!
//! Clean-room implementation of the band-based propagation architecture
//! pioneered by JCZSolve (Zhouyundong et al., New Sudoku Players Forum,
//! 2012-2016) and described at a high level in the tdoku write-up. All code
//! and tables here are derived from the published ideas, not from any
//! existing source.
//!
//! Representation: for each digit d (9) and each horizontal band b (3 rows),
//! a 27-bit mask of positions where d is still possible. Bit i = local row
//! (i/9) * 9 + column (i%9). Invariant: a solved cell keeps exactly its
//! digit's bit set, which makes every unit constraint uniform.
//!
//! Key trick: a band-digit mask "shrinks" to a 9-bit minirow-occupancy matrix
//! (3 rows x 3 boxes). A 512-entry fixpoint table applies both directions of
//! locked candidates (row confined to box / box confined to row) plus
//! row/box-empty contradiction detection in a couple of table lookups.

#[cfg(feature = "stats")]
use crate::GUESSES;

const BAND_ALL: u32 = 0x07FF_FFFF;

/// Cross-digit cardinality inference (see `band_inference`). Measured a net
/// loss -- it shrinks the search tree by ~1.25x on hard puzzles but the scan
/// costs more than the nodes it saves -- so it is off by default.
const CARDINALITY: bool = false;

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// 9-bit row -> 3-bit box occupancy (bit j set if any of cols 3j..3j+3 set).
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

// A digit occupies exactly 3 minirows of a band: one per row, one per box.
// So its occupancy must contain a 3x3 permutation matrix, and the exact
// arc-consistent reduction is the union of the permutations it contains.
// PERMS holds those 6 matrices (bit 3r+j = row r, box j).
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

/// Union of the permutations contained in the occupancy: the exact reduction.
/// 0 = contradiction (no valid placement of this digit in this band).
const SUPPORT: [u16; 512] = build_support();

const fn build_support() -> [u16; 512] {
    let mut t = [0u16; 512];
    let mut v = 0usize;
    while v < 512 {
        let mut s = 0u16;
        let mut k = 0;
        while k < 6 {
            if PERMS[k] & !(v as u16) == 0 {
                s |= PERMS[k];
            }
            k += 1;
        }
        t[v] = s;
        v += 1;
    }
    t
}

/// Intersection of the permutations contained in the occupancy: the minirows
/// this digit must occupy no matter which placement is chosen.
const FORCED: [u16; 512] = build_forced();

const fn build_forced() -> [u16; 512] {
    let mut t = [0u16; 512];
    let mut v = 0usize;
    while v < 512 {
        let mut f = 0x1FFu16;
        let mut any = false;
        let mut k = 0;
        while k < 6 {
            if PERMS[k] & !(v as u16) == 0 {
                f &= PERMS[k];
                any = true;
            }
            k += 1;
        }
        t[v] = if any { f } else { 0 };
        v += 1;
    }
    t
}

/// Row and box companion masks of each minirow, for forcing a digit into it.
const MINIROW_ROW: [u16; 9] = build_minirow_row();
const MINIROW_BOX: [u16; 9] = build_minirow_box();

const fn build_minirow_row() -> [u16; 9] {
    let mut t = [0u16; 9];
    let mut i = 0;
    while i < 9 {
        t[i] = 0b111 << (3 * (i / 3));
        i += 1;
    }
    t
}

const fn build_minirow_box() -> [u16; 9] {
    let mut t = [0u16; 9];
    let mut i = 0;
    while i < 9 {
        t[i] = 0o111 << (i % 3);
        i += 1;
    }
    t
}

/// Minirow-occupancy fixpoint: applies "row confined to one box" and "box
/// confined to one row" reductions until stable. 0 = contradiction (some row
/// or box of the band has no place for the digit). Retained to verify that
/// SUPPORT (the exact reduction) agrees with it.
#[cfg(test)]
const COMPLEX: [u16; 512] = build_complex();

#[cfg(test)]
const fn complex_entry(s0: u16) -> u16 {
    let mut s = s0;
    loop {
        let mut ns = s;
        let mut r = 0;
        while r < 3 {
            let row = (ns >> (3 * r)) & 7;
            if row == 0 {
                return 0;
            }
            if row & (row - 1) == 0 {
                // row r's digit must be in box j: strip box j's other minirows
                let j = row.trailing_zeros() as u16;
                let colbits = 0o111u16 << j;
                ns = (ns & !colbits) | (1 << (3 * r + j));
            }
            r += 1;
        }
        let mut j = 0;
        while j < 3 {
            let col = ns & (0o111 << j);
            if col == 0 {
                return 0;
            }
            if col & (col - 1) == 0 {
                // box j's digit must be in row r: strip row r's other boxes
                let m = col.trailing_zeros() as u16;
                let rowbits = 7u16 << (3 * (m / 3));
                ns = (ns & !rowbits) | (1 << m);
            }
            j += 1;
        }
        if ns == s {
            return s;
        }
        s = ns;
    }
}

#[cfg(test)]
const fn build_complex() -> [u16; 512] {
    let mut t = [0u16; 512];
    let mut v = 0;
    while v < 512 {
        t[v] = complex_entry(v as u16);
        v += 1;
    }
    t
}

/// 9-bit minirow set -> 27-bit cell mask (0b111 per surviving minirow).
const EXPAND: [u32; 512] = build_expand();

const fn build_expand() -> [u32; 512] {
    let mut t = [0u32; 512];
    let mut v = 0;
    while v < 512 {
        let mut mask = 0u32;
        let mut m = 0;
        while m < 9 {
            if v & (1 << m) != 0 {
                let (r, j) = (m / 3, m % 3);
                mask |= 0b111 << (r * 9 + 3 * j);
            }
            m += 1;
        }
        t[v] = mask;
        v += 1;
    }
    t
}

/// Post-assignment mask for the assigned digit's own band: clears the cell's
/// row, box, and in-band column, but keeps the assigned bit itself.
const ELIM: [u32; 27] = build_elim();

const fn build_elim() -> [u32; 27] {
    let mut t = [0u32; 27];
    let mut i = 0;
    while i < 27 {
        let r = i / 9;
        let c = i % 9;
        let j = c / 3;
        let rowm = 0x1FFu32 << (9 * r);
        let seg = 0b111u32 << (3 * j);
        let boxm = seg | seg << 9 | seg << 18;
        let colm = (1u32 << c) | (1 << (c + 9)) | (1 << (c + 18));
        t[i] = (!(rowm | boxm | colm) | (1 << i)) & BAND_ALL;
        i += 1;
    }
    t
}

/// SPREAD[dm] spreads a 9-bit digit mask to bits 3e (for marking
/// digit*3+band dirty bits of one band in a single OR).
const SPREAD: [u32; 512] = build_spread();

const fn build_spread() -> [u32; 512] {
    let mut t = [0u32; 512];
    let mut v = 0;
    while v < 512 {
        let mut m = 0u32;
        let mut e = 0;
        while e < 9 {
            if v & (1 << e) != 0 {
                m |= 1 << (3 * e);
            }
            e += 1;
        }
        t[v] = m;
        v += 1;
    }
    t
}

/// 3-bit column mask within a band (rows 0..3 at column c).
const COL3: [u32; 9] = build_col3();

const fn build_col3() -> [u32; 9] {
    let mut t = [0u32; 9];
    let mut c = 0;
    while c < 9 {
        t[c] = (1u32 << c) | (1 << (c + 9)) | (1 << (c + 18));
        c += 1;
    }
    t
}

// The transposed ("stack") representation reuses the same tables: a stack s
// covers grid columns 3s..3s+3; its local layout is bit = (c%3)*9 + global
// row. A "T row" is then a grid column, a "T box" is a grid box, and the
// COMPLEX fixpoint applied to a stack delivers the column-direction locked
// candidates the horizontal representation cannot see.

/// 3x3 bit-matrix transpose: bit r*3+c -> bit c*3+r.
const T9: [u16; 512] = build_t9();

const fn build_t9() -> [u16; 512] {
    let mut t = [0u16; 512];
    let mut v = 0;
    while v < 512 {
        let mut z = 0u16;
        let mut r = 0;
        while r < 3 {
            let mut c = 0;
            while c < 3 {
                if v & (1 << (r * 3 + c)) != 0 {
                    z |= 1 << (c * 3 + r);
                }
                c += 1;
            }
            r += 1;
        }
        t[v] = z;
        v += 1;
    }
    t
}

/// T2H[s][it] = b << 5 | i: map a stack bit back to its horizontal band bit.
const T2H: [[u8; 27]; 3] = build_t2h();

const fn build_t2h() -> [[u8; 27]; 3] {
    let mut t = [[0u8; 27]; 3];
    let mut s = 0;
    while s < 3 {
        let mut it = 0;
        while it < 27 {
            let (cl, rr) = (it / 9, it % 9);
            let c = s * 3 + cl;
            let (b, r) = (rr / 3, rr % 3);
            let i = r * 9 + c;
            t[s][it] = ((b << 5) | i) as u8;
            it += 1;
        }
        s += 1;
    }
    t
}

/// Clear `bit` from all nine digit masks of one band, returning a 9-bit mask
/// of the digits that held it. This is the hottest step in the solver: it
/// runs on every assignment, and the band-major layout makes the nine masks
/// one contiguous 36-byte block. With AVX2 the first eight digits become a
/// single load/compare/andnot/store and the ninth is handled scalar.
#[inline(always)]
fn clear_cell_in_band(band: &mut [u32; 9], bit: u32) -> u16 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let p = band.as_mut_ptr();
        let v = _mm256_loadu_si256(p as *const __m256i);
        let bv = _mm256_set1_epi32(bit as i32);
        let has = _mm256_cmpeq_epi32(_mm256_and_si256(v, bv), bv);
        let dm = _mm256_movemask_ps(_mm256_castsi256_ps(has)) as u16;
        _mm256_storeu_si256(p as *mut __m256i, _mm256_andnot_si256(bv, v));
        let d8 = *p.add(8);
        *p.add(8) = d8 & !bit;
        dm | ((((d8 & bit) != 0) as u16) << 8)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut dm = 0u16;
        for e in 0..9 {
            dm |= ((band[e] & bit != 0) as u16) << e;
            band[e] &= !bit;
        }
        dm
    }
}

// ---------------------------------------------------------------------------
// Queue: pending assignments, encoded (digit*3 + band) << 5 | bit.
// ---------------------------------------------------------------------------

struct Queue {
    buf: [u16; 512],
    len: usize,
}

impl Queue {
    #[inline]
    fn new() -> Self {
        Queue { buf: [0; 512], len: 0 }
    }
    #[inline]
    fn push(&mut self, e: u16) {
        self.buf[self.len] = e;
        self.len += 1;
    }
    #[inline]
    fn pop(&mut self) -> Option<u16> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(self.buf[self.len])
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

#[derive(Clone)]
struct BState {
    /// bands[b][d]: 27-bit candidate positions of digit d in horizontal band
    /// b. Band-major layout keeps the 9 digits of a band contiguous so the
    /// per-cell digit loops in assign/scan_naked/pick vectorize.
    bands: [[u32; 9]; 3],
    /// Unsolved cells per band (horizontal layout).
    unsolved: [u32; 3],
    /// Per digit: 9-bit mask over (band*3 + row) of rows where d is placed.
    solved_rows: [u16; 9],
    /// Per digit: 9-bit mask of columns where d is placed.
    solved_cols: [u16; 9],
    /// Dirty (digit*3 + band) needing a horizontal band scan.
    dirty_bd: u32,
    /// Digits whose candidates changed since their last transposed pass.
    tdirty: u16,
    /// Dirty bands needing a naked-single scan.
    dirty_cells: u8,
    n_unsolved: u8,
}

impl BState {
    fn new() -> Self {
        BState {
            bands: [[BAND_ALL; 9]; 3],
            unsolved: [BAND_ALL; 3],
            solved_rows: [0; 9],
            solved_cols: [0; 9],
            dirty_bd: (1 << 27) - 1,
            tdirty: 0x1FF,
            dirty_cells: 0b111,
            n_unsolved: 81,
        }
    }

    /// Place digit d at band b, bit i. Pure eliminations; inference happens
    /// in the scans. Returns false on contradiction.
    #[inline]
    fn assign(&mut self, d: usize, b: usize, i: usize) -> bool {
        let bit = 1u32 << i;
        // SAFETY: d < 9, b < 3, i < 27 by construction of queue entries.
        unsafe {
            let band = self.bands.get_unchecked_mut(b);
            let od = *band.get_unchecked(d);
            if od & bit == 0 {
                return false; // the candidate was eliminated: contradiction
            }
            if self.unsolved.get_unchecked(b) & bit == 0 {
                return true; // already solved as d (bit still set)
            }
            let c = i % 9;
            let r = i / 9;
            // Clear the cell from every digit of the band, and report which
            // digits held it. Band-major layout keeps those nine masks
            // contiguous, so one vector pass replaces two scalar loops.
            let dm = clear_cell_in_band(band, bit);
            *band.get_unchecked_mut(d) = od & ELIM.get_unchecked(i);
            let mut bd_mark = *SPREAD.get_unchecked(dm as usize) << b;
            let col3 = *COL3.get_unchecked(c);
            for b2 in 0..3 {
                if b2 != b {
                    let o = *self.bands.get_unchecked(b2).get_unchecked(d);
                    let n = o & !col3;
                    if n != o {
                        *self.bands.get_unchecked_mut(b2).get_unchecked_mut(d) = n;
                        bd_mark |= 1 << (d * 3 + b2);
                        self.dirty_cells |= 1 << b2;
                    }
                }
            }
            self.dirty_bd |= bd_mark;
            self.tdirty |= dm;
            self.dirty_cells |= 1 << b;
            *self.unsolved.get_unchecked_mut(b) &= !bit;
            self.n_unsolved -= 1;
            *self.solved_rows.get_unchecked_mut(d) |= 1 << (b * 3 + r);
            *self.solved_cols.get_unchecked_mut(d) |= 1 << c;
        }
        true
    }

    /// Horizontal band scan: locked-candidates fixpoint via tables + row
    /// hidden singles. Prunes are mirrored into the transposed representation.
    #[inline]
    fn scan_band(&mut self, d: usize, b: usize, q: &mut Queue) -> bool {
        // SAFETY: d < 9, b < 3; table indices are 9-bit values.
        unsafe {
            let mut a = *self.bands.get_unchecked(b).get_unchecked(d);
            let (a0, a1, a2) = (a & 511, (a >> 9) & 511, a >> 18);
            let s = *SHRINK.get_unchecked(a0 as usize)
                | *SHRINK.get_unchecked(a1 as usize) << 3
                | *SHRINK.get_unchecked(a2 as usize) << 6;
            let s2 = *SUPPORT.get_unchecked(s as usize);
            if s2 == 0 {
                return false;
            }
            if s2 != s {
                a &= EXPAND.get_unchecked(s2 as usize);
                *self.bands.get_unchecked_mut(b).get_unchecked_mut(d) = a;
                self.dirty_cells |= 1 << b;
                self.tdirty |= 1 << d;
            }
            // Hidden singles in the band's three rows.
            let solved = *self.solved_rows.get_unchecked(d);
            for r in 0..3 {
                if solved & (1 << (b * 3 + r)) == 0 {
                    let row = (a >> (9 * r)) & 511;
                    if row & (row - 1) == 0 {
                        // row==0 was caught by COMPLEX; this is a single
                        let i = 9 * r + row.trailing_zeros() as usize;
                        q.push(((d * 3 + b) << 5 | i) as u16);
                    }
                }
            }
        }
        true
    }

    /// Gather stack `st` of digit `d` into transposed band layout, via three
    /// 3x3 bit-transposes.
    #[inline]
    fn gather_stack(&self, d: usize, st: usize) -> u32 {
        let mut t = 0u32;
        // SAFETY: d < 9, st < 3; T9 is indexed by a 9-bit value.
        unsafe {
            for b in 0..3 {
                let a = *self.bands.get_unchecked(b).get_unchecked(d) >> (3 * st);
                let y = (a & 7) | ((a >> 9) & 7) << 3 | ((a >> 18) & 7) << 6;
                let z = *T9.get_unchecked(y as usize) as u32;
                t |= ((z & 7) | ((z >> 3) & 7) << 9 | ((z >> 6) & 7) << 18) << (b * 3);
            }
        }
        t
    }

    /// Stall-time inference over all 6 bands (3 horizontal, 3 vertical).
    ///
    /// Per band this applies two layers of reasoning to the 9x9 incidence
    /// matrix of digits against the band's 9 minirows:
    ///
    /// - Per digit (matrix rows): the digit occupies exactly 3 minirows, one
    ///   per row and one per box, so its occupancy must contain a 3x3
    ///   permutation matrix. SUPPORT is the union of the permutations it
    ///   contains -- the exact reduction, subsuming locked candidates.
    /// - Per minirow (matrix columns): a minirow is 3 cells, so it holds
    ///   exactly 3 distinct digits. If only 3 digits can occupy it they are
    ///   all forced into it; if 3 digits are already forced into it, no other
    ///   digit may use it. This cross-digit cardinality reasoning is what a
    ///   per-digit solver cannot see, and it is where most of the search-tree
    ///   reduction comes from.
    ///
    /// Returns None on contradiction, Some(changed) otherwise.
    fn band_inference(&mut self, q: &mut Queue, digits: u16) -> Option<bool> {
        let mut changed = false;
        // Horizontal bands are already reduced eagerly by `scan_band`; when
        // the cardinality rules are off there is nothing left to learn from
        // revisiting them here, so only the stacks need a pass.
        let first = if CARDINALITY { 0 } else { 3 };
        // The cardinality rules are cross-digit and need every digit's
        // occupancy; the plain support reduction is per-digit, so with the
        // rules off only digits that actually changed need revisiting.
        for bi in first..6 {
            let mut masks = [0u32; 9];
            let mut occ = [0u16; 9];
            // SAFETY: all table indices below are 9-bit values or < 9.
            unsafe {
                for d in 0..9 {
                    if !CARDINALITY && digits >> d & 1 == 0 {
                        continue;
                    }
                    let m = if bi < 3 {
                        *self.bands.get_unchecked(bi).get_unchecked(d)
                    } else {
                        self.gather_stack(d, bi - 3)
                    };
                    masks[d] = m;
                    let s = *SHRINK.get_unchecked((m & 511) as usize)
                        | *SHRINK.get_unchecked(((m >> 9) & 511) as usize) << 3
                        | *SHRINK.get_unchecked((m >> 18) as usize) << 6;
                    let sup = *SUPPORT.get_unchecked(s as usize);
                    if sup == 0 {
                        return None;
                    }
                    occ[d] = sup;
                }

                // Column cardinality: how many digits can use each minirow.
                if CARDINALITY {
                let (mut one, mut two, mut three, mut four) = (0u16, 0, 0, 0);
                for d in 0..9 {
                    let s = *occ.get_unchecked(d);
                    four |= three & s;
                    three |= two & s;
                    two |= one & s;
                    one |= s;
                }
                // Every minirow needs 3 distinct digits.
                if three != 0x1FF {
                    return None;
                }
                let mut exact = three & !four;
                while exact != 0 {
                    let t = exact.trailing_zeros() as usize;
                    exact &= exact - 1;
                    let tb = 1u16 << t;
                    // Exactly 3 digits can go here and 3 must, so each of them
                    // is pinned to this minirow: its remaining placements
                    // cannot reuse this minirow's row or box.
                    let keep = !(*MINIROW_ROW.get_unchecked(t) | *MINIROW_BOX.get_unchecked(t)) | tb;
                    for d in 0..9 {
                        let o = *occ.get_unchecked(d);
                        if o & tb != 0 && o & !keep != 0 {
                            let no = *SUPPORT.get_unchecked((o & keep) as usize);
                            if no == 0 {
                                return None;
                            }
                            *occ.get_unchecked_mut(d) = no;
                        }
                    }
                }

                // Minirows already claimed by 3 forced digits are full.
                let (mut f1, mut f2, mut f3) = (0u16, 0, 0);
                let mut forced = [0u16; 9];
                for d in 0..9 {
                    let f = *FORCED.get_unchecked(*occ.get_unchecked(d) as usize);
                    *forced.get_unchecked_mut(d) = f;
                    f3 |= f2 & f;
                    f2 |= f1 & f;
                    f1 |= f;
                }
                if f3 != 0 {
                    for d in 0..9 {
                        let o = *occ.get_unchecked(d);
                        let drop = f3 & !*forced.get_unchecked(d) & o;
                        if drop != 0 {
                            let no = *SUPPORT.get_unchecked((o & !drop) as usize);
                            if no == 0 {
                                return None;
                            }
                            *occ.get_unchecked_mut(d) = no;
                        }
                    }
                }
                } // CARDINALITY

                // Write reductions back to cells and queue hidden singles.
                for d in 0..9 {
                    if !CARDINALITY && digits >> d & 1 == 0 {
                        continue;
                    }
                    let m = *masks.get_unchecked(d);
                    let nm = m & *EXPAND.get_unchecked(*occ.get_unchecked(d) as usize);
                    if nm != m {
                        changed = true;
                        // This digit's cells moved, so its stacks are worth
                        // revisiting -- but only this digit's.
                        self.tdirty |= 1 << d;
                        if bi < 3 {
                            *self.bands.get_unchecked_mut(bi).get_unchecked_mut(d) = nm;
                            self.dirty_bd |= 1 << (d * 3 + bi);
                            self.dirty_cells |= 1 << bi;
                        } else {
                            let st = bi - 3;
                            let mut gone = m & !nm;
                            while gone != 0 {
                                let it = gone.trailing_zeros() as usize;
                                gone &= gone - 1;
                                let t2h = *T2H.get_unchecked(st).get_unchecked(it) as usize;
                                let (b, i) = (t2h >> 5, t2h & 31);
                                *self.bands.get_unchecked_mut(b).get_unchecked_mut(d) &=
                                    !(1u32 << i);
                                self.dirty_bd |= 1 << (d * 3 + b);
                                self.dirty_cells |= 1 << b;
                            }
                        }
                    }
                    // Hidden singles along the band's three lines. For a
                    // horizontal band these are grid rows; for a stack, the
                    // transposed layout makes them grid columns.
                    let solved = if bi < 3 {
                        *self.solved_rows.get_unchecked(d) >> (bi * 3)
                    } else {
                        *self.solved_cols.get_unchecked(d) >> ((bi - 3) * 3)
                    };
                    for r in 0..3 {
                        if solved & (1 << r) == 0 {
                            let line = (nm >> (9 * r)) & 511;
                            if line & (line - 1) == 0 {
                                let idx = 9 * r + line.trailing_zeros() as usize;
                                let (b, i) = if bi < 3 {
                                    (bi, idx)
                                } else {
                                    let t2h = *T2H.get_unchecked(bi - 3).get_unchecked(idx) as usize;
                                    (t2h >> 5, t2h & 31)
                                };
                                q.push(((d * 3 + b) << 5 | i) as u16);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        Some(changed)
    }

    /// Naked singles in band b via cross-digit accumulators; also detects
    /// cells with no remaining candidates.
    #[inline]
    fn scan_naked(&mut self, b: usize, q: &mut Queue) -> bool {
        let mut one = 0u32;
        let mut two = 0u32;
        for d in 0..9 {
            let m = self.bands[b][d];
            two |= one & m;
            one |= m;
        }
        let uns = self.unsolved[b];
        if uns & !one != 0 {
            return false; // empty cell
        }
        let mut naked = one & !two & uns;
        while naked != 0 {
            let i = naked.trailing_zeros() as usize;
            naked &= naked - 1;
            for d in 0..9 {
                if self.bands[b][d] >> i & 1 != 0 {
                    q.push(((d * 3 + b) << 5 | i) as u16);
                    break;
                }
            }
        }
        true
    }

    /// Drain assignments and dirty scans to fixpoint.
    fn propagate(&mut self, q: &mut Queue) -> bool {
        loop {
            while let Some(e) = q.pop() {
                let i = (e & 31) as usize;
                let db = (e >> 5) as usize;
                if !self.assign(db / 3, db % 3, i) {
                    q.clear();
                    return false;
                }
            }
            if self.dirty_bd != 0 {
                let db = self.dirty_bd.trailing_zeros() as usize;
                self.dirty_bd &= self.dirty_bd - 1;
                if !self.scan_band(db / 3, db % 3, q) {
                    q.clear();
                    return false;
                }
                continue;
            }
            if self.dirty_cells != 0 {
                let b = self.dirty_cells.trailing_zeros() as usize;
                self.dirty_cells &= self.dirty_cells - 1;
                if !self.scan_naked(b, q) {
                    q.clear();
                    return false;
                }
                continue;
            }
            // Full stall: run the all-band inference (column direction plus
            // the cross-digit cardinality rules).
            if self.n_unsolved == 0 || self.tdirty == 0 {
                return true;
            }
            let digits = self.tdirty;
            self.tdirty = 0;
            match self.band_inference(q, digits) {
                None => {
                    q.clear();
                    return false;
                }
                // Re-propagate, then revisit whatever is still marked dirty;
                // candidate sets shrink monotonically, so this terminates.
                Some(true) => {}
                Some(false) => return true,
            }
        }
    }

    /// Pick a guess point: first bivalue cell, else a min-candidate cell.
    /// Returns (band, bit, digits, ndigits).
    fn pick(&self) -> (usize, usize, [u8; 9], usize) {
        for b in 0..3 {
            let mut one = 0u32;
            let mut two = 0u32;
            let mut three = 0u32;
            for d in 0..9 {
                let m = self.bands[b][d];
                three |= two & m;
                two |= one & m;
                one |= m;
            }
            let biv = two & !three & self.unsolved[b];
            if biv != 0 {
                let i = biv.trailing_zeros() as usize;
                return self.cell_digits(b, i);
            }
        }
        // Fallback: fewest candidates among unsolved cells (all are 3+).
        let mut best = (0usize, 0usize, [0u8; 9], 10usize);
        for b in 0..3 {
            let mut uns = self.unsolved[b];
            while uns != 0 {
                let i = uns.trailing_zeros() as usize;
                uns &= uns - 1;
                let g = self.cell_digits(b, i);
                if g.3 < best.3 {
                    best = g;
                    if best.3 == 3 {
                        return best;
                    }
                }
            }
        }
        best
    }

    #[inline]
    fn cell_digits(&self, b: usize, i: usize) -> (usize, usize, [u8; 9], usize) {
        let mut digits = [0u8; 9];
        let mut n = 0;
        for d in 0..9 {
            if self.bands[b][d] >> i & 1 != 0 {
                digits[n] = d as u8;
                n += 1;
            }
        }
        (b, i, digits, n)
    }

    fn write_grid(&self, out: &mut [u8; 81]) {
        for b in 0..3 {
            for i in 0..27 {
                for d in 0..9 {
                    if self.bands[b][d] >> i & 1 != 0 {
                        out[b * 27 + i] = d as u8 + 1;
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
pub fn check_support_tables() {
    for v in 0..512usize {
        assert_eq!(
            SUPPORT[v], COMPLEX[v],
            "occupancy {v:09b}: support {:09b} != iterative {:09b}",
            SUPPORT[v], COMPLEX[v]
        );
        // Forced minirows must be a subset of supported ones.
        assert_eq!(FORCED[v] & !SUPPORT[v], 0, "occupancy {v:09b}");
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn search(st: &BState, q: &mut Queue, limit: u64, count: &mut u64, out: &mut [u8; 81]) {
    if st.n_unsolved == 0 {
        if *count == 0 {
            st.write_grid(out);
        }
        *count += 1;
        return;
    }
    let (b, i, digits, n) = st.pick();
    for &d in &digits[..n] {
        #[cfg(feature = "stats")]
        GUESSES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let mut s2 = st.clone();
        q.push(((d as usize * 3 + b) << 5 | i) as u16);
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
            let (b, i) = (cell / 27, cell % 27);
            q.push((((d as usize - 1) * 3 + b) << 5 | i) as u16);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API (used via lib.rs wrappers)
// ---------------------------------------------------------------------------

pub fn solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    let mut st = BState::new();
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
    let mut st = BState::new();
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
