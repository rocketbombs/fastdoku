//! Scalar engine in the JCZSolve family: the state is one 27-bit mask per
//! (digit, band) — "where in this band can this digit still go" — and the
//! propagation workhorse is band-level locked candidates driven by small
//! lookup tables, with naked singles swept between rounds.
//!
//! The architecture is due to zhouyundong_2012 (JCZSolve), with the 128-bit
//! and 32-bit refinements by champagne and JasonLion, as published on the
//! enjoysudoku forum. This file implements that architecture from its
//! published description; it is not a port of any existing implementation,
//! and its tables are derived from first principles below (and verified
//! exhaustively in the tests at the bottom).
//!
//! One deliberate strengthening over the canonical tables: the closure of a
//! band pattern is computed *exactly*. A digit's placements in a band form a
//! 3x3 minirow matrix, and a full solution restricted to that band is a
//! permutation matrix (one minirow per row, one per box). So the exact
//! closure of a pattern is the union of the permutation matrices it
//! contains, and the exactly-forced minirows are their intersection — six
//! subset tests per table entry at build time, the same single lookup at
//! run time. Canonical JCZSolve approximates this with a pointing/claiming
//! fixpoint, which is sound but weaker.
//!
//! Where this engine wins is minimal work per deduction: an update touches
//! one u32 and a couple of table lookups, so puzzles whose solving path is
//! a stream of easy deductions (most of them) never pay for heavier
//! machinery. The mid-hard range belongs to the triad engine; `run`'s
//! difficulty gate and guess budget exist so a caller (the CLI's `auto`
//! engine) can route each puzzle to the right one.

/// All 27 cells of a band.
const ALL: u32 = 0o777_777_777;

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// The six 3x3 permutation matrices as 9-bit minirow masks
/// (bit 3*row + box).
const PERMS: [u16; 6] = build_perms();

const fn build_perms() -> [u16; 6] {
    let p = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let mut out = [0u16; 6];
    let mut k = 0;
    while k < 6 {
        let mut r = 0;
        while r < 3 {
            out[k] |= 1 << (3 * r + p[k][r]);
            r += 1;
        }
        k += 1;
    }
    out
}

/// FORCED[s]: intersection of the permutation matrices contained in `s` —
/// minirows that *must* hold the digit in every completion.
static FORCED: [u16; 512] = build_forced();

const fn build_closed() -> [u16; 512] {
    let perms = PERMS;
    let mut t = [0u16; 512];
    let mut s = 0usize;
    while s < 512 {
        let mut k = 0;
        while k < 6 {
            if perms[k] & s as u16 == perms[k] {
                t[s] |= perms[k];
            }
            k += 1;
        }
        s += 1;
    }
    t
}

const fn build_forced() -> [u16; 512] {
    let perms = PERMS;
    let mut t = [0u16; 512];
    let mut s = 0usize;
    while s < 512 {
        let mut acc = 0x1ffu16;
        let mut any = false;
        let mut k = 0;
        while k < 6 {
            if perms[k] & s as u16 == perms[k] {
                acc &= perms[k];
                any = true;
            }
            k += 1;
        }
        t[s] = if any { acc } else { 0 };
        s += 1;
    }
    t
}

/// CLOSED_CELLS[s]: the exact closure of minirow pattern `s` — the union of
/// the permutation matrices it contains — expanded to a 27-bit cell mask, so
/// the hot path chains one table load instead of two. Zero iff no
/// permutation fits, i.e. the band is unsatisfiable for the digit.
static CLOSED_CELLS: [u32; 512] = build_closed_cells();

const fn build_closed_cells() -> [u32; 512] {
    let closed = build_closed();
    let expand = build_expand();
    let mut t = [0u32; 512];
    let mut sp = 0usize;
    while sp < 512 {
        t[sp] = expand[closed[sp] as usize];
        sp += 1;
    }
    t
}

/// EXPAND[m]: 9-bit minirow mask -> the 27-bit cell mask of those minirows.
static EXPAND: [u32; 512] = build_expand();

const fn build_expand() -> [u32; 512] {
    let mut t = [0u32; 512];
    let mut m = 0usize;
    while m < 512 {
        let mut i = 0;
        while i < 9 {
            if m & (1 << i) != 0 {
                let (r, b) = (i / 3, i % 3);
                t[m] |= 0b111 << (9 * r + 3 * b);
            }
            i += 1;
        }
        m += 1;
    }
    t
}

/// COL_SINGLE[cols]: given the 9-bit column-occupancy of the band, the
/// minirow mask of boxes whose candidates sit in a single column — every
/// minirow of such a box holds at most one cell.
static COL_SINGLE: [u16; 512] = build_col_single();

const fn build_col_single() -> [u16; 512] {
    let mut t = [0u16; 512];
    let mut c = 0usize;
    while c < 512 {
        let mut b = 0;
        while b < 3 {
            let cb = (c >> (3 * b)) & 7;
            if cb != 0 && cb & (cb - 1) == 0 {
                t[c] |= 0o111 << b; // bits b, b+3, b+6
            }
            b += 1;
        }
        c += 1;
    }
    t
}

/// NEIGH_OK[cols]: pointing eliminations for the two neighbor bands. A box
/// whose candidates are confined to one column must place the digit in that
/// column, so the neighbors lose that whole column.
static NEIGH_OK: [u32; 512] = build_neigh_ok();

const fn build_neigh_ok() -> [u32; 512] {
    let mut t = [0u32; 512];
    let mut c = 0usize;
    while c < 512 {
        let mut cleared = 0u32;
        let mut b = 0;
        while b < 3 {
            let cb = ((c >> (3 * b)) & 7) as u32;
            if cb != 0 && cb & (cb - 1) == 0 {
                let j = 3 * b as u32 + cb.trailing_zeros();
                cleared |= 0o001_001_001 << j;
            }
            b += 1;
        }
        t[c] = ALL & !cleared;
        c += 1;
    }
    t
}

/// SAME_BAND_OK[pos]: placing the digit at cell `pos` of a band removes it
/// from the rest of the cell's row and box within the band (columns are the
/// neighbors' business), keeping the cell itself.
static SAME_BAND_OK: [u32; 27] = build_same_band_ok();

const fn build_same_band_ok() -> [u32; 27] {
    let mut t = [0u32; 27];
    let mut pos = 0usize;
    while pos < 27 {
        let (row, boxcol) = (pos / 9, (pos % 9) / 3);
        let row_mask = 0x1ffu32 << (9 * row);
        let box_mask = (0b111u32 << (3 * boxcol)) * 0o001_001_001;
        t[pos] = (ALL & !(row_mask | box_mask)) | (1 << pos);
        pos += 1;
    }
    t
}

/// CELL_UNITS[cell]: the three unit bits a clue in this cell contributes to
/// its digit's `unit_mask` -- its row and its box, as six bits per band
/// (rows low, boxes high), and its column in bits 18..27. One table and one
/// or per clue in place of three of each.
static CELL_UNITS: [u32; 81] = build_cell_units();

const fn build_cell_units() -> [u32; 81] {
    let mut t = [0u32; 81];
    let mut c = 0;
    while c < 81 {
        let (row, col) = (c / 9, c % 9);
        let band = row / 3;
        t[c] = (1 << (6 * band + row % 3)) | (1 << (6 * band + 3 + col / 3)) | (1 << (18 + col));
        c += 1;
    }
    t
}

/// CELL_POS[cell]: the cell's bit within its band.
static CELL_POS: [u32; 81] = build_cell_pos();

const fn build_cell_pos() -> [u32; 81] {
    let mut t = [0u32; 81];
    let mut c = 0;
    while c < 81 {
        t[c] = 1 << (c % 27);
        c += 1;
    }
    t
}

/// CELL_SLOT[cell]: the cell's band, so a clue's `clue_bits` slot is
/// `digit * 3 + CELL_SLOT[cell]`.
static CELL_SLOT: [u8; 81] = build_cell_slot();

const fn build_cell_slot() -> [u8; 81] {
    let mut t = [0u8; 81];
    let mut c = 0;
    while c < 81 {
        t[c] = (c / 27) as u8;
        c += 1;
    }
    t
}

/// ROWBOX[sel]: for one band, the cells covered by the full rows and full
/// boxes named by a six-bit selector (rows in bits 0..3, boxes in 3..6).
/// Carrying rows and boxes in one 256-byte table makes each subband's
/// eliminations a single lookup, and the layout pairs with `CELL_UNITS` so
/// the selector is one shift and one mask.
static ROWBOX: [u32; 64] = build_rowbox();

const fn build_rowbox() -> [u32; 64] {
    let mut t = [0u32; 64];
    let mut i = 0usize;
    while i < 64 {
        let mut m = 0u32;
        let mut r = 0;
        while r < 3 {
            if i & (1 << r) != 0 {
                m |= 0x1ff << (9 * r);
            }
            r += 1;
        }
        let mut b = 0;
        while b < 3 {
            if i & (1 << (3 + b)) != 0 {
                m |= (0b111 << (3 * b)) * 0o001_001_001;
            }
            b += 1;
        }
        t[i] = m;
        i += 1;
    }
    t
}

/// Condense a band's 27 cell bits to its 9-bit minirow occupancy.
///
/// `m | m>>1 | m>>2` puts each minirow's or at the minirow's first cell, so
/// the nine bits wanted are those at positions 0, 3, 6, ... -- exactly a bit
/// extract. On Zen 3 `pext` is 3-cycle hardware, which beats the three
/// dependent table loads this replaces: it takes a load off the critical
/// path of every subband update (the result feeds straight into the
/// `CLOSED_CELLS` lookup) and frees 512 bytes of L1 besides.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
fn shrink_band(m: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::_pext_u32;
    // SAFETY: `_pext_u32` is a pure BMI2 arithmetic intrinsic, gated above.
    unsafe { _pext_u32(m | (m >> 1) | (m >> 2), 0o111_111_111) }
}

/// MINIROW_OCC[x]: for a 9-bit row of a band, which of its three minirows
/// hold anything, as a 3-bit mask.
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
static MINIROW_OCC: [u8; 512] = build_minirow_occ();

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
const fn build_minirow_occ() -> [u8; 512] {
    let mut t = [0u8; 512];
    let mut x = 0usize;
    while x < 512 {
        let mut i = 0;
        while i < 3 {
            if (x >> (3 * i)) & 7 != 0 {
                t[x] |= 1 << i;
            }
            i += 1;
        }
        x += 1;
    }
    t
}

/// Without `pext`, three *independent* byte lookups, one per row of the
/// band, shifted into place.
///
/// The obvious portable form -- nine shift-mask-or steps, or the canonical
/// JCZSolve chain of three dependent table loads -- both put a serial chain
/// where this result cannot afford one: it feeds straight into the
/// `CLOSED_CELLS` lookup on the critical path of every subband update. Here
/// the three loads issue together off three independent extracts, so the
/// whole condense is one shift plus one load plus the combine, and the
/// tables absorb the `m | m>>1 | m>>2` folding as well (an entry is indexed
/// by a raw 9-bit row, not a pre-folded one).
///
/// A table-free alternative exists -- two chained multiplies gather bits at
/// stride 3 without collisions (`* 0x15`, mask, `* 0x1041`) -- and was
/// rejected: seven instructions but two dependent 3-cycle multiplies, which
/// is longer than the load latency it would replace.
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline(always)]
fn shrink_band(m: u32) -> u32 {
    // SAFETY: each index is masked to 9 bits, and `m` is a 27-bit band mask
    // so the third extract is in range too.
    unsafe {
        let r0 = *MINIROW_OCC.get_unchecked((m & 0x1ff) as usize) as u32;
        let r1 = *MINIROW_OCC.get_unchecked(((m >> 9) & 0x1ff) as usize) as u32;
        let r2 = *MINIROW_OCC.get_unchecked(((m >> 18) & 0x1ff) as usize) as u32;
        r0 | (r1 << 3) | (r2 << 6)
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Subband index = band * 9 + digit: one band's nine digits are contiguous,
/// which is the access pattern of both the naked-single scan and the
/// solved-cell sweep.
#[derive(Clone, Copy)]
struct State {
    /// poss[band * 9 + digit]: cells of `band` where `digit` remains possible.
    poss: [u32; 27],
    /// Worklist: subbands whose mask changed since their last update. Every
    /// write to `poss` maintains it, so reaching the fixpoint is `dirty == 0`
    /// with no rescans - and the state copied per guess is 27 words plus
    /// this one, not a second shadow array.
    dirty: u32,
    /// Cells not yet fixed, per band.
    unsolved: [u32; 3],
    /// Cells with exactly two candidate digits, per band (guess targets).
    pairs: [u32; 3],
}

struct Unsat;





/// NEIGH[sb]: the same digit's other two bands, `(sb + 9) % 27` and
/// `(sb + 18) % 27`. A table because the modulo compiles to three `lea`s,
/// two compares and two `cmov`s on the critical path of every update, and
/// the loads issue on ports the update loop has to spare.
static NEIGH: [[u8; 2]; 27] = build_neigh();

const fn build_neigh() -> [[u8; 2]; 27] {
    let mut t = [[0u8; 2]; 27];
    let mut sb = 0;
    while sb < 27 {
        t[sb] = [((sb + 9) % 27) as u8, ((sb + 18) % 27) as u8];
        sb += 1;
    }
    t
}

impl State {
    /// Build the state from the clue grid in one batched pass.
    ///
    /// Per clue this is now two table loads and two read-modify-writes: the
    /// digit's row, box and column bits go into one `unit_mask` word, and the
    /// cell's own bit into one `clue_bits` word. The three separate unit
    /// arrays, the three duplicate tests and the separate `clue_cells`
    /// accumulation it replaces were together about two thirds of the scan.
    ///
    /// Duplicates fall out of a count instead of a test: every clue
    /// contributes exactly three unit bits, so if two clues share a row, box
    /// or column the or loses one and the total falls short.
    fn init_from_clues(clues: &[u8; 81]) -> Result<State, Unsat> {
        // unit_mask[digit]: rows|boxes per band in bits 0..18, columns in 18..27.
        let mut unit_mask = [0u32; 10];
        // clue_bits[digit * 3 + band]: the clue cells of that digit in that band.
        let mut clue_bits = [0u32; 30];

        let visit = |cell: usize| {
            // SAFETY: cell < 81, and clue digits are 1..=9 by construction of
            // the scan below, so both indices are in range.
            unsafe {
                let d = *clues.get_unchecked(cell) as usize;
                *unit_mask.get_unchecked_mut(d) |= *CELL_UNITS.get_unchecked(cell);
                let slot = d * 3 + *CELL_SLOT.get_unchecked(cell) as usize;
                *clue_bits.get_unchecked_mut(slot) |= *CELL_POS.get_unchecked(cell);
            }
        };

        // Clue positions as a bitmask, so the scan touches only clue cells
        // and its control flow does not depend on where they are.
        // SAFETY: every bit set in the masks is a cell index below 81.
        let (mut lo, mut hi) = unsafe { crate::clue_scan::clue_masks(clues) };
        let n_clues = lo.count_ones() + hi.count_ones();
        let mut visit = visit;
        while lo != 0 {
            visit(lo.trailing_zeros() as usize);
            lo &= lo - 1;
        }
        while hi != 0 {
            visit(64 + hi.trailing_zeros() as usize);
            hi &= hi - 1;
        }

        let mut unit_bits = 0u32;
        for d in 1..=9usize {
            unit_bits += unit_mask[d].count_ones();
        }
        if unit_bits != 3 * n_clues {
            return Err(Unsat);
        }

        // Solved cells per band, folded out of the per-digit clue masks.
        let mut clue_cells = [0u32; 3];
        for d in 1..=9usize {
            for b in 0..3 {
                clue_cells[b] |= clue_bits[d * 3 + b];
            }
        }

        // Every subband is written here, so the array starts uninitialised
        // rather than being filled with ALL and immediately overwritten.
        let mut poss = [0u32; 27];
        for d in 1..=9usize {
            // SAFETY: d <= 9 and b < 3 keep every index inside its table.
            unsafe {
                let um = *unit_mask.get_unchecked(d);
                let cs = um >> 18;
                let colspread = cs | (cs << 9) | (cs << 18);
                for b in 0..3usize {
                    let sel = ((um >> (6 * b)) & 63) as usize;
                    let elim = *ROWBOX.get_unchecked(sel) | colspread;
                    *poss.get_unchecked_mut(b * 9 + d - 1) =
                        (ALL & !elim & !clue_cells[b]) | *clue_bits.get_unchecked(d * 3 + b);
                }
            }
        }

        Ok(State {
            poss,
            dirty: (1 << 27) - 1,
            unsolved: [
                ALL & !clue_cells[0],
                ALL & !clue_cells[1],
                ALL & !clue_cells[2],
            ],
            pairs: [0; 3],
        })
    }

    /// Locked candidates, pointing eliminations, and solved-cell cleanup for
    /// one subband. Returns the subbands this update changed, for the
    /// caller's worklist, or Err on an unsatisfiable band.
    ///
    /// The dirty bits are returned rather than OR'd into `self.dirty`
    /// because that field is otherwise a load-modify-store per write, three
    /// times per update, on the critical path; the caller keeps the
    /// worklist in a register instead.
    ///
    /// SAFETY: callers pass `sb < 27` (worklist bits are only ever set for
    /// indices 0..27). Table indices are 9-bit by construction: `s` and
    /// `cols` are masked, and FORCED/COL_SINGLE entries only carry bits
    /// 0..9. The unchecked accesses eliminate bounds checks and their panic
    /// branches from the innermost loop.
    #[inline(always)]
    unsafe fn update_subband(&mut self, sb: usize) -> Result<u32, Unsat> {
        let m = *self.poss.get_unchecked(sb);
        let s = shrink_band(m) as usize;
        let allowed = *CLOSED_CELLS.get_unchecked(s);
        if allowed == 0 {
            return Err(Unsat);
        }
        let m = m & allowed;

        // Column occupancy drives both the pointing eliminations to the
        // neighbor bands and the solved-cell detection below.
        let cols = ((m | (m >> 9) | (m >> 18)) & 0x1ff) as usize;
        let ok = *NEIGH_OK.get_unchecked(cols);
        let nb = NEIGH.get_unchecked(sb);
        let (n1, n2) = (nb[0] as usize, nb[1] as usize);
        let o1 = *self.poss.get_unchecked(n1);
        let o2 = *self.poss.get_unchecked(n2);
        let w1 = o1 & ok;
        let w2 = o2 & ok;
        *self.poss.get_unchecked_mut(n1) = w1;
        *self.poss.get_unchecked_mut(n2) = w2;
        let mut dirty = (((w1 != o1) as u32) << n1) | (((w2 != o2) as u32) << n2);

        // A minirow forced to hold the digit, in a box confined to a single
        // column, is a solved cell: clear the cell from the band's other
        // digits and from the unsolved set. The branch was also tried
        // unconditional (JCZSolve lore says it mispredicts badly), but on
        // this machine the guarded form measured faster on every corpus —
        // the exact closure tables make the no-solve case the strongly
        // biased common one. The sweep covers the subband's own digit too —
        // harmlessly, since `m` keeps its solved cells — so its store lands
        // after the loop. The dirty marking must be exact: forced minirows
        // stay forced, so blanket re-dirtying here never converges.
        let solved_mr = *FORCED.get_unchecked(s) & *COL_SINGLE.get_unchecked(cols);
        if solved_mr == 0 {
            *self.poss.get_unchecked_mut(sb) = m;
            return Ok(dirty);
        }
        let solved = *EXPAND.get_unchecked(solved_mr as usize) & m;
        let keep = !solved;
        let band = sb / 9;
        *self.unsolved.get_unchecked_mut(band) &= keep;
        let base = band * 9;
        dirty |= self.sweep_band(base, keep) & !(1 << sb);
        *self.poss.get_unchecked_mut(sb) = m;
        Ok(dirty)
    }

    /// Clear `!keep` from all nine of a band's subbands, returning which of
    /// them changed. The nine are contiguous, so with AVX2 eight fall out of
    /// one masked store plus a `movemask` — against roughly seven scalar
    /// instructions each, and this runs on every solved cell.
    ///
    /// SAFETY: `base` is 0, 9 or 18, so the 8-lane access covers indices
    /// `base..base+8` of a 27-element array and the scalar tail is
    /// `base + 8 <= 26`.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    unsafe fn sweep_band(&mut self, base: usize, keep: u32) -> u32 {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        let p = self.poss.as_mut_ptr().add(base);
        let old = _mm256_loadu_si256(p as *const __m256i);
        let new = _mm256_and_si256(old, _mm256_set1_epi32(keep as i32));
        _mm256_storeu_si256(p as *mut __m256i, new);
        let same = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(old, new)));
        let mut dirty = ((!same as u32) & 0xff) << base;
        let ov = *p.add(8);
        let nv = ov & keep;
        *p.add(8) = nv;
        dirty |= ((nv != ov) as u32) << (base + 8);
        dirty
    }

    /// NEON has no 32-bit `movemask`, so the eight comparison results are
    /// narrowed to 16-bit lanes, masked against bit weights and summed with
    /// one `addv` -- five operations for the mask against x86's one, but
    /// still against roughly seven scalar instructions per subband.
    ///
    /// SAFETY: as above.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn sweep_band(&mut self, base: usize, keep: u32) -> u32 {
        use core::arch::aarch64::*;
        static WEIGHTS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        let p = self.poss.as_mut_ptr().add(base);
        let k = vdupq_n_u32(keep);
        let (o0, o1) = (vld1q_u32(p), vld1q_u32(p.add(4)));
        let (n0, n1) = (vandq_u32(o0, k), vandq_u32(o1, k));
        vst1q_u32(p, n0);
        vst1q_u32(p.add(4), n1);
        let same = vcombine_u16(vmovn_u32(vceqq_u32(o0, n0)), vmovn_u32(vceqq_u32(o1, n1)));
        let same_bits = vaddvq_u16(vandq_u16(same, vld1q_u16(WEIGHTS.as_ptr()))) as u32;
        let mut dirty = (!same_bits & 0xff) << base;
        let ov = *p.add(8);
        let nv = ov & keep;
        *p.add(8) = nv;
        dirty |= ((nv != ov) as u32) << (base + 8);
        dirty
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        target_arch = "aarch64"
    )))]
    #[inline(always)]
    unsafe fn sweep_band(&mut self, base: usize, keep: u32) -> u32 {
        let mut dirty = 0;
        for d in 0..9 {
            let o = base + d;
            let ov = *self.poss.get_unchecked(o);
            let nv = ov & keep;
            *self.poss.get_unchecked_mut(o) = nv;
            dirty |= ((nv != ov) as u32) << o;
        }
        dirty
    }

    /// Drain the dirty worklist in batches: capture the current mask, process
    /// every set bit, repeat. Processing against a captured batch keeps each
    /// iteration's control flow independent of the dirtiness the updates are
    /// computing (the loop is not serialized on their load chains), and
    /// multiple re-dirtyings of one subband within a round coalesce into a
    /// single update the next round.
    #[inline]
    fn update_all(&mut self) -> Result<(), Unsat> {
        let mut pending = self.dirty;
        loop {
            let mut batch = pending;
            if batch == 0 {
                self.dirty = 0;
                return Ok(());
            }
            pending = 0;
            while batch != 0 {
                let sb = batch.trailing_zeros() as usize;
                batch &= batch - 1;
                // An earlier update this round may have re-dirtied `sb`;
                // this visit sees the fresh state, so drop the pending bit.
                pending &= !(1 << sb);
                // SAFETY: dirty bits are only set for subbands 0..27.
                pending |= unsafe { self.update_subband(sb)? };
            }
        }
    }

    #[inline(always)]
    fn is_solved(&self) -> bool {
        self.unsolved[0] | self.unsolved[1] | self.unsolved[2] == 0
    }

    /// Naked singles: cells with one candidate get it; zero-candidate cells
    /// are a contradiction. Also refreshes `pairs`. Returns whether any
    /// single was placed.
    #[inline]
    fn naked_singles(&mut self) -> Result<bool, Unsat> {
        let mut placed = false;
        for band in 0..3 {
            let base = band * 9;
            let (mut c1, mut c2, mut c3) = (0u32, 0u32, 0u32);
            for d in 0..9 {
                let m = self.poss[base + d];
                c3 |= c2 & m;
                c2 |= c1 & m;
                c1 |= m;
            }
            if c1 != ALL {
                return Err(Unsat);
            }
            self.pairs[band] = c2 ^ c3;
            let mut singles = (c1 ^ c2) & self.unsolved[band];
            if singles == 0 {
                continue;
            }
            // Which digit is the single? The obvious answer -- scan the
            // band's nine subbands for the one holding this cell -- runs
            // 1..9 iterations with the trip count set by the digit, which is
            // as good as random, so it mispredicts on nearly every single
            // placed. Instead the digit's *index* is bit-sliced across four
            // masks: `slice[k]` carries, at each cell, bit k of the index of
            // every digit still possible there, so at a cell where only one
            // digit is possible the four masks spell out exactly that
            // digit's index. Reading it back is four shifts and three ors,
            // branchless.
            //
            // Twelve ors to build, so it is built here rather than in the
            // accumulation above: on the hard corpora most calls place
            // nothing and would never use it.
            //
            // SAFETY: base + 8 <= 26 indexes `poss`.
            let sl = unsafe {
                let p = |d: usize| *self.poss.get_unchecked(base + d);
                [
                    p(1) | p(3) | p(5) | p(7), // digits with bit 0 of the index
                    p(2) | p(3) | p(6) | p(7), // bit 1
                    p(4) | p(5) | p(6) | p(7), // bit 2
                    p(8),                      // bit 3
                ]
            };
            while singles != 0 {
                let bit = singles & singles.wrapping_neg();
                singles &= singles - 1;
                placed = true;
                // `bit` comes from a mask ANDed with ALL, so pos < 27.
                let pos = bit.trailing_zeros() as usize;
                let d = (((sl[0] >> pos) & 1)
                    | ((sl[1] >> pos) & 1) << 1
                    | ((sl[2] >> pos) & 1) << 2
                    | ((sl[3] >> pos) & 1) << 3) as usize;
                // The slices were taken before this loop placed anything, so
                // they can be stale: an earlier placement can have cleared
                // this cell's last candidate, which is a contradiction. The
                // guard covers that, and keeps the index below in bounds
                // whatever the slices said.
                if d >= 9 {
                    return Err(Unsat);
                }
                // SAFETY: base + d <= 26, and pos < 27.
                unsafe {
                    let cur = self.poss.get_unchecked_mut(base + d);
                    if *cur & bit == 0 {
                        return Err(Unsat); // every digit eliminated meanwhile
                    }
                    *cur &= *SAME_BAND_OK.get_unchecked(pos);
                }
                self.dirty |= 1 << (base + d);
            }
        }
        Ok(placed)
    }

    /// Propagate to fixpoint: locked candidates, then naked singles, until
    /// neither moves. On a not-solved return, `pairs` is fresh.
    fn full_update(&mut self) -> Result<(), Unsat> {
        loop {
            self.update_all()?;
            if self.is_solved() {
                return Ok(());
            }
            if !self.naked_singles()? {
                return Ok(());
            }
        }
    }

    /// Write the solved grid. The scalar form is 81 iterations of an
    /// unpredictable bit-scan loop; with AVX2 each subband instead becomes a
    /// 27-lane byte mask (broadcast, shuffle, bit-test) multiplied by its
    /// digit and OR-accumulated — no data-dependent branches at all.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn extract(&self, out: &mut [u8; 81]) {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        // SAFETY: plain AVX2 arithmetic; the 32-byte stores land in a
        // 33-byte scratch buffer and 27 bytes are copied out per band.
        unsafe {
            let spread = _mm256_setr_epi8(
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                3, 3, 3, 3, 3, 3,
            );
            let bits = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
            for band in 0..3 {
                let mut acc = _mm256_setzero_si256();
                for d in 0..9 {
                    let m = _mm256_set1_epi32(self.poss[band * 9 + d] as i32);
                    let bytes = _mm256_shuffle_epi8(m, spread);
                    let hit = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bits), bits);
                    let dig = _mm256_set1_epi8(d as i8 + 1);
                    acc = _mm256_or_si256(acc, _mm256_and_si256(hit, dig));
                }
                let mut buf = [0u8; 32];
                _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
                out[band * 27..band * 27 + 27].copy_from_slice(&buf[..27]);
            }
        }
    }

    /// The same construction on NEON, over two 16-byte registers per band
    /// instead of one 32-byte one. `vqtbl1q_u8` spreads the broadcast band
    /// mask to one byte per cell exactly as `vpshufb` does.
    #[cfg(target_arch = "aarch64")]
    fn extract(&self, out: &mut [u8; 81]) {
        use core::arch::aarch64::*;
        static SPREAD: [u8; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
            3, 3, 3, 3, 3,
        ];
        // SAFETY: plain NEON arithmetic; the two 16-byte stores land in a
        // 32-byte scratch buffer and 27 bytes are copied out per band.
        unsafe {
            let sp0 = vld1q_u8(SPREAD.as_ptr());
            let sp1 = vld1q_u8(SPREAD.as_ptr().add(16));
            let bits = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
            for band in 0..3 {
                let mut a0 = vdupq_n_u8(0);
                let mut a1 = vdupq_n_u8(0);
                for d in 0..9 {
                    let m = vreinterpretq_u8_u32(vdupq_n_u32(self.poss[band * 9 + d]));
                    let dig = vdupq_n_u8(d as u8 + 1);
                    // `vtst` is the whole test: the weights have one bit
                    // per lane, so "the masked bits equal the weight" and
                    // "the masked bits are non-zero" agree, and NEON has the
                    // second as one instruction.
                    let h0 = vtstq_u8(vqtbl1q_u8(m, sp0), bits);
                    let h1 = vtstq_u8(vqtbl1q_u8(m, sp1), bits);
                    a0 = vorrq_u8(a0, vandq_u8(h0, dig));
                    a1 = vorrq_u8(a1, vandq_u8(h1, dig));
                }
                let mut buf = [0u8; 32];
                vst1q_u8(buf.as_mut_ptr(), a0);
                vst1q_u8(buf.as_mut_ptr().add(16), a1);
                out[band * 27..band * 27 + 27].copy_from_slice(&buf[..27]);
            }
        }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        target_arch = "aarch64"
    )))]
    fn extract(&self, out: &mut [u8; 81]) {
        for band in 0..3 {
            for d in 0..9 {
                let mut m = self.poss[band * 9 + d];
                while m != 0 {
                    let pos = m.trailing_zeros() as usize;
                    m &= m - 1;
                    out[band * 27 + pos] = d as u8 + 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

struct Solver {
    limit: u64,
    count: u64,
    /// Snapshot of the first counted solution's masks.
    solution: [u32; 27],
    guesses: u32,
    budget: u32,
    aborted: bool,
}

impl Solver {
    /// Reached a stable state: count it if solved, otherwise branch.
    fn descend(&mut self, state: &mut State) {
        if state.is_solved() {
            if self.count == 0 {
                self.solution = state.poss;
            }
            self.count += 1;
            return;
        }
        if self.guesses >= self.budget {
            self.aborted = true;
            return;
        }
        self.guesses += 1;
        if !self.guess_pair(state) {
            self.guess_cell(state);
        }
    }

    /// Run `state` to fixpoint and descend. Consumes the state.
    fn solve_from(&mut self, state: &mut State) {
        if state.full_update().is_ok() {
            self.descend(state);
        }
    }

    /// Branch on the first bivalue cell if one exists. Returns false when
    /// there is none.
    fn guess_pair(&mut self, state: &mut State) -> bool {
        for band in 0..3 {
            let pairs = state.pairs[band];
            if pairs == 0 {
                continue;
            }
            let bit = pairs & pairs.wrapping_neg();
            let pos = bit.trailing_zeros() as usize;
            let base = band * 9;
            let mut first = true;
            for d in 0..9 {
                if state.poss[base + d] & bit == 0 {
                    continue;
                }
                if first {
                    first = false;
                    let mut copy = *state;
                    copy.poss[base + d] &= SAME_BAND_OK[pos];
                    copy.dirty |= 1 << (base + d);
                    self.solve_from(&mut copy);
                    if self.count >= self.limit || self.aborted {
                        return true;
                    }
                    state.poss[base + d] &= !bit;
                    state.dirty |= 1 << (base + d);
                } else {
                    state.poss[base + d] &= SAME_BAND_OK[pos];
                    state.dirty |= 1 << (base + d);
                    self.solve_from(state);
                    return true;
                }
            }
            // The second candidate vanished while trying the first; the
            // remaining branch is the eliminations already applied.
            self.solve_from(state);
            return true;
        }
        false
    }

    /// No bivalue cell: probe the first unsolved cell of each band and
    /// branch on the one with fewest candidates.
    fn guess_cell(&mut self, state: &mut State) {
        let mut best = usize::MAX;
        let mut best_n = u32::MAX;
        let mut best_bit = 0u32;
        for band in 0..3 {
            let unsolved = state.unsolved[band];
            if unsolved == 0 {
                continue;
            }
            let bit = unsolved & unsolved.wrapping_neg();
            let mut n = 0;
            for d in 0..9 {
                n += (state.poss[band * 9 + d] & bit != 0) as u32;
            }
            if n < best_n {
                best_n = n;
                best = band;
                best_bit = bit;
            }
        }
        if best == usize::MAX {
            return;
        }
        let pos = best_bit.trailing_zeros() as usize;
        let base = best * 9;
        for d in 0..9 {
            if state.poss[base + d] & best_bit == 0 {
                continue;
            }
            let mut copy = *state;
            copy.poss[base + d] &= SAME_BAND_OK[pos];
            copy.dirty |= 1 << (base + d);
            self.solve_from(&mut copy);
            if self.count >= self.limit || self.aborted {
                return;
            }
            state.poss[base + d] &= !best_bit;
            state.dirty |= 1 << (base + d);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Outcome of a bounded run.
pub enum Outcome {
    /// Solutions found (up to the limit), with the first one if any.
    Done(u64, Option<[u8; 81]>),
    /// The difficulty gate or guess budget tripped; caller should hand the
    /// puzzle to a stronger engine.
    Deferred,
}

/// Solve with a difficulty gate and a guess budget.
///
/// `max_unsolved`: after the initial propagation (all of it useful work no
/// matter which engine finishes the job), a puzzle still this far from
/// solved is not this engine's regime — bail out before the first guess.
/// `budget`: safety valve for puzzles that pass the gate but blow up anyway.
pub fn run(clues: &[u8; 81], limit: u64, max_unsolved: u32, budget: u32) -> Outcome {
    let mut state = match State::init_from_clues(clues) {
        Ok(st) => st,
        Err(Unsat) => return Outcome::Done(0, None),
    };
    if state.full_update().is_err() {
        return Outcome::Done(0, None);
    }
    if !state.is_solved() {
        let open = state.unsolved[0].count_ones()
            + state.unsolved[1].count_ones()
            + state.unsolved[2].count_ones();
        if open > max_unsolved {
            return Outcome::Deferred;
        }
    }
    let mut solver = Solver {
        limit,
        count: 0,
        solution: [0; 27],
        guesses: 0,
        budget,
        aborted: false,
    };
    solver.descend(&mut state);
    if solver.aborted {
        return Outcome::Deferred;
    }
    if solver.count == 0 {
        return Outcome::Done(0, None);
    }
    let mut out = [0u8; 81];
    let snap = State { poss: solver.solution, dirty: 0, unsolved: [0; 3], pairs: [0; 3] };
    snap.extract(&mut out);
    Outcome::Done(solver.count, Some(out))
}

/// Solve to the first solution, unbounded.
pub fn solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    match run(clues, 1, u32::MAX, u32::MAX) {
        Outcome::Done(n, sol) if n > 0 => sol,
        _ => None,
    }
}

/// Count solutions up to `limit`, unbounded.
pub fn count_solutions(clues: &[u8; 81], limit: u64) -> u64 {
    match run(clues, limit, u32::MAX, u32::MAX) {
        Outcome::Done(n, _) => n,
        Outcome::Deferred => unreachable!("unbounded run cannot defer"),
    }
}

// ---------------------------------------------------------------------------
// Table verification
// ---------------------------------------------------------------------------

#[cfg(test)]
mod table_tests {
    use super::*;

    /// Brute-force truth for every 512-entry table against the definition
    /// "a full solution restricted to one band and digit is a permutation
    /// matrix over minirows".
    #[test]
    fn closure_tables_are_exact() {
        for s in 0..512u16 {
            let fits: Vec<u16> = PERMS.iter().copied().filter(|&p| p & s == p).collect();
            let union = fits.iter().fold(0, |a, &p| a | p);
            let inter = fits.iter().fold(0x1ff, |a, &p| a & p);
            let mut want_cells = 0u32;
            for i in 0..9 {
                if union & (1 << i) != 0 {
                    want_cells |= 0b111 << (9 * (i / 3) + 3 * (i % 3));
                }
            }
            assert_eq!(CLOSED_CELLS[s as usize], want_cells, "CLOSED_CELLS[{s:o}]");
            let forced = if fits.is_empty() { 0 } else { inter };
            assert_eq!(FORCED[s as usize], forced, "FORCED[{s:o}]");
            // Soundness: forced minirows appear in every fitting permutation.
            for &p in &fits {
                assert_eq!(forced & p, forced);
            }
        }
    }

    #[test]
    fn expand_and_column_tables() {
        for m in 0..512usize {
            let mut want = 0u32;
            for i in 0..9 {
                if m & (1 << i) != 0 {
                    want |= 0b111 << (9 * (i / 3) + 3 * (i % 3));
                }
            }
            assert_eq!(EXPAND[m], want);

            let mut single = 0u16;
            let mut ok = ALL;
            for b in 0..3 {
                let cb = (m >> (3 * b)) & 7;
                if cb.count_ones() == 1 {
                    single |= 0o111 << b;
                    ok &= !(0o001_001_001 << (3 * b as u32 + (cb as u32).trailing_zeros()));
                }
            }
            assert_eq!(COL_SINGLE[m], single);
            assert_eq!(NEIGH_OK[m], ok);
        }
    }

    /// `shrink_band` must agree with its definition on every band mask, for
    /// whichever of the three implementations this target compiled.
    #[test]
    fn condense_matches_definition() {
        let reference = |m: u32| {
            let mut s = 0;
            for i in 0..9 {
                if (m >> (3 * i)) & 7 != 0 {
                    s |= 1 << i;
                }
            }
            s
        };
        // Exhaustive over a band's 27 bits is 134M; instead sweep every
        // single- and double-minirow pattern (which is what the hot path
        // mostly sees) plus a large pseudorandom sample.
        for a in 0..27u32 {
            for b in 0..27u32 {
                let m = (1 << a) | (1 << b);
                assert_eq!(super::shrink_band(m), reference(m), "m = {m:o}");
            }
        }
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..200_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let m = (rng as u32) & super::ALL;
            assert_eq!(super::shrink_band(m), reference(m), "m = {m:o}");
        }
    }
}
