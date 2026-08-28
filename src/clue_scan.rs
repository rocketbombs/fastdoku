//! Locating the clues of an 81-byte grid without branching on each cell.
//!
//! Both engines start by walking the filled cells of a puzzle, and the clue
//! pattern is essentially random, so a per-cell `if clues[i] != 0` costs a
//! mispredict on a large fraction of 81 cells. Comparing whole vectors
//! against zero and bit-scanning the result instead makes the scan's control
//! flow depend only on the clue *count*, not on where the clues are.
//!
//! This is the one piece of vector code both `jcz` and `triad` need, and it
//! runs on every puzzle by both routes (`auto` pays it twice when it defers),
//! so it lives here rather than being written out in each engine.
//!
//! What is shared is the *masks*, not the walk over them. An earlier version
//! took the per-clue work as a closure and owned the bit-scan loops too,
//! which reads better and measured ~1% slower on `kaggle` — the callers'
//! accumulators are small arrays, and handing them to a closure that crosses
//! a (fully inlined) call boundary was enough to stop LLVM keeping them
//! where it had. Returning two integers leaves each caller's loop exactly
//! the code it was before.

/// Clue cells as bitmasks: cells 0..64 in the `u64`, cells 64..81 in the
/// low 17 bits of the `u32`. A set bit means the cell is filled.
///
/// Three overlapping 32-byte compares cover 81 bytes: 0..32, 32..64 and
/// 49..81, the last shifted back into place.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn clue_masks(clues: &[u8; 81]) -> (u64, u32) {
    use core::arch::x86_64::*;
    // SAFETY: all three loads stay inside the 81-byte array.
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
    (!(m_a as u64 | (m_b as u64) << 32), (!m_c >> 15) & 0x1ffff)
}

/// NEON has no `movemask`. The substitute here is one `and` against per-lane
/// bit weights and two `addv` reductions per 16 bytes: `addv` sums the eight
/// surviving weights, which is the byte mask for that half. (The narrowing
/// `shrn` idiom is shorter but leaves one *nibble* per byte, and compacting
/// 80 nibbles back down would cost more than the reductions it saved.)
///
/// Five 16-byte chunks cover cells 0..80 and the last cell is scalar, rather
/// than a sixth overlapping load: overlapping would need the duplicate lanes
/// masked off, which costs more than the one byte it saves.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn clue_masks(clues: &[u8; 81]) -> (u64, u32) {
    use core::arch::aarch64::*;
    static WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let w = vld1q_u8(WEIGHTS.as_ptr());
    let p = clues.as_ptr();
    // SAFETY: chunk 4 ends at byte 80, inside the 81-byte array.
    let chunk = |base: usize| -> u64 {
        let v = vld1q_u8(p.add(base));
        let bits = vandq_u8(vtstq_u8(v, v), w);
        vaddv_u8(vget_low_u8(bits)) as u64 | (vaddv_u8(vget_high_u8(bits)) as u64) << 8
    };
    let lo = chunk(0) | chunk(16) << 16 | chunk(32) << 32 | chunk(48) << 48;
    let hi = chunk(64) as u32 | ((*clues.get_unchecked(80) != 0) as u32) << 16;
    (lo, hi)
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64"
)))]
#[inline(always)]
pub(crate) unsafe fn clue_masks(clues: &[u8; 81]) -> (u64, u32) {
    let mut lo = 0u64;
    let mut hi = 0u32;
    for cell in 0..64 {
        lo |= ((*clues.get_unchecked(cell) != 0) as u64) << cell;
    }
    for cell in 64..81 {
        hi |= ((*clues.get_unchecked(cell) != 0) as u32) << (cell - 64);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    #[test]
    fn marks_exactly_the_non_zero_cells() {
        // Random patterns exercise every chunk boundary the vector
        // implementations use (16, 32, 48, 49, 64, 80).
        let mut rng = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..2000 {
            let mut grid = [0u8; 81];
            let (mut want_lo, mut want_hi) = (0u64, 0u32);
            for (cell, slot) in grid.iter_mut().enumerate() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                if rng & 3 == 0 {
                    *slot = (rng >> 40) as u8 | 1;
                    if cell < 64 {
                        want_lo |= 1 << cell;
                    } else {
                        want_hi |= 1 << (cell - 64);
                    }
                }
            }
            assert_eq!(unsafe { super::clue_masks(&grid) }, (want_lo, want_hi));
        }
    }
}
