//! AVX2 backend for the triad engine's vector vocabulary. See
//! [`tvec.rs`](tvec.rs) for what the operations mean.
//!
//! `C16` is one `__m256i`: a matrix row is exactly one 64-bit lane, and the
//! two 128-bit halves are matrix rows 0,1 and 2,3.

use core::arch::x86_64::*;

use super::{S0, S1, S2, S3, S4, S5, S6, S7, XX};

#[derive(Copy, Clone)]
pub(crate) struct C8(__m128i);

#[derive(Copy, Clone)]
pub(crate) struct C16(__m256i);

#[inline(always)]
pub(crate) unsafe fn c8(a: &[u16; 8]) -> C8 {
    C8(_mm_loadu_si128(a.as_ptr() as *const __m128i))
}

#[inline(always)]
pub(crate) unsafe fn c16(a: &[u16; 16]) -> C16 {
    C16(_mm256_loadu_si256(a.as_ptr() as *const __m256i))
}

#[inline(always)]
pub(crate) unsafe fn c16_bytes(a: &[u8; 32]) -> C16 {
    C16(_mm256_loadu_si256(a.as_ptr() as *const __m256i))
}

impl C8 {
    #[inline(always)]
    pub(crate) unsafe fn all(v: u16) -> C8 {
        C8(_mm_set1_epi16(v as i16))
    }
    #[inline(always)]
    pub(crate) unsafe fn zero() -> C8 {
        C8(_mm_setzero_si128())
    }
    #[inline(always)]
    pub(crate) unsafe fn and(self, o: C8) -> C8 {
        C8(_mm_and_si128(self.0, o.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn or(self, o: C8) -> C8 {
        C8(_mm_or_si128(self.0, o.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn xor(self, o: C8) -> C8 {
        C8(_mm_xor_si128(self.0, o.0))
    }
    /// self & !o
    #[inline(always)]
    pub(crate) unsafe fn and_not(self, o: C8) -> C8 {
        C8(_mm_andnot_si128(o.0, self.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn shuffle(self, ctrl: C8) -> C8 {
        C8(_mm_shuffle_epi8(self.0, ctrl.0))
    }
    /// Swap the two rows of a 2x4 view (64-bit halves).
    #[inline(always)]
    pub(crate) unsafe fn rotate_cols(self) -> C8 {
        C8(_mm_shuffle_epi32::<0b01001110>(self.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn all_zero(self) -> bool {
        _mm_testz_si128(self.0, self.0) != 0
    }
    #[inline(always)]
    pub(crate) unsafe fn intersects(self, o: C8) -> bool {
        _mm_testz_si128(self.0, o.0) == 0
    }
    /// Lowest set bit of each 16-bit lane.
    #[inline(always)]
    pub(crate) unsafe fn low_bit_per_lane(self) -> C8 {
        let neg = _mm_sub_epi16(_mm_setzero_si128(), self.0);
        C8(_mm_and_si128(self.0, neg))
    }
    /// Clear the lowest set bit of the vector viewed as one long integer.
    #[inline(always)]
    pub(crate) unsafe fn clear_low_bit(self) -> C8 {
        let cmp = _mm_cmpgt_epi64(self.0, _mm_setzero_si128());
        let one = _mm_andnot_si128(_mm_slli_si128::<1>(cmp), _mm_srli_epi64::<63>(cmp));
        C8(_mm_and_si128(self.0, _mm_sub_epi64(self.0, one)))
    }
    /// (min value, lane) over lanes after subtracting `floor`; packed as
    /// value in bits 0..16, lane in bits 16..19 (via phminposuw).
    #[inline(always)]
    pub(crate) unsafe fn minpos_after_sub(self, floor: u16) -> u32 {
        let adj = _mm_sub_epi16(self.0, _mm_set1_epi16(floor as i16));
        _mm_cvtsi128_si32(_mm_minpos_epu16(adj)) as u32
    }
}

/// Total set bits of each of six band-configuration vectors, packed into
/// lanes 0..6 of one vector, with lanes 6,7 filled with a sentinel above
/// every possible count (see `minpos_after_sub`, the only consumer).
///
/// Two `movq`s and a `popcnt` per band, assembled through the stack: 128-bit
/// horizontal popcount has no instruction, and the scalar unit is idle here.
#[inline(always)]
pub(crate) unsafe fn band_config_counts(bands: [C8; 6]) -> C8 {
    let total = |v: C8| {
        let lo = _mm_cvtsi128_si64(v.0) as u64;
        let hi = _mm_extract_epi64::<1>(v.0) as u64;
        (lo.count_ones() + hi.count_ones()) as u16
    };
    c8(&[
        total(bands[0]),
        total(bands[1]),
        total(bands[2]),
        total(bands[3]),
        total(bands[4]),
        total(bands[5]),
        0xffff,
        0xffff,
    ])
}

impl C16 {
    #[inline(always)]
    pub(crate) unsafe fn all(v: u16) -> C16 {
        C16(_mm256_set1_epi16(v as i16))
    }
    /// Every 64-bit lane (i.e. every matrix row) set to `v`.
    #[inline(always)]
    pub(crate) unsafe fn splat_u64(v: u64) -> C16 {
        C16(_mm256_set1_epi64x(v as i64))
    }
    #[inline(always)]
    pub(crate) unsafe fn from_parts(lo: C8, hi: C8) -> C16 {
        C16(_mm256_set_m128i(hi.0, lo.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn get_lo(self) -> C8 {
        C8(_mm256_castsi256_si128(self.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn get_hi(self) -> C8 {
        C8(_mm256_extracti128_si256::<1>(self.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn and(self, o: C16) -> C16 {
        C16(_mm256_and_si256(self.0, o.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn or(self, o: C16) -> C16 {
        C16(_mm256_or_si256(self.0, o.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn xor(self, o: C16) -> C16 {
        C16(_mm256_xor_si256(self.0, o.0))
    }
    /// self & !o
    #[inline(always)]
    pub(crate) unsafe fn and_not(self, o: C16) -> C16 {
        C16(_mm256_andnot_si256(o.0, self.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn shuffle(self, ctrl: C16) -> C16 {
        C16(_mm256_shuffle_epi8(self.0, ctrl.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn subset_of(self, o: C16) -> bool {
        _mm256_testc_si256(o.0, self.0) != 0
    }
    #[inline(always)]
    pub(crate) unsafe fn which_equal(self, o: C16) -> C16 {
        C16(_mm256_cmpeq_epi16(self.0, o.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn which_nonzero(self) -> C16 {
        C16(_mm256_cmpgt_epi16(self.0, _mm256_setzero_si256()))
    }
    #[inline(always)]
    pub(crate) unsafe fn any_less_than(self, o: C16) -> bool {
        let lt = _mm256_cmpgt_epi16(o.0, self.0);
        // Deliberately a movemask rather than the shorter PTEST form: PTEST
        // keeps the check on the vector ports, which are the bottleneck in
        // this loop, while movemask hands it to the integer side. Measured
        // ~1% faster despite costing an extra instruction.
        _mm256_movemask_epi8(lt) != 0
    }
    /// Per-lane popcount, assuming the 7 high bits of every lane are zero.
    ///
    /// The nibble table is left as a plain constant. Pinning it in a register
    /// with inline asm was tried: it did remove the per-iteration
    /// `vbroadcasti128`, but spending one of sixteen vector registers cost
    /// three instructions elsewhere and measured no faster. The broadcast
    /// issues on the load port, which this loop is not short of.
    #[inline(always)]
    pub(crate) unsafe fn popcounts9(self) -> C16 {
        let lut = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
            2, 3, 3, 4,
        );
        let mask4 = _mm256_set1_epi16(0x0f);
        let sum_0_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(self.0, mask4));
        let sum_4_7 = _mm256_shuffle_epi8(lut, _mm256_srli_epi16::<4>(self.0));
        let sum_0_7 = _mm256_add_epi16(sum_0_3, sum_4_7);
        C16(_mm256_add_epi16(sum_0_7, _mm256_srli_epi16::<8>(self.0)))
    }
    /// Shift each matrix row's elements up by one position, zero-filling.
    /// A matrix row is one 64-bit lane, so this is a lane-wise shift.
    #[inline(always)]
    pub(crate) unsafe fn shift_rows_up1(self) -> C16 {
        C16(_mm256_slli_epi64::<16>(self.0))
    }
    /// Shift each matrix row's elements up by two positions, zero-filling.
    #[inline(always)]
    pub(crate) unsafe fn shift_rows_up2(self) -> C16 {
        C16(_mm256_slli_epi64::<32>(self.0))
    }
    /// Per lane, the union of the other three elements of its matrix row:
    /// three rotations of the 4-element row under a two-level OR tree. The
    /// rotate-by-two is an in-lane `vpshufd`, the other two `vpshufb`.
    #[inline(always)]
    pub(crate) unsafe fn row_peers(self) -> C16 {
        static ROT1: [u16; 16] =
            [S1, S2, S3, S0, S5, S6, S7, S4, S1, S2, S3, S0, S5, S6, S7, S4];
        static ROT3: [u16; 16] =
            [S3, S0, S1, S2, S7, S4, S5, S6, S3, S0, S1, S2, S7, S4, S5, S6];
        let r1 = C16(_mm256_shuffle_epi8(self.0, c16(&ROT1).0));
        let r2 = C16(_mm256_shuffle_epi32::<0b10110001>(self.0));
        let r3 = C16(_mm256_shuffle_epi8(self.0, c16(&ROT3).0));
        r1.or(r2).or(r3)
    }
    /// Per lane, the union of the other three elements of its matrix column.
    ///
    /// Rotate three ways off the same source and OR as a balanced tree. The
    /// obvious `x |= rot(x)` log-reduction is one shuffle cheaper, but it
    /// chains permute -> or -> permute -> or, and a cross-lane permute costs 3
    /// cycles: 8 cycles of latency against 5 for three independent permutes
    /// plus a two-level OR. This sits on the loop-carried critical path, where
    /// latency is worth more than the extra op.
    #[inline(always)]
    pub(crate) unsafe fn col_peers(self) -> C16 {
        let c1 = C16(_mm256_permute4x64_epi64::<0b00111001>(self.0));
        let c2 = C16(_mm256_permute4x64_epi64::<0b01001110>(self.0));
        let c3 = C16(_mm256_permute4x64_epi64::<0b10010011>(self.0));
        c1.or(c2).or(c3)
    }
    /// Spread an asserted-cell vector into the eliminations it implies
    /// positionally: the whole box's union in the nine cell lanes, each
    /// matrix column's union in matrix row 3 (the vertical triads), and zero
    /// in column 3 (the horizontal triads, which `across_rows` supplies).
    ///
    /// Fold the four rows together, then OR in two rotations of the 3x3's
    /// columns -- which, the rows being already folded, unions the entire
    /// box. Row 3's lanes are left out of the rotation, so they keep the
    /// per-column union.
    #[inline(always)]
    pub(crate) unsafe fn box_and_column_unions(self) -> C16 {
        static ROT1: [u16; 16] =
            [S1, S2, S0, S3, S5, S6, S4, S7, S1, S2, S0, S3, S4, S5, S6, S7];
        static ROT2: [u16; 16] =
            [S2, S0, S1, S3, S6, S4, S5, S7, S2, S0, S1, S3, S4, S5, S6, S7];
        let half = self.or(self.swap_row_pairs());
        let cols = half.or(half.swap_rows_in_pair());
        cols.or(cols.shuffle(c16(&ROT1))).or(cols.shuffle(c16(&ROT2)))
    }
    /// Gather the box's triad literals into band-message form: the low half
    /// becomes `[_, _, _, _, lane3, lane7, lane11, _]` and the high half is
    /// unchanged (its vertical triads already sit in positions 4..6).
    ///
    /// One fused permutation rather than extracting the two triad sets and
    /// reassembling them: of the three horizontal triads only the third
    /// lives in the other 128-bit lane, so a single half-swap plus two
    /// in-lane shuffles reaches everything. That replaces shuffle + 2x
    /// vextracti128 + or + vinserti128 (4 shuffle-port ops) with vpermq +
    /// 2x vpshufb + or (3). The identity on the high half is not free here:
    /// a 256-bit shuffle has to cover both halves.
    #[inline(always)]
    pub(crate) unsafe fn triad_message(self) -> C16 {
        static MSG_A: [u16; 16] = [
            XX, XX, XX, XX, S3, S7, XX, XX, // low: lanes 3,7 -> 4,5
            S0, S1, S2, S3, S4, S5, S6, S7, // high: identity passthrough
        ];
        static MSG_B: [u16; 16] = [
            XX, XX, XX, XX, XX, XX, S3, XX, // low: swapped lane 3 (= 11) -> 6
            XX, XX, XX, XX, XX, XX, XX, XX, // high: contribute nothing
        ];
        self.shuffle(c16(&MSG_A))
            .or(self.swap_row_pairs().shuffle(c16(&MSG_B)))
    }
    /// Exchange matrix rows 0,1 with rows 2,3 (the two 128-bit halves).
    #[inline(always)]
    unsafe fn swap_row_pairs(self) -> C16 {
        C16(_mm256_permute2x128_si256::<0x01>(self.0, self.0))
    }
    /// Exchange matrix row 0 with 1 and row 2 with 3 (in-lane).
    #[inline(always)]
    unsafe fn swap_rows_in_pair(self) -> C16 {
        C16(_mm256_shuffle_epi32::<0x4E>(self.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn extract_rows_u64(self) -> [u64; 4] {
        [
            _mm256_extract_epi64::<0>(self.0) as u64,
            _mm256_extract_epi64::<1>(self.0) as u64,
            _mm256_extract_epi64::<2>(self.0) as u64,
            _mm256_extract_epi64::<3>(self.0) as u64,
        ]
    }
}
