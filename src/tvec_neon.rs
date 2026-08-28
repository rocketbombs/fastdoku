//! NEON backend for the triad engine's vector vocabulary. See
//! [`tvec.rs`](tvec.rs) for what the operations mean.
//!
//! `C16` is a pair of `uint16x8_t`: `lo` is matrix rows 0 and 1, `hi` rows 2
//! and 3, so a matrix row is one 64-bit lane exactly as it is under AVX2 and
//! every shuffle-control table carries over unchanged.
//!
//! Splitting the box across two registers is not the loss it looks like.
//! Apple's cores issue four 128-bit vector operations per cycle — the same
//! width per cycle as two 256-bit ports — and aarch64 has 32 vector
//! registers to AVX2's 16, which matters for a loop the x86 notes repeatedly
//! record as register-bound. Three properties then go the other way:
//!
//! - **Half-swaps are free.** Exchanging matrix rows 0,1 with 2,3 is a
//!   rename, not an instruction; on AVX2 it is a 3-cycle `vperm2i128`. That
//!   is why `box_and_column_unions` folds where AVX2 rotates.
//! - **`col_peers` costs 5 operations, not 6**, because the two `ext`s it
//!   needs supply both the rotate-by-1 and the rotate-by-3 (they are the
//!   same pair of values, swapped), and their union is common to both
//!   halves. AVX2 needs three independent `vpermq`s.
//! - **`popcounts9` is two instructions**, `cnt` + `uaddlp`, against the
//!   nibble-table dance AVX2 needs — NEON has per-byte popcount in hardware
//!   and a 16-bit lane holding a 9-bit set is exactly two bytes.
//!
//! What NEON lacks is a cheap `movemask`/`ptest`: every "is anything set"
//! question ends in a `umaxv` reduction and a vector-to-general-register
//! move. Those sit on branch conditions rather than on the loop-carried
//! dependency chain, so the extra latency is absorbed by speculation, but it
//! is why `any_less_than` and `subset_of` are written to reduce *once* over
//! both halves rather than testing each.

use core::arch::aarch64::*;

use super::{CELLS_3X3, S0, S1, S2, S3, S4, S5, S6, S7};

#[derive(Copy, Clone)]
pub(crate) struct C8(uint16x8_t);

/// `lo` = matrix rows 0,1; `hi` = matrix rows 2,3.
#[derive(Copy, Clone)]
pub(crate) struct C16 {
    lo: uint16x8_t,
    hi: uint16x8_t,
}

#[inline(always)]
unsafe fn as_u64(x: uint16x8_t) -> uint64x2_t {
    vreinterpretq_u64_u16(x)
}

#[inline(always)]
unsafe fn as_u16(x: uint64x2_t) -> uint16x8_t {
    vreinterpretq_u16_u64(x)
}

/// One 128-bit byte shuffle. `vqtbl1q_u8` and `vpshufb` agree on the
/// encoding this vocabulary uses: an index of 0..16 selects that byte, and
/// an out-of-range index (`XX` is `0xffff`) yields zero.
#[inline(always)]
unsafe fn tbl(v: uint16x8_t, ctrl: uint16x8_t) -> uint16x8_t {
    vreinterpretq_u16_u8(vqtbl1q_u8(
        vreinterpretq_u8_u16(v),
        vreinterpretq_u8_u16(ctrl),
    ))
}

/// True iff any bit of `x` is set, in one horizontal reduction.
#[inline(always)]
unsafe fn any_bit(x: uint16x8_t) -> bool {
    vmaxvq_u32(vreinterpretq_u32_u16(x)) != 0
}

#[inline(always)]
pub(crate) unsafe fn c8(a: &[u16; 8]) -> C8 {
    C8(vld1q_u16(a.as_ptr()))
}

#[inline(always)]
pub(crate) unsafe fn c16(a: &[u16; 16]) -> C16 {
    C16 {
        lo: vld1q_u16(a.as_ptr()),
        hi: vld1q_u16(a.as_ptr().add(8)),
    }
}

#[inline(always)]
pub(crate) unsafe fn c16_bytes(a: &[u8; 32]) -> C16 {
    C16 {
        lo: vreinterpretq_u16_u8(vld1q_u8(a.as_ptr())),
        hi: vreinterpretq_u16_u8(vld1q_u8(a.as_ptr().add(16))),
    }
}

impl C8 {
    #[inline(always)]
    pub(crate) unsafe fn all(v: u16) -> C8 {
        C8(vdupq_n_u16(v))
    }
    #[inline(always)]
    pub(crate) unsafe fn zero() -> C8 {
        C8(vdupq_n_u16(0))
    }
    #[inline(always)]
    pub(crate) unsafe fn and(self, o: C8) -> C8 {
        C8(vandq_u16(self.0, o.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn or(self, o: C8) -> C8 {
        C8(vorrq_u16(self.0, o.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn xor(self, o: C8) -> C8 {
        C8(veorq_u16(self.0, o.0))
    }
    /// self & !o
    #[inline(always)]
    pub(crate) unsafe fn and_not(self, o: C8) -> C8 {
        C8(vbicq_u16(self.0, o.0))
    }
    #[inline(always)]
    pub(crate) unsafe fn shuffle(self, ctrl: C8) -> C8 {
        C8(tbl(self.0, ctrl.0))
    }
    /// Swap the two rows of a 2x4 view (64-bit halves).
    #[inline(always)]
    pub(crate) unsafe fn rotate_cols(self) -> C8 {
        C8(as_u16(vextq_u64::<1>(as_u64(self.0), as_u64(self.0))))
    }
    #[inline(always)]
    pub(crate) unsafe fn all_zero(self) -> bool {
        !any_bit(self.0)
    }
    #[inline(always)]
    pub(crate) unsafe fn intersects(self, o: C8) -> bool {
        any_bit(vandq_u16(self.0, o.0))
    }
    /// Lowest set bit of each 16-bit lane.
    #[inline(always)]
    pub(crate) unsafe fn low_bit_per_lane(self) -> C8 {
        let neg = vreinterpretq_u16_s16(vnegq_s16(vreinterpretq_s16_u16(self.0)));
        C8(vandq_u16(self.0, neg))
    }
    /// Clear the lowest set bit of the vector viewed as one long integer,
    /// i.e. `x & (x - 1)` in 128-bit arithmetic. The borrow out of the low
    /// half is exactly "the low half was zero", so the subtrahend is
    /// `[1, lo == 0]` and no 128-bit adder is needed.
    #[inline(always)]
    pub(crate) unsafe fn clear_low_bit(self) -> C8 {
        let x = as_u64(self.0);
        let one = vdupq_n_u64(1);
        let is_zero = vandq_u64(vceqzq_u64(x), one);
        // vextq_u64::<1>(a, b) is [a.1, b.0], so this is [1, (lo == 0)].
        let sub = vextq_u64::<1>(one, is_zero);
        C8(as_u16(vandq_u64(x, vsubq_u64(x, sub))))
    }
    /// (min value, lane) over lanes after subtracting `floor`; packed as
    /// value in bits 0..16 and lane in bits 16..19, matching `phminposuw`.
    ///
    /// NEON has a horizontal minimum but no argmin, so the lane index rides
    /// in the low three bits of the value being minimized: the smallest
    /// packed word is the smallest adjusted count, and among ties the lowest
    /// lane — which is `phminposuw`'s tie rule too. The shift is *saturating*
    /// so that the caller's sentinel lanes (`0xffff`, which underflow the
    /// subtraction) stay above every genuine count instead of wrapping back
    /// under it.
    #[inline(always)]
    pub(crate) unsafe fn minpos_after_sub(self, floor: u16) -> u32 {
        static LANES: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let adj = vsubq_u16(self.0, vdupq_n_u16(floor));
        let packed = vorrq_u16(vqshlq_n_u16::<3>(adj), vld1q_u16(LANES.as_ptr()));
        let best = vminvq_u16(packed) as u32;
        (best >> 3) | ((best & 7) << 16)
    }
}

/// Total set bits of each of six band-configuration vectors, packed into
/// lanes 0..6 of one vector, with lanes 6,7 filled with a sentinel above
/// every possible count (see `minpos_after_sub`, the only consumer).
///
/// The obvious form -- six independent `addv` reductions written into an
/// array and reloaded -- is the worst thing NEON does: six horizontal
/// reductions, six vector-to-general-register moves and a round trip
/// through memory, all to build a vector. `addp` adds *adjacent pairs
/// across two vectors*, so three levels of it fold six 8-lane count vectors
/// into exactly the packed layout wanted, without a scalar in sight.
///
/// This runs once per search node, so it is paid on every branch decision
/// of every puzzle that guesses at all.
#[inline(always)]
pub(crate) unsafe fn band_config_counts(bands: [C8; 6]) -> C8 {
    static PAD: [u16; 8] = [0, 0, 0, 0, 0, 0, 0xffff, 0xffff];
    // Per-lane popcounts: `cnt` then one pairwise-widening add per band.
    let c = bands.map(|b| vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u16(b.0))));
    // Level 1 halves each band's lanes; level 2 quarters them; level 3
    // finishes bands 0..4 and, from the doubled `q1`, bands 4 and 5.
    let q0 = vpaddq_u16(vpaddq_u16(c[0], c[1]), vpaddq_u16(c[2], c[3]));
    let p2 = vpaddq_u16(c[4], c[5]);
    let q1 = vpaddq_u16(p2, p2);
    // Lanes 6,7 hold copies of bands 4 and 5; the sentinel overwrites them,
    // and no genuine count can reach it (a band has at most 54 bits).
    C8(vorrq_u16(vpaddq_u16(q0, q1), vld1q_u16(PAD.as_ptr())))
}

impl C16 {
    #[inline(always)]
    pub(crate) unsafe fn all(v: u16) -> C16 {
        let x = vdupq_n_u16(v);
        C16 { lo: x, hi: x }
    }
    /// Every 64-bit lane (i.e. every matrix row) set to `v`.
    #[inline(always)]
    pub(crate) unsafe fn splat_u64(v: u64) -> C16 {
        let x = as_u16(vdupq_n_u64(v));
        C16 { lo: x, hi: x }
    }
    #[inline(always)]
    pub(crate) unsafe fn from_parts(lo: C8, hi: C8) -> C16 {
        C16 { lo: lo.0, hi: hi.0 }
    }
    #[inline(always)]
    pub(crate) unsafe fn get_lo(self) -> C8 {
        C8(self.lo)
    }
    #[inline(always)]
    pub(crate) unsafe fn get_hi(self) -> C8 {
        C8(self.hi)
    }
    #[inline(always)]
    pub(crate) unsafe fn and(self, o: C16) -> C16 {
        C16 {
            lo: vandq_u16(self.lo, o.lo),
            hi: vandq_u16(self.hi, o.hi),
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn or(self, o: C16) -> C16 {
        C16 {
            lo: vorrq_u16(self.lo, o.lo),
            hi: vorrq_u16(self.hi, o.hi),
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn xor(self, o: C16) -> C16 {
        C16 {
            lo: veorq_u16(self.lo, o.lo),
            hi: veorq_u16(self.hi, o.hi),
        }
    }
    /// self & !o
    #[inline(always)]
    pub(crate) unsafe fn and_not(self, o: C16) -> C16 {
        C16 {
            lo: vbicq_u16(self.lo, o.lo),
            hi: vbicq_u16(self.hi, o.hi),
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn shuffle(self, ctrl: C16) -> C16 {
        C16 {
            lo: tbl(self.lo, ctrl.lo),
            hi: tbl(self.hi, ctrl.hi),
        }
    }
    /// One reduction over both halves rather than two: the `bic`s are free
    /// on the vector units, the `umaxv` and its register move are not.
    #[inline(always)]
    pub(crate) unsafe fn subset_of(self, o: C16) -> bool {
        let outside = vorrq_u16(vbicq_u16(self.lo, o.lo), vbicq_u16(self.hi, o.hi));
        !any_bit(outside)
    }
    #[inline(always)]
    pub(crate) unsafe fn which_equal(self, o: C16) -> C16 {
        C16 {
            lo: vceqq_u16(self.lo, o.lo),
            hi: vceqq_u16(self.hi, o.hi),
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn which_nonzero(self) -> C16 {
        C16 {
            lo: vtstq_u16(self.lo, self.lo),
            hi: vtstq_u16(self.hi, self.hi),
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn any_less_than(self, o: C16) -> bool {
        let lt = vorrq_u16(vcltq_u16(self.lo, o.lo), vcltq_u16(self.hi, o.hi));
        any_bit(lt)
    }
    /// Per-lane popcount. A 16-bit lane holding a 9-bit set is two bytes, so
    /// hardware per-byte `cnt` plus one pairwise-widening add is exact — no
    /// assumption about the high bits, and no nibble table.
    #[inline(always)]
    pub(crate) unsafe fn popcounts9(self) -> C16 {
        C16 {
            lo: vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u16(self.lo))),
            hi: vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u16(self.hi))),
        }
    }
    /// Shift each matrix row's elements up by one position, zero-filling.
    /// A matrix row is one 64-bit lane, so this is a lane-wise shift.
    #[inline(always)]
    pub(crate) unsafe fn shift_rows_up1(self) -> C16 {
        C16 {
            lo: as_u16(vshlq_n_u64::<16>(as_u64(self.lo))),
            hi: as_u16(vshlq_n_u64::<16>(as_u64(self.hi))),
        }
    }
    /// Shift each matrix row's elements up by two positions, zero-filling.
    #[inline(always)]
    pub(crate) unsafe fn shift_rows_up2(self) -> C16 {
        C16 {
            lo: as_u16(vshlq_n_u64::<32>(as_u64(self.lo))),
            hi: as_u16(vshlq_n_u64::<32>(as_u64(self.hi))),
        }
    }
    /// Per lane, the union of the other three elements of its matrix row.
    ///
    /// Rotate-by-two is `rev64` on 32-bit elements: one instruction and no
    /// control register, since rotating four 16-bit lanes by two inside a
    /// 64-bit row is exactly swapping that row's two 32-bit halves. The
    /// other two rotations are table lookups, issued independently off
    /// `self` so the OR tree is two levels deep rather than serialized.
    #[inline(always)]
    pub(crate) unsafe fn row_peers(self) -> C16 {
        static ROT1: [u16; 16] =
            [S1, S2, S3, S0, S5, S6, S7, S4, S1, S2, S3, S0, S5, S6, S7, S4];
        static ROT3: [u16; 16] =
            [S3, S0, S1, S2, S7, S4, S5, S6, S3, S0, S1, S2, S7, S4, S5, S6];
        let r1 = c16(&ROT1);
        let r3 = c16(&ROT3);
        let rev = |v: uint16x8_t| {
            vreinterpretq_u16_u32(vrev64q_u32(vreinterpretq_u32_u16(v)))
        };
        C16 {
            lo: vorrq_u16(vorrq_u16(tbl(self.lo, r1.lo), tbl(self.lo, r3.lo)), rev(self.lo)),
            hi: vorrq_u16(vorrq_u16(tbl(self.hi, r1.hi), tbl(self.hi, r3.hi)), rev(self.hi)),
        }
    }
    /// Per lane, the union of the other three elements of its matrix column.
    ///
    /// Five operations, against AVX2's three `vpermq` plus two ORs, because
    /// the split representation makes two of the three rotations free and
    /// shares work between the halves. Writing the rows `r0..r3`:
    ///
    /// - rotate-by-2 is `(hi, lo)` — a rename;
    /// - rotate-by-1 is `(ext(lo,hi), ext(hi,lo))` = `(E1, E2)`, and
    ///   rotate-by-3 is the same pair swapped, `(E2, E1)`;
    /// - so both halves want `E1 | E2`, computed once, and each then ORs in
    ///   the opposite half.
    ///
    /// Two `ext`s and three ORs, two levels deep — which matters, because
    /// this is on the fixpoint loop's carried dependency chain.
    #[inline(always)]
    pub(crate) unsafe fn col_peers(self) -> C16 {
        let (lo, hi) = (as_u64(self.lo), as_u64(self.hi));
        let e1 = vextq_u64::<1>(lo, hi); // (r1, r2)
        let e2 = vextq_u64::<1>(hi, lo); // (r3, r0)
        let shared = vorrq_u64(e1, e2); // (r1|r3, r2|r0)
        C16 {
            lo: as_u16(vorrq_u64(shared, hi)), // (r1|r2|r3, r0|r2|r3)
            hi: as_u16(vorrq_u64(shared, lo)), // (r0|r1|r3, r0|r1|r2)
        }
    }
    /// Spread an asserted-cell vector into the eliminations it implies
    /// positionally: the whole box's union in the nine cell lanes, each
    /// matrix column's union in matrix row 3 (the vertical triads), and zero
    /// in column 3 (the horizontal triads, which `across_rows` supplies).
    ///
    /// Nine operations against AVX2's eleven, and none of them a table
    /// lookup -- which is the point, because AVX2's two row rotations need
    /// four control registers here (one per half) and four loads to fill
    /// them, in a loop that is already reloading constants. Both reductions
    /// fall out of reversals instead:
    ///
    /// - the four matrix rows fold with one OR (the half swap is a rename)
    ///   and one `ext`, leaving every lane of both halves holding its
    ///   column's union;
    /// - those four columns then fold with `rev64`/`rev32`, two instructions
    ///   that need no control register at all, leaving the box's union in
    ///   every lane.
    ///
    /// The two are then blended by the cell mask, which is bitwise rather
    /// than lane-wise -- safe because both inputs are 9-bit sets and the
    /// mask's set lanes are exactly nine bits wide, so a masked-out lane
    /// contributes nothing. It is the same depth as the AVX2 form, which
    /// matters more than the instruction count: this sits on the fixpoint
    /// loop's carried dependency chain.
    #[inline(always)]
    pub(crate) unsafe fn box_and_column_unions(self) -> C16 {
        let cells = c16(&CELLS_3X3);
        let h = vorrq_u16(self.lo, self.hi);
        let cu = vorrq_u16(h, as_u16(vextq_u64::<1>(as_u64(h), as_u64(h))));
        let u = vorrq_u16(
            cu,
            vreinterpretq_u16_u32(vrev64q_u32(vreinterpretq_u32_u16(cu))),
        );
        let bu = vorrq_u16(u, vrev32q_u16(u));
        C16 {
            lo: vbslq_u16(cells.lo, bu, cu),
            hi: vbslq_u16(cells.hi, bu, cu),
        }
    }
    /// Gather the box's triad literals into band-message form: the low half
    /// becomes `[_, _, _, _, lane3, lane7, lane11, _]` and the high half is
    /// unchanged (its vertical triads already sit in positions 4..6).
    ///
    /// One instruction. `vqtbl2q_u8` indexes all 32 bytes of a register
    /// pair, which no AVX2 shuffle can do, so the third horizontal triad --
    /// the one in the other half, and the whole reason the x86 form needs a
    /// half-swap and a second shuffle -- is just another index here. And the
    /// high half is a literal identity, so it costs nothing at all rather
    /// than the half of a 256-bit shuffle AVX2 must spend on it.
    #[inline(always)]
    pub(crate) unsafe fn triad_message(self) -> C16 {
        // Byte indices into the 32-byte pair; 0xff selects nothing.
        static GATHER: [u8; 16] = [
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // lanes 0..4: empty
            6, 7, // lane 4 <- lane 3 (horizontal triad of matrix row 0)
            14, 15, // lane 5 <- lane 7 (matrix row 1)
            22, 23, // lane 6 <- lane 11 (matrix row 2, in the high half)
            0xff, 0xff, // lane 7: empty
        ];
        let pair = uint8x16x2_t(
            vreinterpretq_u8_u16(self.lo),
            vreinterpretq_u8_u16(self.hi),
        );
        C16 {
            lo: vreinterpretq_u16_u8(vqtbl2q_u8(pair, vld1q_u8(GATHER.as_ptr()))),
            hi: self.hi,
        }
    }
    #[inline(always)]
    pub(crate) unsafe fn extract_rows_u64(self) -> [u64; 4] {
        let (lo, hi) = (as_u64(self.lo), as_u64(self.hi));
        [
            vgetq_lane_u64::<0>(lo),
            vgetq_lane_u64::<1>(lo),
            vgetq_lane_u64::<0>(hi),
            vgetq_lane_u64::<1>(hi),
        ]
    }
}
