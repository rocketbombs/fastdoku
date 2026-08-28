//! The triad engine's vector vocabulary, and its per-architecture backends.
//!
//! `src/triad.rs` is written entirely against the two types defined here, so
//! the engine's *architecture* — 4x4 matrices of 9-bit candidate sets, band
//! configurations, the message-passing fixpoint — is portable, and only the
//! two dozen primitives below are not. A backend is one file:
//!
//! - [`tvec_x86.rs`](tvec_x86.rs): AVX2, `C16` = one `__m256i`.
//! - [`tvec_neon.rs`](tvec_neon.rs): NEON, `C16` = a pair of `uint16x8_t`.
//!
//! **`C16` is a 4x4 matrix of 16-bit lanes** (a box), lanes 0..16 row-major,
//! each holding a 9-bit candidate set. **`C8` is 8 lanes** (a band's six
//! configuration lanes plus two of padding, or half a box).
//!
//! The vocabulary is deliberately semantic rather than mechanical: it names
//! *"the union of a lane's three peers along this axis"* (`row_peers`,
//! `col_peers`), not the individual rotations that x86 happens to reach it
//! with. Those two operations sit on the fixpoint loop's carried dependency
//! chain and are where the two architectures diverge most — see each
//! backend's implementation for what it costs there. `triad_message`, the
//! box-to-band permutation, is here for the same reason: it is a fixed
//! shuffle either way, but the two reach it so differently (three
//! shuffle-port operations against one) that spelling it out at the call
//! site would fix the x86 answer for both.
//!
//! **Shuffle controls are byte-pair selectors**, shared by both backends
//! because `vpshufb` and `vqtbl1q_u8` agree on the encoding: an index of
//! 0..16 selects that byte of the same 128-bit half, and an out-of-range
//! index (`XX`, whose bytes are `0xff`) emits zero. A control is written as
//! 16-bit lanes `S0`..`S7` naming which *lane* of the half to take.

/// A shuffle-control lane selecting nothing (emits zero in both backends).
pub(crate) const XX: u16 = 0xffff;

/// A full 9-digit candidate set: what one lane holds when nothing is known.
pub(crate) const ALL: u16 = 0x1ff;

/// Which lanes of the 4x4 matrix are cells rather than triad margins. It is
/// a full candidate set rather than `0xffff` in those lanes so that it also
/// serves as a blend selector between two 9-bit quantities.
pub(crate) static CELLS_3X3: [u16; 16] = [
    ALL, ALL, ALL, 0, ALL, ALL, ALL, 0, ALL, ALL, ALL, 0, 0, 0, 0, 0,
];

// Byte-pair shuffle selectors for 16-bit lanes 0..8 of a 128-bit half.
pub(crate) const S0: u16 = 0x0100;
pub(crate) const S1: u16 = 0x0302;
pub(crate) const S2: u16 = 0x0504;
pub(crate) const S3: u16 = 0x0706;
pub(crate) const S4: u16 = 0x0908;
pub(crate) const S5: u16 = 0x0b0a;
pub(crate) const S6: u16 = 0x0d0c;
pub(crate) const S7: u16 = 0x0f0e;

// The module itself only exists under `cfg(triad_engine)` (see `build.rs`),
// which is where the AVX2 requirement is tested; here the architecture alone
// picks the backend. Testing `target_feature` again would be wrong as well as
// redundant, because rustdoc type-checks doctests without the build's
// `RUSTFLAGS` and would then select neither.
#[cfg(target_arch = "x86_64")]
#[path = "tvec_x86.rs"]
mod backend;

#[cfg(target_arch = "aarch64")]
#[path = "tvec_neon.rs"]
mod backend;

pub(crate) use backend::{band_config_counts, c16, c16_bytes, c8, C16, C8};

/// Every operation of the vocabulary, against a scalar model of what it is
/// supposed to mean.
///
/// The engine's own cross-validation (`lib.rs`) would catch a broken backend
/// too, but only as "the triad engine disagrees with the baseline on puzzle
/// N". These localize it, and they are the only check on the NEON backend
/// that can run before hardware does: they exercise each primitive directly,
/// on inputs drawn from the domain the engine actually uses (9-bit candidate
/// sets, small popcounts, in-range shuffle controls).
#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// A candidate set: 9 bits, as every lane the engine builds is.
        fn set9(&mut self) -> u16 {
            (self.next() & 0x1ff) as u16
        }
    }

    fn read16(v: C16) -> [u16; 16] {
        let rows = unsafe { v.extract_rows_u64() };
        let mut out = [0u16; 16];
        for (r, row) in rows.iter().enumerate() {
            for c in 0..4 {
                out[r * 4 + c] = (row >> (16 * c)) as u16;
            }
        }
        out
    }

    fn read8(v: C8) -> [u16; 8] {
        let full = read16(unsafe { C16::from_parts(v, v) });
        full[..8].try_into().unwrap()
    }

    fn make16(a: &[u16; 16]) -> C16 {
        unsafe { c16(a) }
    }

    fn make8(a: &[u16; 8]) -> C8 {
        unsafe { c8(a) }
    }

    /// The shuffle contract both backends implement: byte indices 0..16
    /// select within the same 128-bit half, `XX` (0xffff, i.e. bytes 0xff)
    /// selects nothing. Indices in 16..255 are outside the contract, and no
    /// table in the engine uses them.
    fn shuffle_ref(src: &[u16; 16], ctrl: &[u16; 16]) -> [u16; 16] {
        let sb: [u8; 32] = unsafe { core::mem::transmute(*src) };
        let cb: [u8; 32] = unsafe { core::mem::transmute(*ctrl) };
        let mut ob = [0u8; 32];
        for i in 0..32 {
            let idx = cb[i];
            assert!(idx < 16 || idx == 0xff, "control byte outside the contract");
            if idx < 16 {
                ob[i] = sb[(i / 16) * 16 + idx as usize];
            }
        }
        unsafe { core::mem::transmute(ob) }
    }

    /// A random control table over the agreed subset: lane selectors and XX.
    fn random_ctrl(rng: &mut Rng) -> [u16; 16] {
        const SEL: [u16; 8] = [S0, S1, S2, S3, S4, S5, S6, S7];
        let mut ctrl = [0u16; 16];
        for slot in ctrl.iter_mut() {
            let r = rng.next();
            *slot = if r & 7 == 0 { XX } else { SEL[(r >> 8) as usize % 8] };
        }
        ctrl
    }

    #[test]
    fn c16_bitwise_and_layout() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..500 {
            let a: [u16; 16] = core::array::from_fn(|_| rng.set9());
            let b: [u16; 16] = core::array::from_fn(|_| rng.set9());
            let (va, vb) = (make16(&a), make16(&b));
            unsafe {
                assert_eq!(read16(va), a, "load/read round trip");
                assert_eq!(read16(va.and(vb)), core::array::from_fn(|i| a[i] & b[i]));
                assert_eq!(read16(va.or(vb)), core::array::from_fn(|i| a[i] | b[i]));
                assert_eq!(read16(va.xor(vb)), core::array::from_fn(|i| a[i] ^ b[i]));
                assert_eq!(read16(va.and_not(vb)), core::array::from_fn(|i| a[i] & !b[i]));
                assert_eq!(
                    read16(va.which_equal(vb)),
                    core::array::from_fn(|i| if a[i] == b[i] { 0xffff } else { 0 })
                );
                assert_eq!(
                    read16(va.which_nonzero()),
                    core::array::from_fn(|i| if a[i] != 0 { 0xffff } else { 0 })
                );
                assert_eq!(
                    read16(va.popcounts9()),
                    core::array::from_fn(|i| a[i].count_ones() as u16)
                );
                assert_eq!(va.subset_of(vb), (0..16).all(|i| a[i] & !b[i] == 0));
                assert_eq!(va.any_less_than(vb), (0..16).any(|i| a[i] < b[i]));
            }
        }
    }

    #[test]
    fn c16_lane_movement() {
        let mut rng = Rng(0x243f_6a88_85a3_08d3);
        for _ in 0..500 {
            let a: [u16; 16] = core::array::from_fn(|_| rng.set9());
            let va = make16(&a);
            let ctrl = random_ctrl(&mut rng);
            unsafe {
                assert_eq!(read16(va.shuffle(make16(&ctrl))), shuffle_ref(&a, &ctrl));
                assert_eq!(
                    read16(va.shift_rows_up1()),
                    core::array::from_fn(|i| if i % 4 == 0 { 0 } else { a[i - 1] })
                );
                assert_eq!(
                    read16(va.shift_rows_up2()),
                    core::array::from_fn(|i| if i % 4 < 2 { 0 } else { a[i - 2] })
                );
                assert_eq!(
                    read16(va.row_peers()),
                    core::array::from_fn(|i| {
                        let (r, c) = (i / 4, i % 4);
                        (0..4).filter(|k| *k != c).fold(0, |u, k| u | a[r * 4 + k])
                    })
                );
                assert_eq!(
                    read16(va.col_peers()),
                    core::array::from_fn(|i| {
                        let (r, c) = (i / 4, i % 4);
                        (0..4).filter(|k| *k != r).fold(0, |u, k| u | a[k * 4 + c])
                    })
                );
                // Its one caller always hands it a vector already masked to
                // the nine cell lanes, and both backends rely on that, so the
                // model masks too.
                let cellsonly: [u16; 16] = core::array::from_fn(|i| {
                    if i / 4 < 3 && i % 4 < 3 { a[i] } else { 0 }
                });
                let col = |c: usize| (0..4).fold(0, |u, r| u | cellsonly[r * 4 + c]);
                assert_eq!(
                    read16(make16(&cellsonly).box_and_column_unions()),
                    core::array::from_fn(|i| match (i / 4, i % 4) {
                        (3, c) => col(c),
                        (_, 3) => 0,
                        _ => (0..3).fold(0, |u, k| u | col(k)),
                    })
                );
                assert_eq!(read16(C16::from_parts(va.get_lo(), va.get_hi())), a);
                assert_eq!(read8(va.get_lo()), a[..8]);
                assert_eq!(read8(va.get_hi()), a[8..]);
            }
        }
    }

    #[test]
    fn c16_broadcasts() {
        let mut rng = Rng(0xb504_f333_f9de_6484);
        unsafe {
            assert_eq!(read16(C16::all(0x1ff)), [0x1ff; 16]);
            for _ in 0..200 {
                let v = rng.next();
                assert_eq!(
                    read16(C16::splat_u64(v)),
                    core::array::from_fn(|i| (v >> (16 * (i % 4))) as u16)
                );
            }
        }
    }

    #[test]
    fn c8_operations() {
        let mut rng = Rng(0x1357_9bdf_2468_ace0);
        for _ in 0..500 {
            let a: [u16; 8] = core::array::from_fn(|_| rng.set9());
            let b: [u16; 8] = core::array::from_fn(|_| rng.set9());
            let (va, vb) = (make8(&a), make8(&b));
            let ctrl = random_ctrl(&mut rng);
            let ctrl8: [u16; 8] = ctrl[..8].try_into().unwrap();
            let mut wide = [0u16; 16];
            wide[..8].copy_from_slice(&a);
            let mut wide_ctrl = [0u16; 16];
            wide_ctrl[..8].copy_from_slice(&ctrl8);
            unsafe {
                assert_eq!(read8(va), a);
                assert_eq!(read8(C8::zero()), [0; 8]);
                assert_eq!(read8(C8::all(0x1ff)), [0x1ff; 8]);
                assert_eq!(read8(va.and(vb)), core::array::from_fn(|i| a[i] & b[i]));
                assert_eq!(read8(va.or(vb)), core::array::from_fn(|i| a[i] | b[i]));
                assert_eq!(read8(va.xor(vb)), core::array::from_fn(|i| a[i] ^ b[i]));
                assert_eq!(read8(va.and_not(vb)), core::array::from_fn(|i| a[i] & !b[i]));
                assert_eq!(
                    read8(va.shuffle(make8(&ctrl8))),
                    shuffle_ref(&wide, &wide_ctrl)[..8]
                );
                assert_eq!(read8(va.rotate_cols()), core::array::from_fn(|i| a[(i + 4) % 8]));
                assert_eq!(va.all_zero(), a.iter().all(|x| *x == 0));
                assert_eq!(va.intersects(vb), (0..8).any(|i| a[i] & b[i] != 0));
                assert_eq!(
                    read8(va.low_bit_per_lane()),
                    core::array::from_fn(|i| a[i] & a[i].wrapping_neg())
                );
            }
        }
    }

    #[test]
    fn c8_clear_low_bit() {
        let mut rng = Rng(0xcafe_f00d_dead_beef);
        let as_int = |a: &[u16; 8]| {
            a.iter()
                .enumerate()
                .fold(0u128, |acc, (i, x)| acc | ((*x as u128) << (16 * i)))
        };
        for _ in 0..2000 {
            // The caller never passes an empty vector, but sparse ones whose
            // low 64 bits are zero are the interesting case: that is where
            // the 128-bit borrow has to happen.
            let mut a = [0u16; 8];
            let lane = (rng.next() % 8) as usize;
            a[lane] = 1 << (rng.next() % 9);
            for slot in a.iter_mut().skip(lane + 1) {
                if rng.next() & 1 == 0 {
                    *slot = rng.set9();
                }
            }
            let x = as_int(&a);
            let got = read8(unsafe { make8(&a).clear_low_bit() });
            assert_eq!(as_int(&got), x & (x - 1), "a = {a:?}");
        }
    }

    #[test]
    fn band_counts_pack() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..500 {
            let bands: [[u16; 8]; 6] =
                core::array::from_fn(|_| core::array::from_fn(|_| rng.set9()));
            let got = read8(unsafe { band_config_counts(bands.map(|b| make8(&b))) });
            for (i, band) in bands.iter().enumerate() {
                let total: u32 = band.iter().map(|x| x.count_ones()).sum();
                assert_eq!(got[i], total as u16, "band {i} of {bands:?}");
            }
            assert_eq!(&got[6..], &[0xffff, 0xffff], "sentinel lanes");
        }
    }

    /// `minpos_after_sub`'s contract, which is all `choose_band_and_value`
    /// reads: the lane in bits 16..19, and a value in bits 0..16 that is the
    /// adjusted minimum when that is below 256, and something >= 256 when
    /// every lane underflowed the floor. The two backends reach the latter
    /// differently -- x86 by wrapping the subtraction, NEON by saturating
    /// the packed shift -- so only the >= 256 verdict is contracted there,
    /// not the value, and the caller only tests that verdict.
    #[test]
    fn c8_minpos() {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        for _ in 0..5000 {
            let floor = 10u16;
            // The caller's alphabet: a band's total configuration count,
            // which is 9 exactly when the band is fixed, and 0xffff padding.
            let a: [u16; 8] = core::array::from_fn(|_| match rng.next() % 8 {
                0 => 0xffff,
                1 => 9,
                _ => 10 + (rng.next() % 45) as u16,
            });
            let got = unsafe { make8(&a).minpos_after_sub(floor) };
            let live = a
                .iter()
                .enumerate()
                .filter(|(_, x)| **x >= floor && **x != 0xffff)
                .map(|(i, x)| (i, x - floor))
                .min_by_key(|(i, v)| (*v, *i));
            match live {
                None => assert_ne!(got & 0xff00, 0, "a = {a:?}"),
                Some((lane, value)) => {
                    assert_eq!(got & 0xffff, value as u32, "a = {a:?}");
                    assert_eq!(got >> 16, lane as u32, "a = {a:?}");
                }
            }
        }
    }
}
