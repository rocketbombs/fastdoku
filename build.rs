//! Emits `cfg(triad_engine)` for targets that have a SIMD backend for the
//! triad engine's 4x4-of-9-bit-sets vector vocabulary (`src/tvec_*.rs`).
//!
//! The predicate is compound — AVX2 on x86-64, NEON on aarch64 — and it is
//! tested in fifteen places across the library and the binary, so it is
//! named once here rather than spelled out at every site and left to drift.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    // `check-cfg` is a no-op on older cargo, which ignores unknown directives.
    println!("cargo::rustc-check-cfg=cfg(triad_engine)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |f: &str| features.split(',').any(|x| x == f);

    // NEON is architectural on aarch64 (and Rust enables it on every aarch64
    // target), so there is no feature to test there.
    let supported = (arch == "x86_64" && has("avx2")) || arch == "aarch64";
    if supported {
        println!("cargo::rustc-cfg=triad_engine");
    }
}
