// Benchmarks rust_sudoku (the sudoku crate by Emerentius, a JCZSolve
// derivative and the strongest solver in this comparison on easy puzzles)
// using the identical protocol to fastdoku's ench and tdoku_bench.cc:
// same parsing, same in-memory vector, same best-of-N, same checksum, and
// the puzzle handed over inside the timed region.
//
// NOT part of this crate: the sudoku crate is AGPL, so it is kept out of
// fastdoku's dependency graph. To run it, build it standalone:
//
//   cargo new --bin rsbench && cd rsbench
//   cargo add sudoku@0.8
//   # copy this file over src/main.rs, then:
//   RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
//   ./target/release/rsbench <puzzle-file> --rounds 3
// Benchmarks the `sudoku` crate (Emerentius, "rust_sudoku" in tdoku's suite,
// AGPL) using the same protocol as fastdoku's `bench` and bench/tdoku_bench.cc:
// same parsing, same in-memory vector, same best-of-N, same checksum.

use std::time::Instant;

fn parse(line: &str) -> Option<[u8; 81]> {
    let mut g = [0u8; 81];
    let mut i = 0;
    for ch in line.bytes() {
        match ch {
            b'1'..=b'9' => {
                if i >= 81 {
                    return None;
                }
                g[i] = ch - b'0';
                i += 1;
            }
            b'0' | b'.' => {
                if i >= 81 {
                    return None;
                }
                i += 1;
            }
            _ => {}
        }
    }
    if i == 81 { Some(g) } else { None }
}

fn checksum(sol: &[u8; 81]) -> u64 {
    sol.iter().fold(0u64, |a, &d| a.wrapping_mul(31).wrapping_add(d as u64))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let flag = |n: &str, d: usize| -> usize {
        args.iter()
            .position(|a| a == n)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let rounds = flag("--rounds", 3);
    let limit = flag("--limit", usize::MAX);

    let text = std::fs::read_to_string(path).expect("read");
    let mut boards = Vec::new();
    for line in text.lines() {
        if boards.len() >= limit {
            break;
        }
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(g) = parse(line) {
            boards.push(g);
        }
    }

    // from_bytes is inside the timed region: fastdoku and tdoku both do their
    // own input handling inside the call being measured, so this keeps the
    // three harnesses equivalent.
    let run = |bs: &[[u8; 81]]| -> (u64, usize) {
        let mut sum = 0u64;
        let mut ok = 0usize;
        let mut buf = [[0u8; 81]; 1];
        for g in bs {
            let b = sudoku::Sudoku::from_bytes(*g).expect("valid");
            if b.solutions_up_to_buffer(&mut buf, 1) == 1 {
                sum = sum.wrapping_add(checksum(&buf[0]));
                ok += 1;
            }
        }
        (sum, ok)
    };

    let (_, ok) = run(&boards); // warmup + solvable count
    let mut best = f64::INFINITY;
    let mut sum = 0;
    for _ in 0..rounds {
        let t = Instant::now();
        let (s, _) = run(&boards);
        sum = s;
        best = best.min(t.elapsed().as_secs_f64());
    }

    let n = boards.len() as f64;
    let per = best / n;
    let pps = n / best;
    let per_s = if per * 1e9 < 1000.0 {
        format!("{:.1} ns", per * 1e9)
    } else {
        format!("{:.3} us", per * 1e6)
    };
    let rate = if pps >= 1e6 {
        format!("{:.2}M/s", pps / 1e6)
    } else {
        format!("{:.1}K/s", pps / 1e3)
    };
    let name = std::path::Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    println!(
        "{name:<32} {per_s:>10} {rate:>10}  [{} puzzles, {ok} ok, rust_sudoku, 1t, best/{rounds}, sum {sum:016x}]",
        boards.len()
    );
}

