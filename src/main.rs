use std::io::Read;
use std::time::Instant;

use fastdoku::*;

fn read_puzzles(path: &str) -> Vec<[u8; 81]> {
    let mut text = String::new();
    if path == "-" {
        std::io::stdin().read_to_string(&mut text).expect("read stdin");
    } else {
        text = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        });
    }
    let mut puzzles = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_line(line) {
            Some(g) => puzzles.push(g),
            None => eprintln!("warning: skipping unparseable line {}", ln + 1),
        }
    }
    puzzles
}

/// Per-puzzle time, in ns below 1 us and us above.
fn fmt_per(secs: f64) -> String {
    let ns = secs * 1e9;
    if ns < 1000.0 {
        format!("{ns:.1} ns")
    } else {
        format!("{:.3} us", ns / 1000.0)
    }
}

fn fmt_rate(pps: f64) -> String {
    if pps >= 1e6 {
        format!("{:.2}M/s", pps / 1e6)
    } else if pps >= 1e3 {
        format!("{:.1}K/s", pps / 1e3)
    } else {
        format!("{pps:.0}/s")
    }
}

fn checksum(sol: &[u8; 81]) -> u64 {
    // Cheap fold to keep the optimizer honest and detect divergence.
    sol.iter().fold(0u64, |a, &d| a.wrapping_mul(31).wrapping_add(d as u64))
}

type SolveFn = fn(&[u8; 81]) -> Option<[u8; 81]>;

/// Default engine: route each puzzle between the scalar `jcz` engine (the
/// cheapest per easy deduction) and the `triad` engine (the strongest
/// inference). jcz propagates to its fixpoint — useful work regardless of
/// which engine finishes — and defers at its first guess point if the puzzle
/// is still far from solved, or later if its guess budget trips. This
/// dispatch lives in the binary rather than the library: compiled next to
/// the triad hot path it perturbs that engine's LTO codegen by ~12% on the
/// hard corpora.
fn auto_solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    #[cfg(triad_engine)]
    {
        match fastdoku::jcz::run(
            clues,
            1,
            fastdoku::HYBRID_MAX_UNSOLVED,
            fastdoku::HYBRID_GUESS_BUDGET,
        ) {
            fastdoku::jcz::Outcome::Done(n, sol) => {
                if n > 0 {
                    sol
                } else {
                    None
                }
            }
            fastdoku::jcz::Outcome::Deferred => triad_solve_grid(clues),
        }
    }
    #[cfg(not(triad_engine))]
    {
        jcz_solve_grid(clues)
    }
}

/// `auto` for solution counting; see `auto_solve_grid`.
fn auto_count_solutions(clues: &[u8; 81], limit: u64) -> u64 {
    #[cfg(triad_engine)]
    {
        match fastdoku::jcz::run(
            clues,
            limit,
            fastdoku::HYBRID_MAX_UNSOLVED,
            fastdoku::HYBRID_GUESS_BUDGET,
        ) {
            fastdoku::jcz::Outcome::Done(n, _) => n,
            fastdoku::jcz::Outcome::Deferred => triad_count_solutions(clues, limit),
        }
    }
    #[cfg(not(triad_engine))]
    {
        jcz_count_solutions(clues, limit)
    }
}

fn solve_batch(puzzles: &[[u8; 81]], f: SolveFn) -> (u64, usize) {
    let mut sum = 0u64;
    let mut solved = 0usize;
    for p in puzzles {
        if let Some(s) = f(p) {
            sum = sum.wrapping_add(checksum(&s));
            solved += 1;
        }
    }
    (sum, solved)
}

fn solve_batch_parallel(puzzles: &[[u8; 81]], threads: usize, f: SolveFn) -> (u64, usize) {
    let chunk = puzzles.len().div_ceil(threads);
    let results: Vec<(u64, usize)> = std::thread::scope(|scope| {
        let handles: Vec<_> = puzzles
            .chunks(chunk)
            .map(|c| scope.spawn(move || solve_batch(c, f)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    results
        .into_iter()
        .fold((0u64, 0usize), |(s, n), (s2, n2)| (s.wrapping_add(s2), n + n2))
}

fn cmd_solve(file: &str) {
    let puzzles = read_puzzles(file);
    let t = Instant::now();
    for p in &puzzles {
        match auto_solve_grid(p) {
            Some(s) => println!("{}", grid_to_line(&s)),
            None => println!("NO SOLUTION"),
        }
    }
    eprintln!(
        "{} puzzles in {:.3} ms",
        puzzles.len(),
        t.elapsed().as_secs_f64() * 1e3
    );
}

fn cmd_check(file: &str) {
    // Classify each puzzle: 0, 1, or 2+ solutions; verify solutions.
    let puzzles = read_puzzles(file);
    let (mut none, mut unique, mut multi) = (0usize, 0usize, 0usize);
    for (i, p) in puzzles.iter().enumerate() {
        match auto_count_solutions(p, 2) {
            0 => {
                none += 1;
                println!("line {}: NO SOLUTION", i + 1);
            }
            1 => {
                let s = auto_solve_grid(p).unwrap();
                assert!(is_valid_solution(&s, p), "line {}: invalid solution!", i + 1);
                unique += 1;
            }
            _ => {
                multi += 1;
                println!("line {}: MULTIPLE SOLUTIONS", i + 1);
            }
        }
    }
    println!("{} puzzles: {unique} unique, {multi} multiple, {none} unsolvable", puzzles.len());
}

fn cmd_bench(file: &str, rounds: usize, threads: usize, engine: &str, limit: usize) {
    // An unrecognised name is an error rather than a silent fall back to
    // `auto`: the timing line prints whatever was asked for, so falling back
    // would label `auto`'s numbers with the missing engine's name.
    let f: SolveFn = match engine {
        "auto" => auto_solve_grid,
        "baseline" => baseline_solve_grid,
        "jcz" => jcz_solve_grid,
        #[cfg(triad_engine)]
        "triad" => triad_solve_grid,
        #[cfg(not(triad_engine))]
        "triad" => {
            eprintln!("engine `triad` is not compiled for this target");
            std::process::exit(2);
        }
        other => {
            eprintln!("unknown engine `{other}`: expected auto, jcz, triad or baseline");
            std::process::exit(2);
        }
    };
    let mut puzzles = read_puzzles(file);
    puzzles.truncate(limit);
    if puzzles.is_empty() {
        eprintln!("no puzzles");
        return;
    }
    // Verification pass (untimed): every solution must be valid.
    let mut solvable = 0usize;
    for p in &puzzles {
        if let Some(s) = f(p) {
            assert!(is_valid_solution(&s, p), "solver produced an invalid solution");
            solvable += 1;
        }
    }
    #[cfg(feature = "stats")]
    {
        fastdoku::GUESSES.store(0, std::sync::atomic::Ordering::Relaxed);
        let _ = solve_batch(&puzzles, f);
        let g = fastdoku::GUESSES.load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "  stats: {} guesses total, {:.2} guesses/puzzle",
            g,
            g as f64 / puzzles.len() as f64
        );
    }
    // Warmup.
    let _ = if threads > 1 {
        solve_batch_parallel(&puzzles, threads, f)
    } else {
        solve_batch(&puzzles, f)
    };
    let mut best = f64::INFINITY;
    let mut sum_ref = None;
    for _ in 0..rounds {
        let t = Instant::now();
        let (sum, _) = if threads > 1 {
            solve_batch_parallel(&puzzles, threads, f)
        } else {
            solve_batch(&puzzles, f)
        };
        let dt = t.elapsed().as_secs_f64();
        if let Some(r) = sum_ref {
            assert_eq!(r, sum, "nondeterministic results across rounds");
        } else {
            sum_ref = Some(sum);
        }
        if dt < best {
            best = dt;
        }
    }
    let n = puzzles.len() as f64;
    let name = std::path::Path::new(file)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| file.to_string());
    println!(
        "{name:<32} {:>10} {:>10}  [{} puzzles, {solvable} ok, {engine}, {threads}t, best/{rounds}, sum {:016x}]",
        fmt_per(best / n),
        fmt_rate(n / best),
        puzzles.len(),
        sum_ref.unwrap()
    );
}

fn cmd_gen(count: usize, seed: u64) {
    let mut rng = Rng(seed | 1);
    for _ in 0..count {
        let p = generate_puzzle(&mut rng);
        println!("{}", grid_to_line(&p).replace('0', "."));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage:\n  fastdoku solve <file|->\n  fastdoku check <file|->\n  fastdoku bench <file> [--rounds N] [--threads N] [--limit N] [--engine auto|jcz|triad|baseline]\n  fastdoku gen <count> [--seed N]";
    if args.len() < 2 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let flag = |name: &str, default: usize| -> usize {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    match args[1].as_str() {
        "solve" if args.len() >= 3 => cmd_solve(&args[2]),
        "check" if args.len() >= 3 => cmd_check(&args[2]),
        "bench" if args.len() >= 3 => {
            let engine = args
                .iter()
                .position(|a| a == "--engine")
                .and_then(|i| args.get(i + 1))
                .map(|s| s.as_str())
                .unwrap_or("auto")
                .to_string();
            cmd_bench(
                &args[2],
                flag("--rounds", 10),
                flag("--threads", 1),
                &engine,
                flag("--limit", usize::MAX),
            )
        }
        "gen" if args.len() >= 3 => {
            let count = args[2].parse().unwrap_or(10);
            cmd_gen(count, flag("--seed", 0x243F6A8885A308D3) as u64)
        }
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    }
}
