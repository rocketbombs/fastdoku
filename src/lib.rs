//! fastdoku — a fast, complete sudoku solver.
//!
//! Three engines, cross-validated against each other in the test suite:
//! - `triad`: a port of tdoku's DPLL + triad + SIMD architecture, over AVX2
//!   or NEON; the strongest inference, fastest on hard puzzles.
//! - `jcz`: an original implementation of the JCZSolve architecture (bands
//!   by digit, locked candidates by table lookup); the cheapest per easy
//!   deduction, fastest on easy and typical puzzles. Scalar, portable.
//! - `baseline`: a plain cell-mask solver. Slow, but short enough to read in
//!   one sitting and written without a line of `unsafe`, which is what makes
//!   it useful as the reference the other two are checked against.
//!
//! The CLI's default `auto` engine routes each puzzle between `jcz` and
//! `triad` using `jcz::run`'s difficulty gate (see that function). Common
//! design notes: search state is small and copied per guess (no undo
//! trail), propagation runs to fixpoint before every branch, and the same
//! search counts solutions to a limit, so uniqueness checking and "solve
//! any valid sudoku" completeness fall out of one code path.

pub mod jcz;

mod clue_scan;

#[cfg(triad_engine)]
mod tvec;

/// Gated on `cfg(triad_engine)`, which `build.rs` sets for targets with a
/// backend for the engine's vector vocabulary: AVX2 on x86-64, NEON on
/// aarch64.
#[cfg(triad_engine)]
pub mod triad;

/// True when the `triad` engine is compiled in.
pub const HAS_TRIAD: bool = cfg!(triad_engine);

pub const ALL: u16 = 0x1FF;

// ---------------------------------------------------------------------------
// Compile-time tables
// ---------------------------------------------------------------------------

/// 27 units (9 rows, 9 cols, 9 boxes), each 9 cell indices.
const UNITS: [[u8; 9]; 27] = build_units();
/// The 20 peers of each cell.
const PEERS: [[u8; 20]; 81] = build_peers();
/// The 3 units (row, col, box) each cell belongs to.
const CELL_UNITS: [[u8; 3]; 81] = build_cell_units();
/// Same three units as a 27-bit mask, for dirty-unit tracking.
const CELL_UNIT_MASK: [u32; 81] = build_cell_unit_masks();

const fn build_cell_unit_masks() -> [u32; 81] {
    let mut t = [0u32; 81];
    let mut c = 0;
    while c < 81 {
        let u = CELL_UNITS[c];
        t[c] = (1 << u[0]) | (1 << u[1]) | (1 << u[2]);
        c += 1;
    }
    t
}

/// The 2 box-line intersections each cell belongs to, as a 54-bit mask.
const CELL_INT_MASK: [u64; 81] = build_cell_int_masks();

const fn build_cell_int_masks() -> [u64; 81] {
    let mut t = [0u64; 81];
    let mut c = 0;
    while c < 81 {
        let r = c / 9;
        let cl = c % 9;
        let b = (r / 3) * 3 + cl / 3;
        t[c] = (1u64 << (b * 3 + r % 3)) | (1u64 << (27 + b * 3 + cl % 3));
        c += 1;
    }
    t
}

const fn build_units() -> [[u8; 9]; 27] {
    let mut u = [[0u8; 9]; 27];
    let mut r = 0;
    while r < 9 {
        let mut c = 0;
        while c < 9 {
            u[r][c] = (r * 9 + c) as u8;
            u[9 + c][r] = (r * 9 + c) as u8;
            c += 1;
        }
        r += 1;
    }
    let mut b = 0;
    while b < 9 {
        let br = (b / 3) * 3;
        let bc = (b % 3) * 3;
        let mut i = 0;
        while i < 9 {
            u[18 + b][i] = ((br + i / 3) * 9 + bc + i % 3) as u8;
            i += 1;
        }
        b += 1;
    }
    u
}

const fn build_cell_units() -> [[u8; 3]; 81] {
    let mut t = [[0u8; 3]; 81];
    let mut c = 0;
    while c < 81 {
        let r = c / 9;
        let col = c % 9;
        t[c] = [r as u8, (9 + col) as u8, (18 + (r / 3) * 3 + col / 3) as u8];
        c += 1;
    }
    t
}

const fn build_peers() -> [[u8; 20]; 81] {
    let mut p = [[0u8; 20]; 81];
    let mut c = 0;
    while c < 81 {
        let r = c / 9;
        let col = c % 9;
        let mut n = 0;
        let mut i = 0;
        while i < 81 {
            if i != c {
                let ir = i / 9;
                let ic = i % 9;
                if ir == r || ic == col || (ir / 3 == r / 3 && ic / 3 == col / 3) {
                    p[c][n] = i as u8;
                    n += 1;
                }
            }
            i += 1;
        }
        c += 1;
    }
    p
}

// Box-line intersections for locked candidates: 27 box/row + 27 box/col.
// Each has 3 intersection cells, 6 remaining box cells, 6 remaining line cells.
const INTERSECTIONS: ([[u8; 3]; 54], [[u8; 6]; 54], [[u8; 6]; 54], [[u8; 2]; 54]) =
    build_intersections();
const INT_CELLS: [[u8; 3]; 54] = INTERSECTIONS.0;
const INT_BOX_REST: [[u8; 6]; 54] = INTERSECTIONS.1;
const INT_LINE_REST: [[u8; 6]; 54] = INTERSECTIONS.2;
/// [box unit index, line unit index] for each intersection.
const INT_UNITS: [[u8; 2]; 54] = INTERSECTIONS.3;

const fn build_intersections() -> ([[u8; 3]; 54], [[u8; 6]; 54], [[u8; 6]; 54], [[u8; 2]; 54]) {
    let mut cells = [[0u8; 3]; 54];
    let mut boxrest = [[0u8; 6]; 54];
    let mut linerest = [[0u8; 6]; 54];
    let mut units = [[0u8; 2]; 54];
    let mut b = 0;
    while b < 9 {
        let br = (b / 3) * 3;
        let bc = (b % 3) * 3;
        let mut i = 0;
        while i < 3 {
            // box/row intersection
            let k = b * 3 + i;
            let row = br + i;
            let mut n = 0;
            while n < 3 {
                cells[k][n] = (row * 9 + bc + n) as u8;
                n += 1;
            }
            let mut nb = 0;
            let mut j = 0;
            while j < 9 {
                let (r, c) = (br + j / 3, bc + j % 3);
                if r != row {
                    boxrest[k][nb] = (r * 9 + c) as u8;
                    nb += 1;
                }
                j += 1;
            }
            let mut nl = 0;
            let mut c = 0;
            while c < 9 {
                if c < bc || c >= bc + 3 {
                    linerest[k][nl] = (row * 9 + c) as u8;
                    nl += 1;
                }
                c += 1;
            }
            units[k] = [(18 + b) as u8, row as u8];

            // box/col intersection
            let k = 27 + b * 3 + i;
            let col = bc + i;
            let mut n = 0;
            while n < 3 {
                cells[k][n] = ((br + n) * 9 + col) as u8;
                n += 1;
            }
            let mut nb = 0;
            let mut j = 0;
            while j < 9 {
                let (r, c) = (br + j / 3, bc + j % 3);
                if c != col {
                    boxrest[k][nb] = (r * 9 + c) as u8;
                    nb += 1;
                }
                j += 1;
            }
            let mut nl = 0;
            let mut r = 0;
            while r < 9 {
                if r < br || r >= br + 3 {
                    linerest[k][nl] = (r * 9 + col) as u8;
                    nl += 1;
                }
                r += 1;
            }
            units[k] = [(18 + b) as u8, (9 + col) as u8];
            i += 1;
        }
        b += 1;
    }
    (cells, boxrest, linerest, units)
}

// ---------------------------------------------------------------------------
// Assignment queue (scratch, lives outside the cloned state)
// ---------------------------------------------------------------------------

struct Queue {
    buf: [u8; 512],
    len: usize,
}

impl Queue {
    #[inline]
    fn new() -> Self {
        Queue { buf: [0; 512], len: 0 }
    }
    #[inline]
    fn push(&mut self, c: u8) {
        self.buf[self.len] = c;
        self.len += 1;
    }
    #[inline]
    fn pop(&mut self) -> Option<u8> {
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
// Solver state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct State {
    /// Candidate digit mask per cell. Solved cells keep exactly their digit bit.
    cand: [u16; 81],
    /// Digits already placed in each unit.
    unit_solved: [u16; 27],
    /// Bitset of unsolved cells.
    unsolved: u128,
    n_unsolved: u32,
    /// Units whose candidate distribution changed since their last scan.
    dirty: u32,
    /// Intersections changed since their last locked-candidates scan.
    int_dirty: u64,
}

impl State {
    #[inline]
    fn new() -> Self {
        State {
            cand: [ALL; 81],
            unit_solved: [0; 27],
            unsolved: (1u128 << 81) - 1,
            n_unsolved: 81,
            dirty: (1 << 27) - 1,
            int_dirty: (1 << 54) - 1,
        }
    }

    #[inline]
    fn is_unsolved(&self, c: usize) -> bool {
        self.unsolved >> c & 1 != 0
    }

    /// Commit cell `c` to its (single-bit) candidate and eliminate from peers.
    #[inline]
    fn assign(&mut self, c: usize, q: &mut Queue) -> bool {
        let bit = self.cand[c];
        debug_assert_eq!(bit.count_ones(), 1);
        self.unsolved &= !(1u128 << c);
        self.n_unsolved -= 1;
        // SAFETY: all indices below come from const tables built over 0..81
        // (cells) and 0..27 (units); `c` is always < 81 at call sites.
        unsafe {
            let [u0, u1, u2] = *CELL_UNITS.get_unchecked(c);
            *self.unit_solved.get_unchecked_mut(u0 as usize) |= bit;
            *self.unit_solved.get_unchecked_mut(u1 as usize) |= bit;
            *self.unit_solved.get_unchecked_mut(u2 as usize) |= bit;
            for &p in PEERS.get_unchecked(c) {
                let p = p as usize;
                let m = *self.cand.get_unchecked(p);
                if m & bit != 0 {
                    // A solved peer holding the same digit is a contradiction.
                    if !self.is_unsolved(p) {
                        return false;
                    }
                    let nm = m & !bit;
                    if nm == 0 {
                        return false;
                    }
                    *self.cand.get_unchecked_mut(p) = nm;
                    self.dirty |= *CELL_UNIT_MASK.get_unchecked(p);
                    self.int_dirty |= *CELL_INT_MASK.get_unchecked(p);
                    if nm & (nm - 1) == 0 {
                        q.push(p as u8); // naked single
                    }
                }
            }
        }
        true
    }

    /// Run naked-single + hidden-single propagation to fixpoint.
    /// On failure the queue is left empty.
    fn propagate(&mut self, q: &mut Queue) -> bool {
        loop {
            while let Some(c) = q.pop() {
                let c = c as usize;
                if !self.is_unsolved(c) {
                    continue;
                }
                if !self.assign(c, q) {
                    q.clear();
                    return false;
                }
            }
            if self.dirty == 0 {
                return true;
            }
            let mut dirty = self.dirty;
            self.dirty = 0;
            while dirty != 0 {
                let u = dirty.trailing_zeros() as usize;
                dirty &= dirty - 1;
                // SAFETY: u < 27 (bit index of a 27-bit mask), cells are < 81.
                let cells = unsafe { UNITS.get_unchecked(u) };
                // `once` = digits placeable somewhere; `more` = in 2+ cells.
                let mut once = 0u16;
                let mut more = 0u16;
                for &c in cells {
                    let m = unsafe { *self.cand.get_unchecked(c as usize) };
                    more |= once & m;
                    once |= m;
                }
                if once != ALL {
                    // Some digit has nowhere to go in this unit.
                    q.clear();
                    return false;
                }
                let mut uniq = once & !more & !self.unit_solved[u];
                while uniq != 0 {
                    let bit = uniq & uniq.wrapping_neg();
                    uniq &= uniq - 1;
                    let mut placed = false;
                    for &c in cells {
                        let ci = c as usize;
                        if self.cand[ci] & bit != 0 {
                            self.cand[ci] = bit;
                            self.dirty |= CELL_UNIT_MASK[ci];
                            self.int_dirty |= CELL_INT_MASK[ci];
                            q.push(c);
                            placed = true;
                            break;
                        }
                    }
                    if !placed {
                        // Two hidden singles collided in one cell.
                        q.clear();
                        return false;
                    }
                }
            }
        }
    }

    /// Remove `bits` from cell `c`'s candidates. Returns false on contradiction.
    #[inline]
    fn elim(&mut self, c: usize, bits: u16, q: &mut Queue) -> bool {
        let m = self.cand[c];
        if m & bits == 0 {
            return true;
        }
        if !self.is_unsolved(c) {
            return false; // a solved cell holds an eliminated digit
        }
        let nm = m & !bits;
        if nm == 0 {
            return false;
        }
        self.cand[c] = nm;
        self.dirty |= CELL_UNIT_MASK[c];
        self.int_dirty |= CELL_INT_MASK[c];
        if nm & (nm - 1) == 0 {
            q.push(c as u8);
        }
        true
    }

    /// Locked candidates (pointing + claiming) over all 54 box-line
    /// intersections. Returns None on contradiction, Some(changed) otherwise.
    fn locked_candidates(&mut self, q: &mut Queue) -> Option<bool> {
        let mut changed = false;
        let mut pend = self.int_dirty;
        self.int_dirty = 0;
        while pend != 0 {
            let k = pend.trailing_zeros() as usize;
            pend &= pend - 1;
            let [bu, lu] = INT_UNITS[k];
            let solved = self.unit_solved[bu as usize] | self.unit_solved[lu as usize];
            let mut inter = 0u16;
            for &c in &INT_CELLS[k] {
                inter |= self.cand[c as usize];
            }
            let inter = inter & !solved;
            if inter == 0 {
                continue;
            }
            let mut boxrest = 0u16;
            for &c in &INT_BOX_REST[k] {
                boxrest |= self.cand[c as usize];
            }
            let mut linerest = 0u16;
            for &c in &INT_LINE_REST[k] {
                linerest |= self.cand[c as usize];
            }
            // Pointing: digit confined to this intersection within the box
            // cannot appear elsewhere in the line.
            let pointing = inter & !boxrest;
            if pointing & linerest != 0 {
                for &c in &INT_LINE_REST[k] {
                    if !self.elim(c as usize, pointing, q) {
                        return None;
                    }
                }
                changed = true;
            }
            // Claiming: digit confined to this intersection within the line
            // cannot appear elsewhere in the box.
            let claiming = inter & !linerest;
            if claiming & boxrest != 0 {
                for &c in &INT_BOX_REST[k] {
                    if !self.elim(c as usize, claiming, q) {
                        return None;
                    }
                }
                changed = true;
            }
        }
        Some(changed)
    }

    /// Full propagation: singles to fixpoint, then locked candidates, repeat.
    /// (Naked pairs were tried here and removed: they cut guesses/puzzle from
    /// 21 to 13 on top95 but the scan cost exceeded the savings at ~0.7us per
    /// search node.)
    fn propagate_all(&mut self, q: &mut Queue) -> bool {
        loop {
            if !self.propagate(q) {
                return false;
            }
            if self.n_unsolved == 0 || self.int_dirty == 0 {
                return true;
            }
            match self.locked_candidates(q) {
                None => {
                    q.clear();
                    return false;
                }
                Some(true) => continue,
                Some(false) => return true,
            }
        }
    }

    /// Most-constrained unsolved cell (fewest candidates, early-out at 2).
    #[inline]
    fn best_cell(&self) -> usize {
        let mut best = 0usize;
        let mut best_n = 10;
        let mut bits = self.unsolved;
        while bits != 0 {
            let c = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let n = self.cand[c].count_ones();
            if n < best_n {
                best_n = n;
                best = c;
                if n == 2 {
                    break;
                }
            }
        }
        best
    }

    #[inline]
    fn write_grid(&self, out: &mut [u8; 81]) {
        for c in 0..81 {
            out[c] = (self.cand[c].trailing_zeros() + 1) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[cfg(feature = "stats")]
pub static GUESSES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

fn search(st: &State, q: &mut Queue, limit: u64, count: &mut u64, out: &mut [u8; 81]) {
    if st.n_unsolved == 0 {
        if *count == 0 {
            st.write_grid(out);
        }
        *count += 1;
        return;
    }
    let c = st.best_cell();
    let mut mask = st.cand[c];
    while mask != 0 {
        let bit = mask & mask.wrapping_neg();
        mask &= mask - 1;
        #[cfg(feature = "stats")]
        GUESSES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let mut s2 = st.clone();
        s2.cand[c] = bit;
        s2.dirty |= CELL_UNIT_MASK[c];
        s2.int_dirty |= CELL_INT_MASK[c];
        q.push(c as u8);
        if s2.propagate_all(q) {
            search(&s2, q, limit, count, out);
            if *count >= limit {
                return;
            }
        }
    }
}

fn load(st: &mut State, q: &mut Queue, clues: &[u8; 81]) -> bool {
    for c in 0..81 {
        let d = clues[c];
        if d != 0 {
            let bit = 1u16 << (d - 1);
            if st.cand[c] & bit == 0 {
                return false;
            }
            st.cand[c] = bit;
            q.push(c as u8);
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Recommended difficulty gate for a jcz -> triad hybrid: a puzzle still
/// this many cells from solved when jcz's propagation stalls belongs to the
/// triad engine. Tuned on the benchmark corpora; see README.
pub const HYBRID_MAX_UNSOLVED: u32 = 50;
/// Recommended guess budget for the hybrid: a safety valve for puzzles that
/// pass the gate but search deep anyway.
pub const HYBRID_GUESS_BUDGET: u32 = 16;

/// Solve a puzzle given as 81 digits (0 = blank). Returns the first solution
/// found, or `None` if the puzzle has no solution.
///
/// Where the `triad` engine is available this is it (a port of tdoku's
/// architecture, over AVX2 or NEON); otherwise the scalar `jcz` engine. The
/// CLI's default `auto` engine instead routes each puzzle between them with
/// [`jcz::run`] — that dispatch lives in the binary because defining it
/// here, in the same LTO unit as the triad hot path, measurably degrades
/// the triad engine's codegen (~12% on the hard corpora).
#[inline]
pub fn solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    #[cfg(triad_engine)]
    {
        triad::solve_grid(clues)
    }
    #[cfg(not(triad_engine))]
    {
        jcz::solve_grid(clues)
    }
}

/// Count solutions up to `limit` (use limit=2 for a uniqueness check).
#[inline]
pub fn count_solutions(clues: &[u8; 81], limit: u64) -> u64 {
    #[cfg(triad_engine)]
    {
        triad::count_solutions(clues, limit)
    }
    #[cfg(not(triad_engine))]
    {
        jcz::count_solutions(clues, limit)
    }
}

/// Scalar JCZSolve-family engine (bands by digit, locked candidates).
#[inline]
pub fn jcz_solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    jcz::solve_grid(clues)
}

#[inline]
pub fn jcz_count_solutions(clues: &[u8; 81], limit: u64) -> u64 {
    jcz::count_solutions(clues, limit)
}

/// DPLL+triad+SIMD engine, ported from tdoku (BSD-2-Clause).
#[cfg(triad_engine)]
#[inline]
pub fn triad_solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    triad::solve_grid(clues)
}

#[cfg(triad_engine)]
#[inline]
pub fn triad_count_solutions(clues: &[u8; 81], limit: u64) -> u64 {
    triad::count_solutions(clues, limit)
}

/// Baseline engine: cell masks and unit scans, no `unsafe` and no tables
/// worth verifying. It is the reference the other two are checked against.
pub fn baseline_solve_grid(clues: &[u8; 81]) -> Option<[u8; 81]> {
    let mut st = State::new();
    let mut q = Queue::new();
    if !load(&mut st, &mut q, clues) || !st.propagate_all(&mut q) {
        return None;
    }
    let mut out = [0u8; 81];
    let mut count = 0;
    search(&st, &mut q, 1, &mut count, &mut out);
    if count > 0 { Some(out) } else { None }
}

/// Baseline engine solution counting.
pub fn baseline_count_solutions(clues: &[u8; 81], limit: u64) -> u64 {
    let mut st = State::new();
    let mut q = Queue::new();
    if !load(&mut st, &mut q, clues) || !st.propagate_all(&mut q) {
        return 0;
    }
    let mut out = [0u8; 81];
    let mut count = 0;
    search(&st, &mut q, limit, &mut count, &mut out);
    count
}

/// Parse an 81-character puzzle line ('.' or '0' = blank; whitespace ignored).
pub fn parse_line(s: &str) -> Option<[u8; 81]> {
    let mut g = [0u8; 81];
    let mut i = 0;
    for ch in s.bytes() {
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
            b' ' | b'\t' | b'\r' => {}
            _ => return None,
        }
    }
    if i == 81 { Some(g) } else { None }
}

pub fn grid_to_line(g: &[u8; 81]) -> String {
    g.iter().map(|&d| (b'0' + d) as char).collect()
}

/// Check that `sol` is a complete valid grid consistent with `clues`.
pub fn is_valid_solution(sol: &[u8; 81], clues: &[u8; 81]) -> bool {
    for c in 0..81 {
        if sol[c] < 1 || sol[c] > 9 {
            return false;
        }
        if clues[c] != 0 && clues[c] != sol[c] {
            return false;
        }
    }
    for unit in &UNITS {
        let mut seen = 0u16;
        for &c in unit {
            seen |= 1 << (sol[c as usize] - 1);
        }
        if seen != ALL {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Puzzle generation (for benchmarking on average-case puzzles)
// ---------------------------------------------------------------------------

/// xorshift64 — deterministic, dependency-free.
pub struct Rng(pub u64);

impl Rng {
    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline]
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn fill_search(st: &State, q: &mut Queue, rng: &mut Rng, out: &mut Option<[u8; 81]>) {
    if st.n_unsolved == 0 {
        let mut g = [0u8; 81];
        st.write_grid(&mut g);
        *out = Some(g);
        return;
    }
    let c = st.best_cell();
    let mut bits = [0u16; 9];
    let mut n = 0;
    let mut mask = st.cand[c];
    while mask != 0 {
        bits[n] = mask & mask.wrapping_neg();
        mask &= mask - 1;
        n += 1;
    }
    // Fisher-Yates shuffle of candidate order.
    for i in (1..n).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        bits.swap(i, j);
    }
    for &bit in &bits[..n] {
        let mut s2 = st.clone();
        s2.cand[c] = bit;
        s2.dirty |= CELL_UNIT_MASK[c];
        s2.int_dirty |= CELL_INT_MASK[c];
        q.push(c as u8);
        if s2.propagate_all(q) {
            fill_search(&s2, q, rng, out);
            if out.is_some() {
                return;
            }
        }
    }
}

/// Produce a random completed grid.
pub fn random_solved_grid(rng: &mut Rng) -> [u8; 81] {
    loop {
        let st = State::new();
        let mut q = Queue::new();
        let mut out = None;
        fill_search(&st, &mut q, rng, &mut out);
        if let Some(g) = out {
            return g;
        }
    }
}

/// Generate a random minimal puzzle with a unique solution.
pub fn generate_puzzle(rng: &mut Rng) -> [u8; 81] {
    let sol = random_solved_grid(rng);
    let mut clues = sol;
    let mut order: [u8; 81] = core::array::from_fn(|i| i as u8);
    for i in (1..81usize).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        order.swap(i, j);
    }
    for &c in &order {
        let c = c as usize;
        let d = clues[c];
        clues[c] = 0;
        if count_solutions(&clues, 2) != 1 {
            clues[c] = d;
        }
    }
    clues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_grid_has_many_solutions() {
        let empty = [0u8; 81];
        assert_eq!(count_solutions(&empty, 2), 2);
    }

    #[test]
    fn solves_easy() {
        let p = parse_line(
            "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
        )
        .unwrap();
        let sol = solve_grid(&p).unwrap();
        assert!(is_valid_solution(&sol, &p));
    }

    #[test]
    fn detects_contradiction() {
        // Two 5s in the first row.
        let mut p = [0u8; 81];
        p[0] = 5;
        p[1] = 5;
        assert_eq!(count_solutions(&p, 2), 0);
        assert!(solve_grid(&p).is_none());
    }

    /// The three engines are independent implementations; on every puzzle
    /// they must agree on the solution count, and any solution any of them
    /// returns must be a valid completion of the clues.
    fn cross_check(p: &[u8; 81]) {
        let c_base = baseline_count_solutions(p, 2);
        assert_eq!(
            c_base,
            jcz_count_solutions(p, 2),
            "jcz count differs on {}",
            grid_to_line(p)
        );
        #[cfg(triad_engine)]
        assert_eq!(
            c_base,
            triad_count_solutions(p, 2),
            "triad count differs on {}",
            grid_to_line(p)
        );
        if c_base >= 1 {
            let check = |name: &str, sol: Option<[u8; 81]>| {
                let s = sol.unwrap_or_else(|| panic!("{name} found no solution"));
                assert!(is_valid_solution(&s, p), "{name} returned an invalid grid");
            };
            #[cfg(triad_engine)]
            check("triad", triad_solve_grid(p));
            check("baseline", baseline_solve_grid(p));
            check("jcz", jcz_solve_grid(p));
        } else {
            assert!(solve_grid(p).is_none());
            assert!(jcz_solve_grid(p).is_none());
        }
    }

    #[test]
    fn engines_agree_on_random_puzzles() {
        let mut rng = Rng(0x5DEECE66D_u64 ^ 0x1234_5678);
        for iter in 0..400u64 {
            let sol = random_solved_grid(&mut rng);
            let mut p = sol;
            let removals = 25 + (rng.next() % 50) as usize;
            for _ in 0..removals {
                p[(rng.next() % 81) as usize] = 0;
            }
            if iter % 3 == 2 {
                let c = (rng.next() % 81) as usize;
                if p[c] != 0 {
                    p[c] = (rng.next() % 9) as u8 + 1;
                }
            }
            cross_check(&p);
        }
    }

    #[test]
    fn engines_agree_on_minimal_puzzles() {
        let mut rng = Rng(0xDEADBEEFCAFEBABE);
        for _ in 0..25 {
            let p = generate_puzzle(&mut rng);
            cross_check(&p);
            assert_eq!(count_solutions(&p, 2), 1);
        }
    }

    #[test]
    fn default_engine_matches_baseline_on_random_puzzles() {
        let mut rng = Rng(0x1234ABCD9876EF01);
        for iter in 0..300u64 {
            let sol = random_solved_grid(&mut rng);
            let mut p = sol;
            let removals = 30 + (rng.next() % 40) as usize;
            for _ in 0..removals {
                p[(rng.next() % 81) as usize] = 0;
            }
            if iter % 3 == 2 {
                // corrupt one clue to exercise contradiction paths
                let c = (rng.next() % 81) as usize;
                if p[c] != 0 {
                    p[c] = (rng.next() % 9) as u8 + 1;
                }
            }
            let cb = baseline_count_solutions(&p, 2);
            let cn = count_solutions(&p, 2);
            assert_eq!(cb, cn, "solution count mismatch on {}", grid_to_line(&p));
            if cn >= 1 {
                let sn = solve_grid(&p).expect("default engine found no solution");
                assert!(is_valid_solution(&sn, &p));
                if cn == 1 {
                    let sb = baseline_solve_grid(&p).unwrap();
                    assert_eq!(sb, sn);
                }
            } else {
                assert!(solve_grid(&p).is_none());
            }
        }
    }

    #[test]
    fn generator_produces_unique_puzzles() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..5 {
            let p = generate_puzzle(&mut rng);
            assert_eq!(count_solutions(&p, 2), 1);
            let sol = solve_grid(&p).unwrap();
            assert!(is_valid_solution(&sol, &p));
        }
    }
}
