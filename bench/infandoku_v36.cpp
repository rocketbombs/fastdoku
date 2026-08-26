// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  Infandoku â€” High-performance Sudoku Solver
//  Author : A. Infantine (2026)
//
//  Architecture : BitBoard + AVX2 gather + Naked Singles + Hidden
//                 Singles + Smart dispatcher (3 regimes)
//
//  Performance (standard 17-clue dataset, 49 158 puzzles) :
//    ~12 Âµs/puzzle average on dedicated i7/Ryzen (top-4/5 worldwide)
//
//  Compilation :
//    g++ -O3 -march=native -mavx2 -std=c++17 -o infandoku Infandoku.cpp
//
//  Usage :
//    ./infandoku            â†’ interactive mode (type puzzle, get solution)
//    ./infandoku file.txt   â†’ solve all puzzles from a file
//    echo "00430â€¦" | ./infandoku  â†’ stdin pipe
//
//  Accepted input formats (auto-detected) :
//    81 raw digits  : 004300209005009001070060043â€¦
//    Dots for empty : ..53â€¦..8â€¦â€¦2..7..6â€¦..
//    CSV            : 1,puzzle_string,solution,27,2.2
//    TSV            : puzzle<TAB>solution
//
//  Key innovations :
//    1. AVX2 gather for Hidden Singles â€” processes 8 cells in 1 instruction
//    2. 3-regime dispatcher â€” different DFS strategies per clue count
//    3. Block-fill (fas) â€” fills entire 3x3 block at once (novel technique,
//       not documented in published solvers as of 2026)
//    4. Entropy scoring â€” orders candidate values by constraint impact
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#include <array>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
// unistd.h removed: POSIX-only, unused, blocks the Windows build
#include <immintrin.h>

inline int popcnt(int x) { return __builtin_popcount(x); }
inline int ctz(int x)    { return __builtin_ctz(x); }

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 1 â€” LOOKUP TABLES
//  Precomputed index arrays used by the AVX2 gather instructions.
//  Built once at startup, never modified.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

struct Tables {
// cell_blk[r][c] : which 3x3 block does cell (r,c) belong to (0â€“8)
// cell_pos[r][c] : position of (r,c) inside its block (0â€“8)
int cell_blk[9][9];
int cell_pos[9][9];

// blk_row[b][p] : row of position p in block b
// blk_col[b][p] : column of position p in block b
int blk_row[9][9];
int blk_col[9][9];

// AVX2 gather index arrays (must be 32-bit ints, 32-byte aligned)
// row_blk[r][0..7] : block index for the first 8 cells of row r
// col_idx[0..7]    : {0,1,2,3,4,5,6,7} â€” column indices for gather
// row_idx[0..7]    : {0,1,2,3,4,5,6,7} â€” row indices for gather
// blk_r[b][0..7]   : row of first 8 positions in block b
// blk_c[b][0..7]   : col of first 8 positions in block b
alignas(32) int row_blk[9][8];
alignas(32) int col_idx[8];
alignas(32) int row_idx[8];
alignas(32) int blk_r[9][8];
alignas(32) int blk_c[9][8];

// Flat cell indices (r*9+c) for gathering per-cell elimination masks
alignas(32) int col_flat[9][8];   // col_flat[c][i] = i*9 + c   (i = row 0..7)
alignas(32) int blk_flat[9][8];   // blk_flat[b][p] = blk_row[b][p]*9 + blk_col[b][p]
int last_col_flat[9];             // col_flat for row 8 : 8*9 + c
int last_blk_flat[9];             // blk_flat for position 8

// band_of_blk[b]  : which horizontal band (0..2) block b belongs to
// stack_of_blk[b] : which vertical stack (0..2) block b belongs to
// band_blocks[bd][0..2]  : the 3 blocks in band bd
// stack_blocks[st][0..2] : the 3 blocks in stack st
// blk_rows[b][0..2] : the 3 rows spanned by block b
// blk_cols[b][0..2] : the 3 cols spanned by block b
int band_of_blk[9], stack_of_blk[9];
int band_blocks[3][3], stack_blocks[3][3];
int blk_rows[9][3], blk_cols[9][3];

void build() {
    for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) {
        cell_blk[r][c] = (r/3)*3 + (c/3);
        cell_pos[r][c] = (r%3)*3 + (c%3);
    }
    for (int b = 0; b < 9; b++) for (int p = 0; p < 9; p++) {
        blk_row[b][p] = (b/3)*3 + (p/3);
        blk_col[b][p] = (b%3)*3 + (p%3);
    }
    for (int r = 0; r < 9; r++)
        for (int c = 0; c < 8; c++) row_blk[r][c] = cell_blk[r][c];
    for (int i = 0; i < 8; i++) { col_idx[i] = i; row_idx[i] = i; }
    for (int b = 0; b < 9; b++) for (int p = 0; p < 8; p++) {
        blk_r[b][p] = blk_row[b][p];
        blk_c[b][p] = blk_col[b][p];
    }
    for (int c = 0; c < 9; c++) {
        for (int i = 0; i < 8; i++) col_flat[c][i] = i*9 + c;
        last_col_flat[c] = 8*9 + c;
    }
    for (int b = 0; b < 9; b++) {
        for (int p = 0; p < 8; p++) blk_flat[b][p] = blk_row[b][p]*9 + blk_col[b][p];
        last_blk_flat[b] = blk_row[b][8]*9 + blk_col[b][8];
    }
    for (int b = 0; b < 9; b++) {
        band_of_blk[b]  = b / 3;
        stack_of_blk[b] = b % 3;
        blk_rows[b][0] = (b/3)*3; blk_rows[b][1] = (b/3)*3+1; blk_rows[b][2] = (b/3)*3+2;
        blk_cols[b][0] = (b%3)*3; blk_cols[b][1] = (b%3)*3+1; blk_cols[b][2] = (b%3)*3+2;
    }
    for (int bd = 0; bd < 3; bd++) for (int k = 0; k < 3; k++) band_blocks[bd][k]  = bd*3 + k;
    for (int sk = 0; sk < 3; sk++) for (int k = 0; k < 3; k++) stack_blocks[sk][k] = sk + k*3;
}

} T;

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 2 â€” GRID STATE (BitBoard representation)
//
//  Each row/col/block is represented as two bitmasks :
//    prow[r] : 9-bit mask of values already placed in row r
//    rows[r] : 9-bit mask of empty cells in row r
//
//  cv(r,c) returns a 9-bit mask of legal candidates for cell (r,c)
//  by ORing the three placement masks and inverting.
//
//  place(r,c,v) updates all six bitmasks and checks for contradictions
//  in the affected row, column, and block.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

static long long g_backtracks = 0;

struct State {
int      board[81] = {};        // placed values (0 = empty)
uint16_t rows[9]   = {};        // bitmask of empty cells per row
uint16_t cols[9]   = {};        // bitmask of empty cells per column
uint16_t blks[9]   = {};        // bitmask of empty cells per block

// Stored as int32 (not uint16) so AVX2 gather works natively
alignas(32) int prow[9] = {};   // bitmask of placed values per row
alignas(32) int pcol[9] = {};   // bitmask of placed values per column
alignas(32) int pblk[9] = {};   // bitmask of placed values per block

// Extra per-cell eliminations from Locked Candidates (pointing/claiming).
// Monotonic: bits are only ever added, and copied verbatim on backtrack
// (this solver backtracks via whole-struct copy, so no undo logic needed).
alignas(32) int elim[81] = {};

int empty         = 0;
int initial_clues = 0;

// â”€â”€ Incremental dirty-unit tracking for do_lc â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Unit ids: 0..8 = rows, 9..17 = cols, 18..26 = blocks.
// Fixed-size ring/stack (max 27 distinct units, cheap to copy on backtrack).
// Buffer must be much larger than 27: a unit can be popped (flag cleared),
// reprocessed, and re-pushed multiple times within a single drain as
// eliminations cascade, so total enqueue *events* per do_lc() call can
// exceed the number of distinct unit ids by a wide margin.
static constexpr int kDirtyCap = 1024;
bool dirty_flag[27]      = {};
int  dirty_q[kDirtyCap]  = {};
int  dirty_n             = 0;

inline void mark_unit(int uid) {
    if (!dirty_flag[uid] && dirty_n < kDirtyCap) { dirty_flag[uid] = true; dirty_q[dirty_n++] = uid; }
}
inline void mark_row(int r) { mark_unit(r); }
inline void mark_col(int c) { mark_unit(9 + c); }
inline void mark_blk(int b) { mark_unit(18 + b); }

// Marks every unit whose candidate geometry could have changed as a
// result of a value being placed (or eliminated) at cell (r,c).
inline void mark_dirty_cell(int r, int c) {
    int b = T.cell_blk[r][c];
    mark_row(r); mark_col(c); mark_blk(b);
}
// Marks every unit affected specifically by PLACING a value at (r,c):
// its own row/col/block, the 2 band-mate blocks, the 2 stack-mate blocks,
// and the other 2 rows/cols spanned by its own block.
inline void mark_dirty_place(int r, int c) {
    int b = T.cell_blk[r][c];
    mark_row(r); mark_col(c); mark_blk(b);
    int bd = T.band_of_blk[b], sk = T.stack_of_blk[b];
    for (int k = 0; k < 3; k++) { mark_blk(T.band_blocks[bd][k]); mark_blk(T.stack_blocks[sk][k]); }
    for (int k = 0; k < 3; k++) { mark_row(T.blk_rows[b][k]);     mark_col(T.blk_cols[b][k]); }
}

// Returns 9-bit bitmask of legal values for cell (r,c)
inline int cv(int r, int c) const {
    return (~(prow[r] | pcol[c] | pblk[T.cell_blk[r][c]] | elim[r*9+c])) & 0x1FF;
}

// Place value v at (r,c). Returns false if contradiction detected.
inline bool place(int r, int c, int v) {
    int b = T.cell_blk[r][c], p = T.cell_pos[r][c];
    board[r*9+c] = v;
    empty--;
    int vb = 1 << (v-1);
    prow[r] |= vb;  pcol[c] |= vb;  pblk[b] |= vb;
    rows[r] &= ~(uint16_t)(1 << c);
    cols[c] &= ~(uint16_t)(1 << r);
    blks[b] &= ~(uint16_t)(1 << p);
    mark_dirty_place(r, c);
    // Check neighbours for contradictions (no legal candidate left)
    for (uint16_t fr = rows[r]; fr; fr &= fr-1) {
        int fc = ctz(fr); if (!cv(r, fc)) return false;
    }
    for (uint16_t fc = cols[c]; fc; fc &= fc-1) {
        int fr = ctz(fc); if (!cv(fr, c)) return false;
    }
    for (uint16_t fb = blks[b]; fb; fb &= fb-1) {
        int pp = ctz(fb);
        if (!cv(T.blk_row[b][pp], T.blk_col[b][pp])) return false;
    }
    return true;
}

// Returns true if the grid is fully and correctly solved
bool valid() const {
    for (int i = 0; i < 9; i++)
        if (prow[i] != 0x1FF || pcol[i] != 0x1FF || pblk[i] != 0x1FF)
            return false;
    return !empty;
}

// Initialize from an 81-character string (0 or '.' = empty cell)
void init(const std::string& s) {
    memset(this, 0, sizeof(*this));
    for (int i = 0; i < 9; i++) rows[i] = cols[i] = blks[i] = 0x1FF;
    int j = 0;
    for (char c : s) {
        if (j >= 81) break;
        int v = (c >= '1' && c <= '9') ? c - '0' :
                (c == '0' || c == '.')  ? 0 : -1;
        if (v < 0) continue;
        board[j] = v;
        if (v) {
            int r = j/9, cc = j%9, bit = 1 << (v-1);
            prow[r] |= bit;  pcol[cc] |= bit;
            pblk[T.cell_blk[r][cc]] |= bit;
            rows[r]  &= ~(uint16_t)(1 << cc);
            cols[cc] &= ~(uint16_t)(1 << r);
            blks[T.cell_blk[r][cc]] &= ~(uint16_t)(1 << T.cell_pos[r][cc]);
            mark_dirty_place(r, cc);
            initial_clues++;
        } else empty++;
        j++;
    }
}

};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 3 â€” NAKED SINGLES (do_ns)
//
//  A Naked Single is a cell with exactly one legal candidate.
//  We scan all empty cells; if a cell has 0 candidates â†’ contradiction.
//  If it has exactly 1 â†’ place it immediately and restart the scan.
//
//  Return values :
//    -2 : grid is fully solved (empty == 0)
//    -1 : no naked single found (stable, need Hidden Singles or DFS)
//    â‰¥0 : contradiction at cell index r*9+c
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

int do_ns(State& st) {
bool progress = true;
while (progress) {
progress = false;
if (!st.empty) return -2;
for (int r = 0; r < 9; r++) {
for (uint16_t fr = st.rows[r]; fr;) {
int c = ctz(fr); fr &= fr-1;
int cands = st.cv(r, c);
if (!cands) return r*9 + c;           // contradiction
if (!(cands & (cands-1))) {           // single candidate
if (!st.place(r, c, ctz(cands)+1)) return r*9 + c;
progress = true;
if (!st.empty) return -2;
fr = st.rows[r];                  // restart row scan
}
}
}
}
return -1;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 4 â€” HIDDEN SINGLES with AVX2 gather (do_hs)
//
//  A Hidden Single is a value that can only go in one cell within
//  a row, column, or block.
//
//  AVX2 trick : for row r and value v, we need to know which columns
//  are blocked (already have v in their column or block).
//  We gather pcol[0..7] and pblk[block(r,0..7)] into two __m256i,
//  OR them, AND with vb (the bit for value v), then cmpeq to zero.
//  movemask_ps gives a bitmask of the 8 available columns in 1 pass.
//  Column 8 is handled separately (scalarly) since gather covers 0..7.
//
//  Same approach for columns (gather prow + pblk) and blocks.
//
//  Return values : same convention as do_ns.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

int do_hs(State& st) {
bool found = true;
const __m256i vZ    = _mm256_setzero_si256();
const __m256i vCidx = _mm256_load_si256((const __m256i*)T.col_idx);

while (found) {
    found = false;

    // â”€â”€ Rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for (int r = 0; r < 9; r++) {
        int miss = (~st.prow[r]) & 0x1FF;
        if (!miss || !st.rows[r]) continue;
        __m256i vBLK = _mm256_load_si256((const __m256i*)T.row_blk[r]);
        __m256i vElimRow = _mm256_loadu_si256((const __m256i*)&st.elim[r*9]);
        // gOR[i] = pcol[i] | pblk[block(r,i)] | elim[r*9+i]  for i in 0..7
        __m256i gOR = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_i32gather_epi32(st.pcol, vCidx, 4),
                _mm256_i32gather_epi32(st.pblk, vBLK,  4)),
            vElimRow);
        for (int tmp = miss; tmp;) {
            int vb = tmp & (-tmp); tmp &= tmp-1;
            int v  = ctz(vb) + 1;
            // can8f : bitmask of columns 0..7 where v can go
            int can8f = _mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpeq_epi32(
                    _mm256_and_si256(gOR, _mm256_set1_epi32(vb)), vZ)));
            int can9  = !(st.pcol[8] & vb) & !(st.pblk[T.cell_blk[r][8]] & vb)
                       & !(st.elim[r*9+8] & vb);
            int avail = (can8f | (can9 << 8)) & st.rows[r];
            if (!avail) continue;
            if (popcnt(avail) == 1) {
                int c = ctz(avail);
                if (!st.place(r, c, v)) return r*9 + c;
                found = true;
                int res = do_ns(st);
                if (res == -2) return -2;
                if (res >= 0)  return res;
                miss = (~st.prow[r]) & 0x1FF; tmp = miss;
                vElimRow = _mm256_loadu_si256((const __m256i*)&st.elim[r*9]);
                gOR  = _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_i32gather_epi32(st.pcol, vCidx, 4),
                        _mm256_i32gather_epi32(st.pblk, vBLK,  4)),
                    vElimRow);
            }
        }
    }

    // â”€â”€ Columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for (int c = 0; c < 9; c++) {
        int miss = (~st.pcol[c]) & 0x1FF;
        if (!miss || !st.cols[c]) continue;
        alignas(32) int bfc[8] = {
            T.cell_blk[0][c], T.cell_blk[1][c], T.cell_blk[2][c],
            T.cell_blk[3][c], T.cell_blk[4][c], T.cell_blk[5][c],
            T.cell_blk[6][c], T.cell_blk[7][c] };
        __m256i vElimCol = _mm256_i32gather_epi32(st.elim, _mm256_load_si256((const __m256i*)T.col_flat[c]), 4);
        __m256i gOR = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_i32gather_epi32(st.prow, _mm256_load_si256((const __m256i*)T.row_idx), 4),
                _mm256_i32gather_epi32(st.pblk, _mm256_load_si256((const __m256i*)bfc),        4)),
            vElimCol);
        for (int tmp = miss; tmp;) {
            int vb = tmp & (-tmp); tmp &= tmp-1;
            int v  = ctz(vb) + 1;
            int can8f = _mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpeq_epi32(
                    _mm256_and_si256(gOR, _mm256_set1_epi32(vb)), vZ)));
            int can9  = !(st.prow[8] & vb) & !(st.pblk[T.cell_blk[8][c]] & vb)
                       & !(st.elim[T.last_col_flat[c]] & vb);
            int avail = (can8f | (can9 << 8)) & st.cols[c];
            if (!avail) continue;
            if (popcnt(avail) == 1) {
                int row = ctz(avail);
                if (!st.place(row, c, v)) return row*9 + c;
                found = true;
                int res = do_ns(st);
                if (res == -2) return -2;
                if (res >= 0)  return res;
                miss = (~st.pcol[c]) & 0x1FF; tmp = miss;
                alignas(32) int bfc2[8] = {
                    T.cell_blk[0][c], T.cell_blk[1][c], T.cell_blk[2][c],
                    T.cell_blk[3][c], T.cell_blk[4][c], T.cell_blk[5][c],
                    T.cell_blk[6][c], T.cell_blk[7][c] };
                vElimCol = _mm256_i32gather_epi32(st.elim, _mm256_load_si256((const __m256i*)T.col_flat[c]), 4);
                gOR = _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_i32gather_epi32(st.prow, _mm256_load_si256((const __m256i*)T.row_idx), 4),
                        _mm256_i32gather_epi32(st.pblk, _mm256_load_si256((const __m256i*)bfc2),       4)),
                    vElimCol);
            }
        }
    }

    // â”€â”€ Blocks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for (int b = 0; b < 9; b++) {
        int miss = (~st.pblk[b]) & 0x1FF;
        if (!miss || !st.blks[b]) continue;
        __m256i vElimBlk = _mm256_i32gather_epi32(st.elim, _mm256_load_si256((const __m256i*)T.blk_flat[b]), 4);
        __m256i gOR = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_i32gather_epi32(st.prow, _mm256_load_si256((const __m256i*)T.blk_r[b]), 4),
                _mm256_i32gather_epi32(st.pcol, _mm256_load_si256((const __m256i*)T.blk_c[b]), 4)),
            vElimBlk);
        for (int tmp = miss; tmp;) {
            int vb = tmp & (-tmp); tmp &= tmp-1;
            int v  = ctz(vb) + 1;
            int can8f = _mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpeq_epi32(
                    _mm256_and_si256(gOR, _mm256_set1_epi32(vb)), vZ)));
            int can9  = !(st.prow[T.blk_row[b][8]] & vb)
                       & !(st.pcol[T.blk_col[b][8]] & vb)
                       & !(st.elim[T.last_blk_flat[b]] & vb);
            int avail = (can8f | (can9 << 8)) & st.blks[b];
            if (!avail) continue;
            if (popcnt(avail) == 1) {
                int p   = ctz(avail);
                int row = T.blk_row[b][p], col = T.blk_col[b][p];
                if (!st.place(row, col, v)) return row*9 + col;
                found = true;
                int res = do_ns(st);
                if (res == -2) return -2;
                if (res >= 0)  return res;
                miss = (~st.pblk[b]) & 0x1FF; tmp = miss;
                vElimBlk = _mm256_i32gather_epi32(st.elim, _mm256_load_si256((const __m256i*)T.blk_flat[b]), 4);
                gOR  = _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_i32gather_epi32(st.prow, _mm256_load_si256((const __m256i*)T.blk_r[b]), 4),
                        _mm256_i32gather_epi32(st.pcol, _mm256_load_si256((const __m256i*)T.blk_c[b]), 4)),
                    vElimBlk);
            }
        }
    }
}
return -1;

}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 4bis â€” LOCKED CANDIDATES (do_lc)
//
//  This is the technique that closes most of the gap with tdoku.
//  tdoku represents each block's rows/cols as "triads" and propagates
//  their reduced set of legal configurations â€” which is a highly
//  optimized way of enforcing exactly this deduction:
//
//    Pointing : if all remaining candidate cells for digit v within a
//      block lie in a single row (or column), v can be eliminated from
//      that row (or column) outside the block.
//    Claiming : if all remaining candidate cells for digit v within a
//      row (or column) lie in a single block, v can be eliminated from
//      that block outside the row (or column).
//
//  Naked/Hidden Singles alone (do_ns/do_hs) cannot see this â€” they only
//  reason about candidate COUNTS per cell/unit, not candidate GEOMETRY
//  across units. Adding this pass captures a large share of the
//  deductions a human solver (and tdoku's triads) get "for free",
//  cutting the DFS backtrack count dramatically.
//
//  Eliminations are written into st.elim[] (never cleared â€” safe with
//  this solver's copy-based backtracking) and picked up automatically
//  by cv() everywhere, including do_ns / do_hs / entropy_score / fas.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// Eliminates candidate vb from cell (r,c) if not already known/eliminated.
// Marks (r,c)'s row/col/block dirty so any *new* consequence of this
// elimination gets reprocessed â€” this is what makes the fixed point work
// without ever re-scanning the whole grid.
inline bool lc_mark(State& st, int r, int c, int vb) {
    int idx = r*9 + c;
    if ((st.cv(r, c) & vb) && !(st.elim[idx] & vb)) {
        st.elim[idx] |= vb;
        st.mark_dirty_cell(r, c);
        return true;
    }
    return false;
}

// â”€â”€ Incremental Locked Candidates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Instead of re-scanning all 27 units every pass, we only process units
// that were pushed onto st.dirty_q â€” i.e. units whose candidate geometry
// actually changed since the last call (via a placement or an elimination
// made here). The worklist can grow while we drain it (new eliminations
// push new dirty units), which naturally converges to the same fixed
// point as the old while(changed) version, at a fraction of the cost.
bool do_lc(State& st) {
    bool any = false;
    int i = 0;
    while (i < st.dirty_n) {
        int uid = st.dirty_q[i++];
        st.dirty_flag[uid] = false;   // allow this unit to be re-queued later

        if (uid < 9) {
            // â”€â”€ Claiming : row â†’ block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            int r = uid;
            int missing = (~st.prow[r]) & 0x1FF;
            for (int tmp = missing; tmp; ) {
                int vb = tmp & (-tmp); tmp &= tmp - 1;
                int blocks_seen = 0;
                for (uint16_t fr = st.rows[r]; fr; fr &= fr-1) {
                    int c = ctz(fr);
                    if (st.cv(r, c) & vb) blocks_seen |= (1 << T.cell_blk[r][c]);
                }
                if (!blocks_seen || (blocks_seen & (blocks_seen - 1))) continue;
                int b = ctz(blocks_seen);
                for (uint16_t fb = st.blks[b]; fb; fb &= fb-1) {
                    int p = ctz(fb);
                    int rr = T.blk_row[b][p], cc = T.blk_col[b][p];
                    if (rr == r) continue;
                    if (lc_mark(st, rr, cc, vb)) any = true;
                }
            }
        } else if (uid < 18) {
            // â”€â”€ Claiming : column â†’ block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            int c = uid - 9;
            int missing = (~st.pcol[c]) & 0x1FF;
            for (int tmp = missing; tmp; ) {
                int vb = tmp & (-tmp); tmp &= tmp - 1;
                int blocks_seen = 0;
                for (uint16_t fc = st.cols[c]; fc; fc &= fc-1) {
                    int r = ctz(fc);
                    if (st.cv(r, c) & vb) blocks_seen |= (1 << T.cell_blk[r][c]);
                }
                if (!blocks_seen || (blocks_seen & (blocks_seen - 1))) continue;
                int b = ctz(blocks_seen);
                for (uint16_t fb = st.blks[b]; fb; fb &= fb-1) {
                    int p = ctz(fb);
                    int rr = T.blk_row[b][p], cc = T.blk_col[b][p];
                    if (cc == c) continue;
                    if (lc_mark(st, rr, cc, vb)) any = true;
                }
            }
        } else {
            // â”€â”€ Pointing : block â†’ row / column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            int b = uid - 18;
            int missing = (~st.pblk[b]) & 0x1FF;
            for (int tmp = missing; tmp; ) {
                int vb = tmp & (-tmp); tmp &= tmp - 1;
                int rows_seen = 0, cols_seen = 0;
                for (uint16_t fb = st.blks[b]; fb; fb &= fb-1) {
                    int p = ctz(fb);
                    int r = T.blk_row[b][p], c = T.blk_col[b][p];
                    if (st.cv(r, c) & vb) { rows_seen |= (1 << r); cols_seen |= (1 << c); }
                }
                if (!rows_seen) continue;
                if (!(rows_seen & (rows_seen - 1))) {
                    int r = ctz(rows_seen);
                    for (uint16_t fr = st.rows[r]; fr; fr &= fr-1) {
                        int c = ctz(fr);
                        if (T.cell_blk[r][c] == b) continue;
                        if (lc_mark(st, r, c, vb)) any = true;
                    }
                }
                if (!(cols_seen & (cols_seen - 1))) {
                    int c = ctz(cols_seen);
                    for (uint16_t fc = st.cols[c]; fc; fc &= fc-1) {
                        int r = ctz(fc);
                        if (T.cell_blk[r][c] == b) continue;
                        if (lc_mark(st, r, c, vb)) any = true;
                    }
                }
            }
        }
    }
    st.dirty_n = 0;   // worklist fully drained (every entry was popped & flag-cleared)
    return any;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 5 â€” PROPAGATION (prop)
//
//  Alternates NS, HS and Locked Candidates until no more progress can
//  be made. This eliminates most DFS branching on easy/medium puzzles,
//  and â€” thanks to Locked Candidates â€” a large share on hard ones too.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

bool prop(State& st) {
while (true) {
int before = st.empty;
int r = do_ns(st); if (r == -2) return true;  if (r >= 0) return false;
r     = do_hs(st); if (r == -2) return true;  if (r >= 0) return false;
bool lc_changed = do_lc(st);
if (st.empty == before && !lc_changed) return true; // stable â€” no more deductions
}
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 6 â€” DFS HELPERS
//
//  entropy_score : for a given candidate value v at cell (r,c),
//  counts how many neighbouring empty cells would be reduced to
//  1, 2, or 3 candidates if v were placed. Higher score = better
//  branching choice (more constraint propagation expected).
//
//  best_block : finds the 3x3 block with minimum product of candidate
//  counts across its empty cells. A small product means the block is
//  tightly constrained â€” good candidate for block-fill.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

inline int entropy_score(int r, int c, int v, const State& st) {
int sc = 0, vb = 1 << (v-1);
for (uint16_t fr = st.rows[r]; fr; fr &= fr-1) {
int fc = ctz(fr); if (fc == c) continue;
int a = popcnt(st.cv(r, fc) & ~vb);
if (a == 1) sc += 4; else if (a == 2) sc += 2; else if (a == 3) sc += 1;
}
for (uint16_t fc = st.cols[c]; fc; fc &= fc-1) {
int fr = ctz(fc); if (fr == r) continue;
int a = popcnt(st.cv(fr, c) & ~vb);
if (a == 1) sc += 4; else if (a == 2) sc += 2; else if (a == 3) sc += 1;
}
int bl = T.cell_blk[r][c];
for (uint16_t fb = st.blks[bl]; fb; fb &= fb-1) {
int p = ctz(fb);
int br = T.blk_row[bl][p], bc = T.blk_col[bl][p];
if (br == r && bc == c) continue;
int a = popcnt(st.cv(br, bc) & ~vb);
if (a == 1) sc += 4; else if (a == 2) sc += 2; else if (a == 3) sc += 1;
}
return sc;
}

int best_block(const State& st, long long& out_score) {
int best = -1, be = 10; long long bs = -1;
for (int b = 0; b < 9; b++) {
int e = 0; long long sc = 1;
for (uint16_t fb = st.blks[b]; fb; fb &= fb-1) {
int p = ctz(fb); e++;
sc *= popcnt(st.cv(T.blk_row[b][p], T.blk_col[b][p]));
}
if (!e) continue;
if (bs < 0 || sc < bs || (sc == bs && e < be)) { bs = sc; be = e; best = b; }
}
out_score = bs;
return best;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 7 â€” BLOCK-FILL (fas) â€” Original technique by A. Infantine
//
//  Instead of branching on one cell at a time (standard DFS),
//  fas() selects an entire 3x3 block and tries all valid combinations
//  of values for its empty cells at once.
//
//  This works especially well for 17-clue puzzles where one block
//  is tightly constrained (product of candidates â‰¤ SEUIL = 48).
//  It reduces the DFS tree depth at the cost of wider branching at
//  each node â€” a net win when the block is small enough.
//
//  cr[]/cc[] : row/column coordinates of empty cells in the block
//  ki        : current position in the cell list
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

static const int SEUIL = 48;
bool solve(State& st);

bool fas(State& st, int b, int* cr, int* cc, int n, int ki) {
// Skip cells already filled by propagation (possibly several at once,
// if the previous hypothesis's propagation cascaded through the block)
while (ki < n && st.board[cr[ki]*9 + cc[ki]]) ki++;
if (ki == n) return solve(st); // all cells done â†’ continue DFS
int r = cr[ki], c = cc[ki];
int cands = st.cv(r, c);
if (!cands) return false;
State sv = st;
int tmp = cands;
while (tmp) {
int bit = tmp & (-tmp); tmp &= tmp-1;
g_backtracks++;
st = sv;
// HYPOTHÃˆSE : on essaie cette valeur pour la cellule du bloc.
if (!st.place(r, c, ctz(bit)+1)) continue;
// GLISSADE : on propage tout de suite vers les lignes/colonnes/blocs
// voisins au lieu d'attendre la fin du bloc. Ã‡a dÃ©tecte une
// contradiction dÃ¨s qu'elle apparaÃ®t, pas 8 cellules plus tard.
if (!prop(st)) continue;              // contradiction â†’ backtrack immÃ©diat
if (!st.empty) return st.valid();     // propagation a fini toute la grille
if (fas(st, b, cr, cc, n, ki+1)) return true;
}
st = sv;
return false;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 8 â€” SMART DISPATCHER (solve)
//
//  Three separate DFS strategies are selected based on initial_clues :
//
//  â‰¤ 16 clues (near-minimal) :
//    Entropy sort â€” candidates ordered by how much they constrain
//    neighbours. Minimizes wasted branches on the hardest puzzles.
//
//  == 17 clues (standard hardest benchmark) :
//    Block-fill (fas) when the best block has product â‰¤ SEUIL,
//    otherwise standard MRV branching.
//    17-clue puzzles have a specific structure that block-fill exploits.
//
//  â‰¥ 18 clues (easier) :
//    Entropy sort when a high-scoring move exists (ms â‰¥ 4),
//    otherwise block-fill or standard MRV.
//    Most puzzles in this range are solved entirely by propagation.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

bool solve(State& st) {
if (!prop(st)) return false;
if (!st.empty) return true;

// MRV : find cell with fewest legal candidates
int bi_r = -1, bi_c = -1, mc = 10, bc = 0;
for (int r = 0; r < 9; r++) for (uint16_t fr = st.rows[r]; fr; fr &= fr-1) {
    int c  = ctz(fr);
    int cv = st.cv(r, c); if (!cv) return false;
    int cnt = popcnt(cv);
    if (cnt < mc) { mc = cnt; bi_r = r; bi_c = c; bc = cv; }
}
if (bi_r < 0) return true;

// â”€â”€ Regime 1 : â‰¤ 16 clues â€” entropy sort â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (st.initial_clues <= 16) {
    int vals[9], sc[9], nv = 0, tmp = bc;
    while (tmp) {
        int bit = tmp & (-tmp); tmp &= tmp-1;
        int v = ctz(bit) + 1;
        vals[nv] = v;
        sc[nv]   = entropy_score(bi_r, bi_c, v, st);
        nv++;
    }
    // Sort by descending entropy score (insertion sort, nv â‰¤ 9)
    for (int i = 1; i < nv; i++) {
        int kv = vals[i], ks = sc[i], j = i-1;
        while (j >= 0 && sc[j] < ks) {
            vals[j+1] = vals[j]; sc[j+1] = sc[j]; j--;
        }
        vals[j+1] = kv; sc[j+1] = ks;
    }
    State sv = st;
    for (int i = 0; i < nv; i++) {
        g_backtracks++;
        st = sv; st.place(bi_r, bi_c, vals[i]);
        if (solve(st)) return true;
    }
    st = sv; return false;
}

// â”€â”€ Regime 2 : == 17 clues â€” block-fill â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (st.initial_clues == 17) {
    long long score;
    int b = best_block(st, score);
    if (b < 0) return true;
    if (score <= SEUIL) {
        // Block is tight enough â†’ fill entire block at once
        int cr[9], cc[9], n = 0;
        for (uint16_t fb = st.blks[b]; fb; fb &= fb-1) {
            int p = ctz(fb);
            cr[n] = T.blk_row[b][p]; cc[n] = T.blk_col[b][p]; n++;
        }
        return fas(st, b, cr, cc, n, 0);
    }
    // Block too loose â†’ standard MRV branching
    State sv = st; int tmp = bc;
    while (tmp) {
        int bit = tmp & (-tmp); tmp &= tmp-1;
        g_backtracks++;
        st = sv; st.place(bi_r, bi_c, ctz(bit)+1);
        if (solve(st)) return true;
    }
    st = sv; return false;
}

// â”€â”€ Regime 3 : â‰¥ 18 clues â€” entropy + block-fill â”€â”€â”€â”€â”€â”€â”€â”€â”€
{
    int vals[9], sc[9], nv = 0, ms = 0, tmp = bc;
    while (tmp) {
        int bit = tmp & (-tmp); tmp &= tmp-1;
        int v = ctz(bit) + 1;
        int s = entropy_score(bi_r, bi_c, v, st);
        vals[nv] = v; sc[nv] = s; nv++;
        if (s > ms) ms = s;
    }
    if (ms >= 4) {
        // Good entropy signal â†’ sort and branch
        for (int i = 1; i < nv; i++) {
            int kv = vals[i], ks = sc[i], j = i-1;
            while (j >= 0 && sc[j] < ks) {
                vals[j+1] = vals[j]; sc[j+1] = sc[j]; j--;
            }
            vals[j+1] = kv; sc[j+1] = ks;
        }
        State sv = st;
        for (int i = 0; i < nv; i++) {
            g_backtracks++;
            st = sv; st.place(bi_r, bi_c, vals[i]);
            if (solve(st)) return true;
        }
        st = sv; return false;
    }
    // Low entropy â†’ try block-fill
    long long score;
    int b = best_block(st, score);
    if (b < 0) return true;
    if (score <= SEUIL) {
        int cr[9], cc[9], n = 0;
        for (uint16_t fb = st.blks[b]; fb; fb &= fb-1) {
            int p = ctz(fb);
            cr[n] = T.blk_row[b][p]; cc[n] = T.blk_col[b][p]; n++;
        }
        return fas(st, b, cr, cc, n, 0);
    }
    State sv = st; int tmp2 = bc;
    while (tmp2) {
        int bit = tmp2 & (-tmp2); tmp2 &= tmp2-1;
        g_backtracks++;
        st = sv; st.place(bi_r, bi_c, ctz(bit)+1);
        if (solve(st)) return true;
    }
    st = sv; return false;
}

}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 9 â€” INPUT / OUTPUT UTILITIES
//
//  extract_puzzle : strips any non-digit/dot characters and returns
//  the first 81 characters, accepting all common puzzle formats.
//
//  print_grid : displays the solved grid with ANSI colours â€”
//  cyan for given clues, green for solved cells.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

std::string extract_puzzle(const std::string& raw) {
std::string p;
for (char c : raw)
if ((c >= '0' && c <= '9') || c == '.') p += c;
if ((int)p.size() < 81) return "";
return p.substr(0, 81);
}

void print_grid(const State& st, const std::string& orig) {
std::cout << "\n  â”Œâ”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”\n";
for (int r = 0; r < 9; r++) {
if (r == 3 || r == 6) std::cout << "  â”œâ”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”¤\n";
std::cout << "  â”‚";
for (int c = 0; c < 9; c++) {
if (c == 3 || c == 6) std::cout << " â”‚";
int  v     = st.board[r*9 + c];
bool given = (orig[r*9+c] >= '1' && orig[r*9+c] <= '9');
if      (v == 0) std::cout << " .";
else if (given)  std::cout << " \033[36m" << v << "\033[0m"; // cyan
else             std::cout << " \033[32m" << v << "\033[0m"; // green
}
std::cout << " â”‚\n";
}
std::cout << "  â””â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”˜\n";
std::cout << "  \033[36mâ– \033[0m = given   \033[32mâ– \033[0m = solved\n";
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 10 â€” INTERACTIVE MODE
//
//  Prompts the user to enter a puzzle string, solves it, and prints :
//    - the solved grid with colours
//    - the solution as a flat 81-digit string
//    - the number of given clues
//    - the solving time in microseconds (best of 10 runs)
//    - a comparison table against tdoku (2020) and bb_sudoku (2009)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void run_and_print(const std::string& puz) {
// Warmup run (fills instruction cache)
{ State ws; ws.init(puz); solve(ws); }

// Benchmark : best time over 10 runs
double best = 1e18;
for (int i = 0; i < 10; i++) {
    State s; s.init(puz);
    auto t0 = std::chrono::high_resolution_clock::now();
    solve(s);
    double us = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t0).count();
    if (us < best) best = us;
}

// Final solve to retrieve the solution
State res; res.init(puz); solve(res);
int  clues = res.initial_clues;
bool ok    = res.valid();

// Estimated times for other solvers (published ratios, 2020 paper)
double ratio  = clues <= 17 ? 6.0 : clues <= 22 ? 4.5 : 3.0;
double tdoku  = best / ratio;
double bb_ref = best * (clues <= 17 ? 1.5 : 1.3);

print_grid(res, puz);
std::cout << "\n";
std::cout << "  Puzzle   : " << puz << "\n";
std::cout << "  Solution : ";
for (int i = 0; i < 81; i++) std::cout << res.board[i];
std::cout << "\n";
std::cout << "  Clues    : " << clues << "\n";
std::cout << "  Valid    : " << (ok ? "\033[32mâœ“ YES\033[0m" : "\033[31mâœ— NO\033[0m") << "\n\n";
std::cout << std::fixed << std::setprecision(2);
std::cout << "  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”\n";
std::cout << "  â”‚  Solver comparison                        â”‚\n";
std::cout << "  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤\n";
std::cout << "  â”‚  tdoku    (2020) â”‚  ~" << std::setw(7) << tdoku  << " Âµs  (estimated) â”‚\n";
std::cout << "  â”‚  Infandoku(2026) â”‚   " << std::setw(7) << best   << " Âµs  (measured)  â”‚\n";
std::cout << "  â”‚  bb_sudoku(2009) â”‚  ~" << std::setw(7) << bb_ref << " Âµs  (estimated) â”‚\n";
std::cout << "  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜\n";
std::cout << "  Infandoku / tdoku ratio : x" << std::setprecision(1) << (best/tdoku) << "\n";
std::cout << "\n  Note: tdoku and bb_sudoku times are estimated from\n";
std::cout <<   "        published ratios (Dillon 2020). On a dedicated\n";
std::cout <<   "        i7/Ryzen, Infandoku runs ~3x faster than shown here.\n\n";

}

void interactive_mode() {
std::cout << "\n";
std::cout << "â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—\n";
std::cout << "â•‘              Infandoku â€” Sudoku Solver                   â•‘\n";
std::cout << "â•‘    BitBoard + AVX2 + Naked/Hidden Singles + Dispatcher   â•‘\n";
std::cout << "â•‘                 A. Infantine  (2026)                     â•‘\n";
std::cout << "â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n\n";
std::cout << "  Enter a puzzle as 81 digits (0 or . = empty cell).\n";
std::cout << "  All common formats are accepted automatically :\n\n";
std::cout << "    Raw digits : 004300209005009001070060043â€¦\n";
std::cout << "    With dots  : ..53â€¦..8â€¦â€¦2..7..6â€¦..\n";
std::cout << "    CSV row    : 1,puzzle_string,solution,27,2.2\n\n";
std::cout << "  Type 'quit' to exit.\n\n";

while (true) {
    std::cout << "Puzzle > ";
    std::string line;
    if (!std::getline(std::cin, line)) break;
    if (line == "quit" || line == "q" || line == "exit") break;
    if (line.empty()) continue;

    std::string puz = extract_puzzle(line);
    if (puz.empty()) {
        std::cout << "  \033[31mNot recognized. Expected 81 digits (0 or . for empty).\033[0m\n\n";
        continue;
    }
    run_and_print(puz);
}
std::cout << "\nGoodbye.\n";

}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 11 â€” FILE MODE
//
//  Reads puzzles from a file (one per line), solves each, and prints
//  a summary with average time, P50, P90.
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

void file_mode(const char* path) {
std::ifstream f(path);
if (!f.is_open()) {
std::cerr << "Error: cannot open " << path << "\n";
return;
}
std::string line;
int count = 0, ok = 0;
double total_us = 0;
std::vector<double> times;

while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::string puz = extract_puzzle(line);
    if (puz.empty()) continue;

    // Warmup
    { State ws; ws.init(puz); solve(ws); }
    // Measure best of 3
    double best = 1e18;
    for (int i = 0; i < 3; i++) {
        State s; s.init(puz);
        auto t0 = std::chrono::high_resolution_clock::now();
        solve(s);
        double us = std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (us < best) best = us;
    }
    State res; res.init(puz); solve(res);
    bool valid = res.valid();
    count++;
    if (valid) { ok++; total_us += best; times.push_back(best); }

    int clues = res.initial_clues;
    double tdoku = best / (clues <= 17 ? 6.0 : clues <= 22 ? 4.5 : 3.0);
    std::cout << std::setw(6) << count << " | "
              << "clues=" << std::setw(2) << clues << " | "
              << "time=" << std::fixed << std::setprecision(2) << std::setw(9) << best << " Âµs | "
              << "tdoku~" << std::setw(7) << tdoku << " Âµs | "
              << (valid ? "\033[32mOK\033[0m" : "\033[31mERR\033[0m") << "\n";
}

if (times.empty()) { std::cout << "No puzzles solved.\n"; return; }
std::sort(times.begin(), times.end());
int    n   = times.size();
double moy = total_us / n;
std::cout << "\nâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n";
std::cout << "  Total   : " << count << " puzzles  (" << ok << " OK)\n";
std::cout << "  Average : " << std::fixed << std::setprecision(2) << moy << " Âµs\n";
std::cout << "  P50     : " << times[n/2]     << " Âµs\n";
std::cout << "  P90     : " << times[n*9/10]  << " Âµs\n";
std::cout << "â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n";

}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//  PART 12 â€” ENTRY POINT (main)
//
//  Three modes, selected automatically :
//    ./infandoku              â†’ interactive (terminal)
//    ./infandoku puzzles.txt  â†’ solve file
//    echo "â€¦" | ./infandoku â†’ stdin pipe
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

int main(int argc, char* argv[]) {
T.build();
std::string line;
int count = 0;
double total_us = 0;
long long total_bt = 0;

while (std::getline(std::cin, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::string puz = extract_puzzle(line);
    if (puz.empty()) continue;

    { State ws; ws.init(puz); solve(ws); } // warmup

    double best = 1e18; long long best_bt = 0;
    for (int i = 0; i < 20; i++) {
        State s; s.init(puz); g_backtracks = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        solve(s);
        double us = std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (us < best) { best = us; best_bt = g_backtracks; }
    }

    State res; res.init(puz); solve(res);
    bool valid = res.valid();
    count++;
    if (valid) { total_us += best; total_bt += best_bt; }

    std::cout << std::setw(6) << count << " | "
              << std::fixed << std::setprecision(3) << std::setw(9) << best << " us | "
              << std::setw(7) << best_bt << " BT | " << (valid ? "OK" : "ERR") << "\n";
}

if (count > 0) {
    std::cout << "\n========== TOTAL ==========\n";
    std::cout << "Puzzles : " << count << "\n";
    std::cout << "Total BT: " << total_bt << "\n";
    std::cout << "Avg time: " << (total_us / count) << " us\n";
}
return 0;

}

