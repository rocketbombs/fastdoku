// Benchmark driver for Infandoku (https://github.com/alphonseinfantine/Infandoku,
// MIT), using the identical protocol to bench/tdoku_bench.cc and fastdoku's
// own `bench`: same parsing, same in-memory puzzle vector, same best-of-N,
// same solution checksum.
//
// Infandoku ships as a single self-contained translation unit with its own
// main(), so it is included here with main renamed out of the way.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Infandoku's solve() and State come from its source; rename its entry point.
#define main infandoku_original_main
#include "infandoku_v36.cpp"
#undef main

static const size_t kStride = 82;

static std::vector<char> ReadPuzzles(const char *path, size_t limit, size_t *count) {
    std::vector<char> out;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    char line[1024];
    size_t n_puzzles = 0;
    while (n_puzzles < limit && fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char p[kStride];
        size_t n = 0;
        for (size_t i = 0; line[i] && n < 81; i++) {
            char c = line[i];
            if (c >= '1' && c <= '9') p[n++] = c;
            else if (c == '.' || c == '0') p[n++] = '.';
        }
        if (n != 81) continue;
        p[81] = '\0';
        out.insert(out.end(), p, p + kStride);
        n_puzzles++;
    }
    fclose(f);
    *count = n_puzzles;
    return out;
}

static uint64_t Checksum(const int *board) {
    uint64_t a = 0;
    for (int i = 0; i < 81; i++) a = a * 31 + (uint64_t)board[i];
    return a;
}

// Validate a completed grid independently of the solver that produced it.
static bool ValidSolution(const int *board, const char *clues) {
    for (int i = 0; i < 81; i++) {
        if (board[i] < 1 || board[i] > 9) return false;
        if (clues[i] >= '1' && clues[i] <= '9' && board[i] != clues[i] - '0') return false;
    }
    for (int u = 0; u < 9; u++) {
        int r = 0, c = 0, b = 0;
        for (int k = 0; k < 9; k++) {
            r |= 1 << board[u * 9 + k];
            c |= 1 << board[k * 9 + u];
            b |= 1 << board[(u / 3) * 27 + (u % 3) * 3 + (k / 3) * 9 + (k % 3)];
        }
        if (r != 0x3FE || c != 0x3FE || b != 0x3FE) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: infandoku_bench <file> [--rounds N] [--limit N]\n");
        return 2;
    }
    const char *path = argv[1];
    int rounds = 10;
    size_t limit = (size_t)-1;
    for (int i = 2; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--rounds")) rounds = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--limit")) limit = (size_t)atoll(argv[i + 1]);
    }

    // Infandoku's main() does this first; the lookup tables are all zero
    // until it runs, and every solve fails instantly without it.
    T.build();

    size_t n_puzzles = 0;
    auto puzzles = ReadPuzzles(path, limit, &n_puzzles);
    if (n_puzzles == 0) {
        fprintf(stderr, "no puzzles\n");
        return 1;
    }

    // Untimed verification pass: every returned grid must be a valid
    // completion of its clues.
    size_t solved = 0, invalid = 0;
    for (size_t i = 0; i < n_puzzles; i++) {
        const char *clues = &puzzles[i * kStride];
        State st;
        st.init(std::string(clues, 81));
        if (solve(st)) {
            solved++;
            if (!ValidSolution(st.board, clues)) invalid++;
        }
    }

    double best = 1e30;
    uint64_t sum_ref = 0;
    bool have_ref = false;
    for (int r = 0; r < rounds; r++) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t sum = 0;
        for (size_t i = 0; i < n_puzzles; i++) {
            State st;
            st.init(std::string(&puzzles[i * kStride], 81));
            if (solve(st)) sum += Checksum(st.board);
        }
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (have_ref && sum != sum_ref) {
            fprintf(stderr, "nondeterministic results across rounds\n");
            return 1;
        }
        sum_ref = sum;
        have_ref = true;
        best = std::min(best, dt);
    }

    double n = (double)n_puzzles;
    double per = best / n, pps = n / best;
    char per_s[64], rate_s[64];
    if (per * 1e9 < 1000.0) snprintf(per_s, sizeof(per_s), "%.1f ns", per * 1e9);
    else snprintf(per_s, sizeof(per_s), "%.3f us", per * 1e6);
    if (pps >= 1e6) snprintf(rate_s, sizeof(rate_s), "%.2fM/s", pps / 1e6);
    else if (pps >= 1e3) snprintf(rate_s, sizeof(rate_s), "%.1fK/s", pps / 1e3);
    else snprintf(rate_s, sizeof(rate_s), "%.0f/s", pps);

    const char *base = strrchr(path, '\\');
    if (!base) base = strrchr(path, '/');
    base = base ? base + 1 : path;

    printf("%-32s %10s %10s  [%zu puzzles, %zu ok, %zu INVALID, infandoku, 1t, best/%d, sum %016llx]\n",
           base, per_s, rate_s, n_puzzles, solved, invalid, rounds,
           (unsigned long long)sum_ref);
    return 0;
}
