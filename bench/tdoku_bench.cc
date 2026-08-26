// Benchmark driver for tdoku, mirroring fastdoku's own `bench` harness so the
// two are measured identically: same file parsing, same in-memory puzzle
// vector, same best-of-N protocol, same solution checksum.
//
// Build (see bench/build_tdoku.ps1):
//   clang++ -O3 -march=native -std=c++17 -DNDEBUG -I<tdoku>/include \
//       tdoku_bench.cc solver_dpll_triad_simd.o -o tdoku_bench.exe

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tdoku.h"

// Puzzles are stored contiguously with an 82-byte stride: 81 chars + NUL.
// tdoku decides standard vs. pencilmark by checking input[81], so the
// terminator matters.
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
            if (c >= '1' && c <= '9') {
                p[n++] = c;
            } else if (c == '.' || c == '0') {
                p[n++] = '.';
            }
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

static uint64_t Checksum(const char *sol) {
    uint64_t a = 0;
    for (int i = 0; i < 81; i++) a = a * 31 + (uint64_t)(sol[i] - '0');
    return a;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: tdoku_bench <file> [--rounds N] [--limit N]\n");
        return 2;
    }
    const char *path = argv[1];
    int rounds = 10;
    size_t limit = (size_t)-1;
    for (int i = 2; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--rounds")) rounds = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--limit")) limit = (size_t)atoll(argv[i + 1]);
    }

    size_t n_puzzles = 0;
    auto puzzles = ReadPuzzles(path, limit, &n_puzzles);
    if (n_puzzles == 0) {
        fprintf(stderr, "no puzzles\n");
        return 1;
    }

    char solution[82];
    size_t guesses_total = 0, solved = 0;

    // Untimed verification + guess count pass.
    for (size_t i = 0; i < n_puzzles; i++) {
        size_t guesses = 0;
        size_t n = TdokuSolverDpllTriadSimd(&puzzles[i * kStride], 1, 0, solution, &guesses);
        guesses_total += guesses;
        solved += (n > 0);
    }

    double best = 1e30;
    uint64_t sum_ref = 0;
    bool have_ref = false;
    for (int r = 0; r < rounds; r++) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t sum = 0;
        for (size_t i = 0; i < n_puzzles; i++) {
            size_t guesses = 0;
            if (TdokuSolverDpllTriadSimd(&puzzles[i * kStride], 1, 0, solution, &guesses) > 0) {
                sum += Checksum(solution);
            }
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
    double per = best / n;
    double pps = n / best;

    char per_s[64], rate_s[64];
    if (per * 1e9 < 1000.0) snprintf(per_s, sizeof(per_s), "%.1f ns", per * 1e9);
    else snprintf(per_s, sizeof(per_s), "%.3f us", per * 1e6);
    if (pps >= 1e6) snprintf(rate_s, sizeof(rate_s), "%.2fM/s", pps / 1e6);
    else if (pps >= 1e3) snprintf(rate_s, sizeof(rate_s), "%.1fK/s", pps / 1e3);
    else snprintf(rate_s, sizeof(rate_s), "%.0f/s", pps);

    const char *base = strrchr(path, '\\');
    if (!base) base = strrchr(path, '/');
    base = base ? base + 1 : path;

    printf("%-32s %10s %10s  [%zu puzzles, %zu ok, tdoku, 1t, best/%d, sum %016llx, %.2f g/p]\n",
           base, per_s, rate_s, n_puzzles, solved, rounds,
           (unsigned long long)sum_ref, (double)guesses_total / n);
    return 0;
}
