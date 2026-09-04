// Q1: Distributed Matrix Multiplication using the Row-Row method.
//
//   C = A * B,  A is m x n,  B is n x p,  C is m x p
//
// Row-Row formulation: row i of C is a weighted sum of the rows of B, with the
// weights taken from row i of A:
//
//   c_i = a_i[1]*B[1,:] + a_i[2]*B[2,:] + ... + a_i[n]*B[n,:]
//
// Row i of C therefore depends only on row i of A and on all of B.  That is
// what makes the partitioning clean: A is scattered row-wise, B is broadcast
// in full, every rank computes its own rows of C with no inter-worker
// communication, and rank 0 gathers the row-slices back.
//
// Build:  mpicxx -O2 -std=c++17 -o matmul_rowrow matmul_rowrow.cpp
//
// Usage:
//   mpirun -np P ./matmul_rowrow --gen m n p [--seed S] [--verify] [--print] [--reps R]
//   mpirun -np P ./matmul_rowrow --file A.txt B.txt   [--verify] [--print] [--reps R]
//   mpirun -np P ./matmul_rowrow --gen m n p --csv small   (adds a label column)
//
// Matrix file format:  first line "rows cols", then rows lines of cols integers.

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

namespace {

// ---------------------------------------------------------------------------
// Row distribution: rank r owns rows[r] consecutive rows of A (and of C).
// The first (m % P) ranks take one extra row so that the remainder is spread
// evenly instead of piling onto the last rank.  When P > m the trailing ranks
// legitimately receive zero rows; they still take part in every collective.
// ---------------------------------------------------------------------------
struct RowSplit {
    vector<int> rows;      // rows owned by each rank
    vector<int> row_off;   // index of each rank's first row in A / C
    vector<int> cnt_a;     // Scatterv counts for A, in elements
    vector<int> dsp_a;
    vector<int> cnt_c;     // Gatherv counts for C, in elements
    vector<int> dsp_c;
};

RowSplit make_split(int m, int n, int p, int P) {
    RowSplit s;
    s.rows.resize(P);
    s.row_off.resize(P);
    s.cnt_a.resize(P);
    s.dsp_a.resize(P);
    s.cnt_c.resize(P);
    s.dsp_c.resize(P);

    const int base = m / P;
    const int rem  = m % P;

    int off = 0;
    for (int r = 0; r < P; ++r) {
        s.rows[r]    = base + (r < rem ? 1 : 0);
        s.row_off[r] = off;
        s.cnt_a[r]   = s.rows[r] * n;
        s.dsp_a[r]   = off * n;
        s.cnt_c[r]   = s.rows[r] * p;
        s.dsp_c[r]   = off * p;
        off += s.rows[r];
    }
    return s;
}

// ---------------------------------------------------------------------------
// The Row-Row kernel.  The k loop sits outside the j loop: for each entry
// a = A[i][k] we scale the whole row B[k,:] and accumulate it into C[i,:].
// That is the Row-Row formulation written literally, and it also walks B and C
// along their rows, which is the cache-friendly direction for row-major
// storage.
// ---------------------------------------------------------------------------
void rowrow_multiply(const vector<int>& A, const vector<int>& B, vector<int>& C,
                     int local_rows, int n, int p) {
    fill(C.begin(), C.end(), 0);
    for (int i = 0; i < local_rows; ++i) {
        const int* a_row = &A[static_cast<size_t>(i) * n];
        int*       c_row = &C[static_cast<size_t>(i) * p];
        for (int k = 0; k < n; ++k) {
            const int a = a_row[k];
            if (a == 0) continue;              // skip the no-op scaling
            const int* b_row = &B[static_cast<size_t>(k) * p];
            for (int j = 0; j < p; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

// Straightforward serial multiply, used only by --verify on rank 0.
void serial_multiply(const vector<int>& A, const vector<int>& B, vector<int>& C,
                     int m, int n, int p) {
    C.assign(static_cast<size_t>(m) * p, 0);
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            const int a = A[static_cast<size_t>(i) * n + k];
            if (a == 0) continue;
            for (int j = 0; j < p; ++j) {
                C[static_cast<size_t>(i) * p + j] += a * B[static_cast<size_t>(k) * p + j];
            }
        }
    }
}

bool read_matrix(const string& path, vector<int>& M, int& rows, int& cols) {
    ifstream in(path);
    if (!in) return false;
    if (!(in >> rows >> cols)) return false;
    if (rows < 0 || cols < 0) return false;
    M.resize(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < M.size(); ++i) {
        if (!(in >> M[i])) return false;
    }
    return true;
}

void fill_random(vector<int>& M, size_t count, unsigned seed) {
    M.resize(count);
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(-9, 9);
    for (size_t i = 0; i < count; ++i) M[i] = dist(rng);
}

void print_matrix(const vector<int>& M, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d%c", M[static_cast<size_t>(i) * cols + j],
                   j + 1 == cols ? '\n' : ' ');
        }
    }
}

[[noreturn]] void usage_and_exit(int rank) {
    if (rank == 0) {
        fprintf(stderr,
                "usage: matmul_rowrow --gen m n p [--seed S] [--reps R] [--verify] [--print] [--csv LABEL]\n"
                "       matmul_rowrow --file A.txt B.txt [--reps R] [--verify] [--print] [--csv LABEL]\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // ---------------- argument parsing ----------------
    bool gen = false, from_file = false, verify = false, want_print = false;
    int m = 0, n = 0, p = 0, reps = 1;
    unsigned seed = 42;
    string fa, fb, label;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "--gen" && i + 3 < argc) {
            gen = true;
            m = atoi(argv[++i]);
            n = atoi(argv[++i]);
            p = atoi(argv[++i]);
        } else if (arg == "--file" && i + 2 < argc) {
            from_file = true;
            fa = argv[++i];
            fb = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<unsigned>(strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--reps" && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            label = argv[++i];
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--print") {
            want_print = true;
        } else {
            usage_and_exit(rank);
        }
    }
    if (gen == from_file) usage_and_exit(rank);   // exactly one input mode
    if (reps < 1) reps = 1;

    // ---------------- rank 0 obtains A and B ----------------
    vector<int> A, B, C;
    if (rank == 0) {
        if (from_file) {
            int ar = 0, ac = 0, br = 0, bc = 0;
            if (!read_matrix(fa, A, ar, ac)) {
                fprintf(stderr, "error: cannot read %s\n", fa.c_str());
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            if (!read_matrix(fb, B, br, bc)) {
                fprintf(stderr, "error: cannot read %s\n", fb.c_str());
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            if (ac != br) {
                fprintf(stderr, "error: inner dimensions disagree (%d vs %d)\n", ac, br);
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            m = ar; n = ac; p = bc;
        } else {
            if (m <= 0 || n <= 0 || p <= 0) {
                fprintf(stderr, "error: m, n, p must all be positive\n");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            fill_random(A, static_cast<size_t>(m) * n, seed);
            fill_random(B, static_cast<size_t>(n) * p, seed + 1);
        }
    }

    // Dimensions must reach every rank before any buffer can be sized.
    int dims[3] = {m, n, p};
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    m = dims[0]; n = dims[1]; p = dims[2];

    const RowSplit split      = make_split(m, n, p, P);
    const int      local_rows = split.rows[rank];

    // Zero-row ranks (P > m) still allocate a 1-element buffer so that &v[0] is
    // always a valid pointer to hand to MPI; the count passed is 0 regardless.
    vector<int> local_A(max<size_t>(1, static_cast<size_t>(local_rows) * n));
    vector<int> local_C(max<size_t>(1, static_cast<size_t>(local_rows) * p));
    if (rank != 0) B.resize(static_cast<size_t>(n) * p);
    if (rank == 0) C.resize(static_cast<size_t>(m) * p);

    double t_dist = 0.0, t_comp = 0.0, t_gath = 0.0;

    for (int rep = 0; rep < reps; ++rep) {
        // ---- distribution: broadcast B in full, scatter the rows of A ----
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();

        MPI_Bcast(B.data(), n * p, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(rank == 0 ? A.data() : nullptr, split.cnt_a.data(), split.dsp_a.data(),
                     MPI_INT, local_A.data(), local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        const double t1 = MPI_Wtime();

        // ---- computation: purely local, no inter-worker communication ----
        rowrow_multiply(local_A, B, local_C, local_rows, n, p);

        MPI_Barrier(MPI_COMM_WORLD);
        const double t2 = MPI_Wtime();

        // ---- collection: gather the row-slices (gather, not reduce) ----
        MPI_Gatherv(local_C.data(), local_rows * p, MPI_INT,
                    rank == 0 ? C.data() : nullptr, split.cnt_c.data(), split.dsp_c.data(),
                    MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        const double t3 = MPI_Wtime();

        // Keep the fastest repetition, the usual convention for reducing the
        // effect of transient noise on a shared cluster.
        const double d = t1 - t0, c = t2 - t1, g = t3 - t2;
        if (rep == 0 || d + c + g < t_dist + t_comp + t_gath) {
            t_dist = d; t_comp = c; t_gath = g;
        }
    }

    // The wall time of a phase is set by its slowest rank.
    double local_t[3] = {t_dist, t_comp, t_gath};
    double max_t[3]   = {0.0, 0.0, 0.0};
    MPI_Reduce(local_t, max_t, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---------------- output ----------------
    int ok = 1;
    if (rank == 0) {
        if (verify) {
            vector<int> ref;
            serial_multiply(A, B, ref, m, n, p);
            ok = (ref == C) ? 1 : 0;          // integer entries compare exactly
        }

        const double total = max_t[0] + max_t[1] + max_t[2];
        const double comm  = max_t[0] + max_t[2];

        if (!label.empty()) {
            // CSV row for the benchmark driver:
            // label,m,n,p,P,t_dist,t_comp,t_gather,t_total,comm_pct,verify
            printf("%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%s\n",
                   label.c_str(), m, n, p, P,
                   max_t[0], max_t[1], max_t[2], total,
                   total > 0.0 ? 100.0 * comm / total : 0.0,
                   verify ? (ok ? "PASS" : "FAIL") : "-");
        } else {
            printf("m=%d n=%d p=%d P=%d\n", m, n, p, P);
            printf("dist=%.6f s  comp=%.6f s  gather=%.6f s  total=%.6f s\n",
                   max_t[0], max_t[1], max_t[2], total);
            printf("comm=%.2f%%  comp=%.2f%%\n",
                   total > 0.0 ? 100.0 * comm / total : 0.0,
                   total > 0.0 ? 100.0 * max_t[1] / total : 0.0);
            if (verify) printf("verify=%s\n", ok ? "PASS" : "FAIL");
        }

        if (want_print) print_matrix(C, m, p);
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ok ? 0 : 1;
}
