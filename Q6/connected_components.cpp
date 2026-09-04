// Q6: Connected components of a large undirected graph.
//
// Every vertex must end up labelled with the *smallest* vertex ID in its
// connected component.  The graph arrives as an adjacency list; the vertices
// and their adjacency lists are distributed across the ranks, and the labels
// are settled by parallel message passing.
//
// Method -- Shiloach-Vishkin style label propagation:
//
//   label[v] = v                                  (every vertex its own root)
//   repeat
//     hooking:         for each local edge (u,v)  propose min(label[u],label[v])
//                      for both endpoints
//     combine:         Allreduce(MIN) over the label array, so a proposal made
//                      on one rank reaches the rank that owns the vertex
//     pointer jumping: label[v] <- label[label[v]], repeated until stable,
//                      which collapses long chains in O(log) steps instead of
//                      the O(diameter) that plain propagation would need
//   until no label changed anywhere
//
// Because every update is a minimum and the initial label of a vertex is its
// own ID, the fixed point of that iteration is exactly "smallest vertex ID in
// the component" -- which is the component ID the assignment asks for.
//
// Partitioning: rank r owns a contiguous block of ~V/P vertices together with
// their adjacency lists.  Edges are therefore owned by the rank that owns the
// endpoint they were listed under; an edge crossing a block boundary is seen by
// both of its owners, which is harmless since the operation is idempotent.
//
// The label array itself (V ints) is replicated on every rank.  With V <= 1e5
// that is 400 KB, small next to the E <= 1e6 adjacency data that stays
// distributed, and it makes the combine step a single Allreduce instead of a
// hand-rolled neighbour exchange.
//
// Build:  mpicxx -O2 -std=c++17 -o connected_components connected_components.cpp
//
// Usage:
//   mpirun -np P ./connected_components --file graph.txt [--print] [--verify]
//   mpirun -np P ./connected_components --gen V E [--seed S] [--verify]
//   mpirun -np P ./connected_components --gen V E --reps 3 --csv large
//
// Input format:  first line V, then V lines "k v_1 v_2 ... v_k".
// Output format: V lines "vertex_id component_id", sorted by vertex_id.

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {

// ---------------------------------------------------------------------------
// CSR storage for the slice of the graph a rank owns.  `off` has
// (vertex_count + 1) entries; the neighbours of local vertex i are
// adj[off[i] .. off[i+1]).  Global IDs are used throughout for neighbours, so
// no translation table is needed anywhere.
// ---------------------------------------------------------------------------
struct LocalGraph {
    int              first = 0;   // global ID of this rank's first vertex
    int              count = 0;   // number of vertices owned
    vector<int> off;
    vector<int> adj;
};

// ---------------------------------------------------------------------------
// Vertex distribution: rank r owns count[r] consecutive vertices.  The first
// (V % P) ranks take one extra so the remainder is spread rather than piled
// onto the last rank.  When P > V the trailing ranks own nothing; they still
// take part in every collective.
// ---------------------------------------------------------------------------
struct VertexSplit {
    vector<int> count;
    vector<int> first;
};

VertexSplit make_split(int V, int P) {
    VertexSplit s;
    s.count.resize(P);
    s.first.resize(P);
    const int base = V / P;
    const int rem  = V % P;
    int off = 0;
    for (int r = 0; r < P; ++r) {
        s.count[r] = base + (r < rem ? 1 : 0);
        s.first[r] = off;
        off += s.count[r];
    }
    return s;
}

// ---------------------------------------------------------------------------
// One hooking sweep over the locally owned edges.  For edge (u,v) both
// endpoints are pushed towards min(label[u], label[v]).  Writing both
// directions matters: an adjacency list is only guaranteed to list the edge
// from one side once the graph is split, and pushing both ways means a rank
// can improve a vertex it does not own -- the Allreduce afterwards is what
// delivers that improvement to the owner.
//
// Returns true if this rank changed anything.
// ---------------------------------------------------------------------------
bool hook(const LocalGraph& g, vector<int>& label) {
    bool changed = false;
    for (int i = 0; i < g.count; ++i) {
        const int u  = g.first + i;
        int       lu = label[u];
        for (int e = g.off[i]; e < g.off[i + 1]; ++e) {
            const int v  = g.adj[e];
            const int lv = label[v];
            if (lv < lu) {
                lu      = lv;
                changed = true;
            } else if (lu < lv) {
                label[v] = lu;               // improve the neighbour too
                changed  = true;
            }
        }
        label[u] = lu;
    }
    return changed;
}

// ---------------------------------------------------------------------------
// Pointer jumping: label[v] <- label[label[v]] until nothing moves.  Each pass
// halves the depth of every label chain, so a chain of length L collapses in
// O(log L) passes.  This is purely local -- the label array is replicated, so
// every rank computes the identical result and no communication is needed.
// ---------------------------------------------------------------------------
void jump(vector<int>& label) {
    const int V = static_cast<int>(label.size());
    bool moved = true;
    while (moved) {
        moved = false;
        for (int v = 0; v < V; ++v) {
            const int g = label[label[v]];
            if (g != label[v]) {
                label[v] = g;
                moved    = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The distributed solve.  Returns the number of rounds taken.
// ---------------------------------------------------------------------------
int connected_components(const LocalGraph& g, int V, vector<int>& label,
                         double& t_comm) {
    label.resize(V);
    iota(label.begin(), label.end(), 0);      // label[v] = v

    vector<int> combined(V);
    int rounds = 0;

    for (;;) {
        ++rounds;
        int changed = hook(g, label) ? 1 : 0;

        // Combine: a MIN reduction is exactly the right merge for these
        // proposals -- whichever rank found the smallest label for a vertex
        // wins, and every rank ends the round holding the same array.
        const double c0 = MPI_Wtime();
        MPI_Allreduce(label.data(), combined.data(), V, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        int any_changed = 0;
        MPI_Allreduce(&changed, &any_changed, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - c0;

        label.swap(combined);
        if (!any_changed) break;

        jump(label);
    }
    return rounds;
}

// ---------------------------------------------------------------------------
// Sequential reference: plain BFS from every unvisited vertex, taking the
// component ID to be the smallest vertex ID reached.  Since BFS is started
// from vertices in increasing order, the start vertex *is* the minimum of its
// component, so the label can be assigned directly.
// ---------------------------------------------------------------------------
void serial_components(const vector<int>& off, const vector<int>& adj,
                       int V, vector<int>& label) {
    label.assign(V, -1);
    vector<int> queue;
    queue.reserve(V);
    for (int s = 0; s < V; ++s) {
        if (label[s] != -1) continue;
        label[s] = s;
        queue.clear();
        queue.push_back(s);
        for (size_t h = 0; h < queue.size(); ++h) {
            const int u = queue[h];
            for (int e = off[u]; e < off[u + 1]; ++e) {
                const int v = adj[e];
                if (label[v] == -1) {
                    label[v] = s;
                    queue.push_back(v);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Read the whole graph on rank 0 into CSR form.
// ---------------------------------------------------------------------------
bool read_graph(const string& path, int& V, vector<int>& off,
                vector<int>& adj) {
    ifstream in(path);
    if (!in) return false;
    if (!(in >> V)) return false;
    if (V < 0) return false;

    off.assign(static_cast<size_t>(V) + 1, 0);
    adj.clear();
    for (int i = 0; i < V; ++i) {
        int k = 0;
        if (!(in >> k) || k < 0) return false;
        for (int j = 0; j < k; ++j) {
            int v = 0;
            if (!(in >> v) || v < 0 || v >= V) return false;
            adj.push_back(v);
        }
        off[i + 1] = static_cast<int>(adj.size());
    }
    return true;
}

// ---------------------------------------------------------------------------
// Random graph generator for the timing sweep: E distinct undirected edges over
// V vertices, stored in both directions so the adjacency list is symmetric.
// Fixed seed, so a run is reproducible.
// ---------------------------------------------------------------------------
void generate_graph(int V, long long E, unsigned seed, vector<int>& off,
                    vector<int>& adj) {
    mt19937 rng(seed);
    uniform_int_distribution<int> pick(0, V - 1);

    vector<int> deg(V, 0);
    vector<pair<int, int>> edges;
    edges.reserve(static_cast<size_t>(E));

    for (long long e = 0; e < E; ++e) {
        int u = pick(rng), v = pick(rng);
        if (u == v) continue;                       // no self-loops
        edges.emplace_back(u, v);
        ++deg[u];
        ++deg[v];
    }

    off.assign(static_cast<size_t>(V) + 1, 0);
    for (int v = 0; v < V; ++v) off[v + 1] = off[v] + deg[v];
    adj.assign(off[V], 0);

    vector<int> cursor(off.begin(), off.end() - 1);
    for (const auto& [u, v] : edges) {
        adj[cursor[u]++] = v;
        adj[cursor[v]++] = u;
    }
}

[[noreturn]] void usage_and_exit(int rank) {
    if (rank == 0) {
        fprintf(stderr,
            "usage: connected_components --file graph.txt [--reps R] [--verify] [--print] [--csv LABEL]\n"
            "       connected_components --gen V E [--seed S] [--reps R] [--verify] [--print] [--csv LABEL]\n");
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
    int  V = 0, reps = 1;
    long long E = 0;
    unsigned seed = 42;
    string path, csv_label;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            from_file = true;
            path = argv[++i];
        } else if (arg == "--gen" && i + 2 < argc) {
            gen = true;
            V = atoi(argv[++i]);
            E = atoll(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<unsigned>(strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--reps" && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_label = argv[++i];
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--print") {
            want_print = true;
        } else {
            usage_and_exit(rank);
        }
    }
    if (gen == from_file) usage_and_exit(rank);      // exactly one input mode
    if (reps < 1) reps = 1;

    // ---------------- rank 0 obtains the graph ----------------
    vector<int> off, adj;                       // full CSR, rank 0 only
    if (rank == 0) {
        if (from_file) {
            if (!read_graph(path, V, off, adj)) {
                fprintf(stderr, "error: cannot read graph from %s\n", path.c_str());
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
        } else {
            if (V <= 0 || E < 0) {
                fprintf(stderr, "error: V must be positive and E non-negative\n");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            generate_graph(V, E, seed, off, adj);
        }
    }

    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const VertexSplit split = make_split(V, P);

    // ---------------- distribute the adjacency lists ----------------
    // Each rank receives the degrees of its vertices, then the neighbour IDs
    // themselves.  Two Scatterv calls: one over vertices, one over edges.
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_dist0 = MPI_Wtime();

    LocalGraph g;
    g.first = split.first[rank];
    g.count = split.count[rank];

    vector<int> deg(max(1, g.count));
    {
        vector<int> all_deg;
        if (rank == 0) {
            all_deg.resize(max(1, V));
            for (int v = 0; v < V; ++v) all_deg[v] = off[v + 1] - off[v];
        }
        vector<int> vd(P), vo(P);
        for (int r = 0; r < P; ++r) { vd[r] = split.count[r]; vo[r] = split.first[r]; }
        MPI_Scatterv(rank == 0 ? all_deg.data() : nullptr, vd.data(), vo.data(), MPI_INT,
                     deg.data(), g.count, MPI_INT, 0, MPI_COMM_WORLD);
    }

    g.off.assign(static_cast<size_t>(g.count) + 1, 0);
    for (int i = 0; i < g.count; ++i) g.off[i + 1] = g.off[i] + deg[i];
    const int local_edges = g.off[g.count];
    g.adj.resize(max(1, local_edges));

    {
        // Edge counts/displacements per rank, derived on rank 0 from the CSR
        // offsets: rank r's neighbour block runs from off[first[r]] to
        // off[first[r] + count[r]].
        vector<int> ec(P, 0), eo(P, 0);
        if (rank == 0) {
            for (int r = 0; r < P; ++r) {
                const int a = split.first[r];
                const int b = a + split.count[r];
                eo[r] = off[a];
                ec[r] = off[b] - off[a];
            }
        }
        MPI_Scatterv(rank == 0 ? adj.data() : nullptr, ec.data(), eo.data(), MPI_INT,
                     g.adj.data(), local_edges, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_dist = MPI_Wtime() - t_dist0;

    // ---------------- solve ----------------
    vector<int> label;
    double t_solve = 0.0, t_comm = 0.0;
    int    rounds  = 0;

    for (int rep = 0; rep < reps; ++rep) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double s0 = MPI_Wtime();
        double comm = 0.0;
        rounds = connected_components(g, V, label, comm);
        MPI_Barrier(MPI_COMM_WORLD);
        const double s = MPI_Wtime() - s0;

        // Keep the fastest repetition, the usual convention for damping
        // transient noise on a shared cluster.
        if (rep == 0 || s < t_solve) { t_solve = s; t_comm = comm; }
    }

    // A phase is only over when its slowest rank is done.
    double local_t[3] = {t_dist, t_solve, t_comm};
    double max_t[3]   = {0.0, 0.0, 0.0};
    MPI_Reduce(local_t, max_t, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---------------- output ----------------
    int ok = 1;
    if (rank == 0) {
        if (verify) {
            vector<int> ref;
            serial_components(off, adj, V, ref);
            ok = (ref == label) ? 1 : 0;
        }

        if (!csv_label.empty()) {
            // CSV row for the benchmark driver:
            // label,V,E,P,rounds,t_dist,t_solve,t_comm,t_total,comm_pct,verify
            const long long edges = rank == 0 ? static_cast<long long>(adj.size()) / 2 : 0;
            const double total = max_t[0] + max_t[1];
            printf("%s,%d,%lld,%d,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%s\n",
                        csv_label.c_str(), V, edges, P, rounds,
                        max_t[0], max_t[1], max_t[2], total,
                        max_t[1] > 0.0 ? 100.0 * max_t[2] / max_t[1] : 0.0,
                        verify ? (ok ? "PASS" : "FAIL") : "-");
        } else if (!want_print) {
            const double total = max_t[0] + max_t[1];
            printf("V=%d E=%lld P=%d rounds=%d\n", V,
                        static_cast<long long>(adj.size()) / 2, P, rounds);
            printf("dist=%.6f s  solve=%.6f s  (comm inside solve=%.6f s)  total=%.6f s\n",
                        max_t[0], max_t[1], max_t[2], total);
            printf("comm=%.2f%% of solve\n",
                        max_t[1] > 0.0 ? 100.0 * max_t[2] / max_t[1] : 0.0);
            if (verify) printf("verify=%s\n", ok ? "PASS" : "FAIL");
        }

        // The required program output: V lines of "vertex_id component_id",
        // ascending by vertex_id, and nothing else on stdout.
        if (want_print) {
            string out;
            out.reserve(static_cast<size_t>(V) * 12);
            char buf[32];
            for (int v = 0; v < V; ++v) {
                const int len = snprintf(buf, sizeof buf, "%d %d\n", v, label[v]);
                out.append(buf, len);
            }
            fwrite(out.data(), 1, out.size(), stdout);
        }
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ok ? 0 : 1;
}
