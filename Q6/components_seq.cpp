// Q6: Sequential reference implementation of connected components.
//
// Plain BFS from every not-yet-visited vertex.  Because the outer loop visits
// start vertices in increasing order, the first vertex reached in a component
// is the smallest ID in it, so it can be used directly as the component ID --
// which is exactly the definition the assignment requires.
//
// This exists to be the baseline the MPI version is checked and timed against
// (the T_1 in the speedup table).  The `--verify` flag inside the MPI program
// runs the same algorithm inline; this standalone binary is what makes the
// serial timing measurable on its own.
//
// Build:  g++ -O2 -std=c++17 -o components_seq components_seq.cpp
//
// Usage:
//   ./components_seq --file graph.txt [--print]
//   ./components_seq --gen V E [--seed S] [--reps R] [--csv LABEL]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace std;

namespace {

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

// Must stay identical to the generator in connected_components.cpp, so that the
// same --gen V E --seed S produces the same graph in both programs and the
// serial and MPI timings are measured on the same input.
void generate_graph(int V, long long E, unsigned seed, vector<int>& off,
                    vector<int>& adj) {
    mt19937 rng(seed);
    uniform_int_distribution<int> pick(0, V - 1);

    vector<int> deg(V, 0);
    vector<pair<int, int>> edges;
    edges.reserve(static_cast<size_t>(E));

    for (long long e = 0; e < E; ++e) {
        int u = pick(rng), v = pick(rng);
        if (u == v) continue;
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

}  // namespace

int main(int argc, char** argv) {
    bool gen = false, from_file = false, want_print = false;
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
        } else if (arg == "--print") {
            want_print = true;
        } else {
            fprintf(stderr,
                "usage: components_seq --file graph.txt [--print]\n"
                "       components_seq --gen V E [--seed S] [--reps R] [--csv LABEL]\n");
            return 2;
        }
    }
    if (gen == from_file) {
        fprintf(stderr, "error: give exactly one of --file or --gen\n");
        return 2;
    }
    if (reps < 1) reps = 1;

    vector<int> off, adj;
    if (from_file) {
        if (!read_graph(path, V, off, adj)) {
            fprintf(stderr, "error: cannot read graph from %s\n", path.c_str());
            return 3;
        }
    } else {
        if (V <= 0 || E < 0) {
            fprintf(stderr, "error: V must be positive and E non-negative\n");
            return 3;
        }
        generate_graph(V, E, seed, off, adj);
    }

    vector<int> label;
    double best = 0.0;
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = chrono::steady_clock::now();
        serial_components(off, adj, V, label);
        const double s = chrono::duration<double>(
                             chrono::steady_clock::now() - t0).count();
        if (rep == 0 || s < best) best = s;
    }

    if (!csv_label.empty()) {
        // label,V,E,P,rounds,t_dist,t_solve,t_comm,t_total,comm_pct,verify
        // P=0 marks the sequential baseline; there is no distribution phase and
        // no communication, so those columns are zero.
        printf("%s,%d,%lld,0,0,0.000000,%.6f,0.000000,%.6f,0.00,-\n",
                    csv_label.c_str(), V,
                    static_cast<long long>(adj.size()) / 2, best, best);
    } else if (!want_print) {
        printf("V=%d E=%lld sequential\n", V,
                    static_cast<long long>(adj.size()) / 2);
        printf("solve=%.6f s\n", best);
    }

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
    return 0;
}
