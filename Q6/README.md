# Q6 — Connected Components of a Large Graph

MPI/C++ implementation that labels every vertex of an undirected graph with the
**smallest vertex ID in its connected component**, with the vertices and their
adjacency lists distributed across the processes.

## The method

Labels start at `label[v] = v` and are only ever lowered, so the fixed point of
the iteration is exactly "smallest ID reachable" — which is the component ID the
assignment defines. Each round is three steps:

| Step | What happens | MPI call |
|---|---|---|
| Hooking | for each local edge `(u,v)`, push both endpoints towards `min(label[u], label[v])` | — |
| Combine | the rank that found the smallest label for a vertex wins, and every rank ends the round with the same array | `MPI_Allreduce` / `MPI_MIN` |
| Pointer jumping | `label[v] ← label[label[v]]` until stable, collapsing label chains | — |

A second one-integer `MPI_Allreduce`/`MPI_MAX` carries the "did anything change"
flag, and the loop stops when no rank changed a label.

Hooking writes **both** endpoints, not just the locally-owned one. Once the
graph is split, a rank may discover a better label for a vertex it does not own;
the `MIN` reduction is what delivers that improvement to the owner. Because
every update is a minimum, duplicated work across ranks is idempotent and an
edge seen by both of its endpoint-owners costs correctness nothing.

**Pointer jumping is what keeps the round count low.** Plain propagation moves a
label one hop per round, so a path graph would need O(diameter) rounds — 50,000
of them for a 50,000-vertex chain. Halving every chain each pass instead makes
it O(log V): the 50,000-vertex chain in `inputs/edge_chain.txt` scaled up
converges in **2 rounds**.

### Why the label array is replicated

The `V` labels are replicated on every rank; the `E` adjacency entries are not.
With `V ≤ 10⁵` the label array is 400 KB — small next to the `E ≤ 10⁶` edges
(8 MB) that stay distributed — and replication turns the combine step into one
`Allreduce` instead of a hand-rolled neighbour exchange. It also makes pointer
jumping purely local, since every rank holds the identical array and computes the
identical result. This is the deliberate trade the design rests on: it is what
makes the method simple and its communication a single collective per round, and
it is also what bounds it (see *Scaling limits*).

## Build

```bash
module load hpcx-2.7.0/hpcx-ompi        # on the RCE cluster
mpicxx -O2 -std=c++17 -o connected_components connected_components.cpp
g++    -O2 -std=c++17 -o components_seq  components_seq.cpp
```

## Run

```bash
# the assignment's sample graph, printing the required output
mpirun -np 4 ./connected_components --file inputs/sample.txt --print

# random graph, checked against a sequential BFS on rank 0
mpirun -np 4 ./connected_components --gen 100000 1000000 --verify

# CSV row for the benchmark driver
mpirun -np 8 ./connected_components --gen 100000 1000000 --reps 3 --csv large

# sequential baseline
./components_seq --file inputs/sample.txt --print
```

| Flag | Meaning |
|---|---|
| `--file graph.txt` | read the adjacency list from a file |
| `--gen V E` | generate a random graph with `V` vertices and `E` edges |
| `--seed S` | seed for `--gen` (default 42) |
| `--reps R` | repeat R times, keep the fastest (default 1) |
| `--verify` | compare against a sequential BFS on rank 0, print `PASS`/`FAIL` |
| `--print` | print the `V` result lines |
| `--csv LABEL` | emit one CSV row instead of the human-readable block |

**Input format** — first line `V`, then `V` lines `k v_1 … v_k` giving the
degree and the neighbours of vertex `i`:

```
5
1 1
1 0
2 3 4
1 2
1 2
```

**Output format** — `V` lines `vertex_id component_id`, ascending by
`vertex_id`. For the graph above:

```
0 0
1 0
2 2
3 2
4 2
```

With `--print`, that is the *only* thing on stdout — no timing block, no debug
output, as the assignment's general instructions require.

## On the cluster

```bash
sbatch run_q6.slurm      # compiles, verifies, then benchmarks
squeue -u $USER
cat q6_<JOBID>.log
```

`run_q6.slurm` asks for 4 nodes × 2 tasks = 8 slots, which covers the
`P = 1, 2, 4, 8` sweep.

## Correctness

`bash verify_q6.sh` runs four layers — **83 checks, all passing**:

1. **Fixed cases** — the sample from the assignment PDF, plus `V = 1` (with
   `P > V`, so most ranks own nothing), a graph with no edges at all (every
   vertex its own component), one long chain (the slowest-converging shape), and
   a graph with three components including two isolated vertices. Each is
   compared against a stored expected output in `inputs/`.
2. **Sequential reference** — random graphs at nine `(V, E)` shapes and five
   process counts, compared label-by-label against a BFS. Labels are integers,
   so the comparison is exact.
3. **P-invariance** — the same input must produce byte-identical output at every
   process count.
4. **Sequential-binary agreement** — the MPI output must match the standalone
   `components_seq` byte for byte, at every process count.

## Constraints handled

| Constraint | How |
|---|---|
| `V` not divisible by `P` | vertices split into contiguous blocks, the first `V % P` ranks taking one extra; `Scatterv` with computed counts and displacements |
| `P > V` | trailing ranks own zero vertices — they still join every collective, and their buffers are allocated at size 1 so the pointers stay valid |
| `E = 0` | no hooking happens, the first round reports no change, and every vertex keeps its own ID as its component |
| `V = 1` | single vertex, single component, handled by the same path |
| component ID = min vertex ID | guaranteed by construction: labels start at `v` and only ever decrease under `MIN` |
| output sorted by `vertex_id` | rank 0 holds the whole label array and emits it in index order |
| `V ≤ 10⁵`, `E ≤ 10⁶` | verified at the limits; the sweep also runs `E = 4×10⁶`, above the cap |

## Results

Times in seconds, fastest of 3 repetitions, on 8 local slots.
`P = 0` is the standalone sequential BFS — the `T₁` of the speedup table.

| Input size | `V` | `E` | seq | `P=1` | `P=2` | `P=4` | `P=8` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 10 000 | 50 000 | 0.00059 | 0.00107 | 0.00082 | 0.00091 | 0.00080 |
| Medium | 50 000 | 250 000 | 0.00287 | 0.00533 | 0.00391 | 0.00321 | 0.00329 |
| Large | 100 000 | 1 000 000 | 0.00743 | 0.01416 | 0.01049 | 0.00762 | 0.00740 |
| Very large | 100 000 | 4 000 000 | 0.02390 | 0.04052 | 0.02464 | 0.01852 | 0.02130 |

Speedup `S(P) = T_seq / T_P`, against the sequential baseline:

| Input size | `P=1` | `P=2` | `P=4` | `P=8` |
|---|:---:|:---:|:---:|:---:|
| Small | 0.55 | 0.72 | 0.64 | 0.74 |
| Medium | 0.54 | 0.73 | 0.89 | 0.87 |
| Large | 0.52 | 0.71 | 0.97 | 1.00 |
| Very large | 0.59 | 0.97 | 1.29 | 1.12 |

Communication as a share of solve time:

| Input size | `P=1` | `P=2` | `P=4` | `P=8` |
|---|:---:|:---:|:---:|:---:|
| Small | 0.3% | 11.8% | 44.0% | 48.7% |
| Medium | 0.9% | 15.9% | 27.2% | 49.4% |
| Large | 0.5% | 26.6% | 32.8% | 53.2% |
| Very large | 0.3% | 13.9% | 29.1% | 61.8% |

## Analysis

**Speedup is modest, and the reason is the algorithm, not the implementation.**
Sequential BFS visits each edge once — `O(V + E)` total. Label propagation does
`O(E/P)` work per round over `R` rounds, plus a full `Allreduce` over `V` labels
each round. The parallel version is doing strictly more total work than the
serial one; it wins only when `E/P` is large enough for that extra work to be
outweighed. At `P=1` the MPI version is consistently *slower* than the
sequential BFS (`S ≈ 0.55`), which is the honest measure of that algorithmic
overhead with parallelism contributing nothing.

**The `Allreduce` is the scaling limit.** Its cost is `O(V log P)` and does not
shrink with `P`, while the compute term is `O(E/P)` and does. Communication's
share of solve time therefore climbs steadily with `P` in every row of the
table — from under 1% at `P=1` to roughly half at `P=8` — and once it dominates,
adding ranks stops helping. That is exactly what the *Very large* row shows:
speedup peaks at `P=4` (1.29) and falls back at `P=8` (1.12), because the fifth
through eighth ranks add more `Allreduce` cost than they remove compute.

**Denser graphs parallelise better**, which follows directly: the compute term
grows with `E` while the communication term is fixed by `V`, so a higher `E/V`
ratio is a better ratio of divisible work to irreducible communication. Hence
the best speedup in the table is the *Very large* row (`E/V = 40`), and the
worst is *Small* (`E/V = 5`), where the graph is too small for parallelism to
repay its setup at any `P`.

**The sparse case is where the method genuinely struggles**, and it is worth
recording rather than hiding:

| `P` | rounds | solve (s) | comm share |
|---|---:|---:|---:|
| 1 | 11 | 0.0128 | 1.1% |
| 2 | 18 | 0.0130 | 14.0% |
| 4 | 24 | 0.0183 | 55.5% |
| 8 | 26 | 0.0209 | 74.4% |

At `V = 100 000`, `E = 50 000` the graph is mostly small scattered components,
and **the round count itself grows with `P`** — 11 rounds at `P=1`, 26 at `P=8`.
Splitting a component across more ranks means its label needs more
combine-rounds to settle, so adding processes adds both per-round cost *and*
more rounds. Runtime consequently gets steadily worse with `P`, and 74% of it is
spent in MPI. Sparse, fragmented graphs are the adversarial input for this
method; a graph with few large components (`dense`, `verylarge`) converges in 2
rounds regardless of `P`.

**Scaling limits.** Replicating the label array is what makes the method simple,
and it is also what caps it. Memory is `O(V)` per rank rather than `O(V/P)`, and
the per-round communication is `O(V)` per rank no matter how many ranks there
are. For the assignment's `V ≤ 10⁵` this is comfortably the right trade. Well
beyond it, the replication would have to give way to a distributed label store
with point-to-point exchange restricted to boundary vertices — more code, and a
worthwhile trade only once `V` is large enough that 400 KB per rank and a
full-array collective per round stop being cheap.
