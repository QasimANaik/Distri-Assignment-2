# Q1 — Distributed Matrix Multiplication (Row-Row Method)

MPI/C++ implementation of `C = A × B` where `A` is `m × n` and `B` is `n × p`,
using the Row-Row formulation.

## The method

Row `i` of `C` is a weighted sum of the **rows of `B`**, the weights being the
entries of row `i` of `A`:

```
c_i = a_i[1]·B[1,:] + a_i[2]·B[2,:] + … + a_i[n]·B[n,:]
```

Row `i` of `C` therefore depends on row `i` of `A` and on *all* of `B`, and on
nothing else. That is what the parallel decomposition rests on:

| Phase | What happens | MPI call |
|---|---|---|
| Distribution | `A` split row-wise, one contiguous block per rank; `B` sent in full to everyone | `MPI_Scatterv` + `MPI_Bcast` |
| Computation | each rank forms its own rows of `C`, with **no inter-worker communication** | — |
| Collection | rank 0 stacks the row-slices into `C` | `MPI_Gatherv` |

The collection is a **gather, not a reduce**: each rank owns a distinct set of
final rows, so nothing needs summing. (Q2's Column-Row method is the opposite —
every rank produces a full-size partial matrix and they must be summed with
`MPI_Reduce`/`MPI_SUM`.)

In the kernel the `k` loop sits outside the `j` loop, so the code scales an
entire row of `B` and accumulates it — the Row-Row formula written literally.
It also happens to be the cache-friendly traversal order for row-major storage.

## Build

```bash
module load hpcx-2.7.0/hpcx-ompi        # on the RCE cluster
mpicxx -O2 -std=c++17 -o matmul_rowrow matmul_rowrow.cpp
```

## Run

```bash
# random matrices, timing only
mpirun -np 4 ./matmul_rowrow --gen 1000 1000 1000

# random matrices, checked against a serial multiply on rank 0
mpirun -np 4 ./matmul_rowrow --gen 100 40 70 --verify

# matrices from files, printing C (use only for small cases)
mpirun -np 3 ./matmul_rowrow --file inputs/ex1_A.txt inputs/ex1_B.txt --print

# CSV row for the benchmark driver
mpirun -np 4 ./matmul_rowrow --gen 1000 1000 1000 --reps 3 --csv large
```

| Flag | Meaning |
|---|---|
| `--gen m n p` | generate random `A`, `B` (entries in `[-9, 9]`) |
| `--file A.txt B.txt` | read both matrices from files |
| `--seed S` | seed for `--gen` (default 42) |
| `--reps R` | repeat R times, keep the fastest (default 1) |
| `--verify` | compare against a serial multiply on rank 0, print `PASS`/`FAIL` |
| `--print` | print `C` |
| `--csv LABEL` | emit one CSV row instead of the human-readable block |

**Matrix file format** — first line `rows cols`, then the rows, integers
whitespace-separated:

```
3 2
1 2
0 3
-1 4
```

## On the cluster

```bash
sbatch run_q1.slurm      # compiles, verifies, then benchmarks
squeue -u $USER
cat q1_<JOBID>.log
```

`run_q1.slurm` asks for 4 nodes × 2 tasks = 8 slots, which covers the
`P = 1, 2, 4, 8` sweep.

## Correctness

`bash verify_q1.sh` runs three layers of checks:

1. **Fixed cases** — the two worked examples from the assignment PDF (`ex1`:
   3×2 · 2×3 with an even 3-way split; `ex2`: 4×2 · 2×2 with the uneven 2/1/1
   split) plus `m=1`, `n=1`, `p=1` edge cases, each compared against a stored
   expected result in `inputs/`.
2. **Serial reference** — random matrices at several shapes and process counts,
   compared element-by-element against a serial multiply. Entries are integers,
   so the comparison is exact and needs no floating-point tolerance.
3. **P-invariance** — the same input must produce a byte-identical `C` at every
   process count.

## Constraints handled

| Constraint | How |
|---|---|
| `m` not divisible by `P` | `Scatterv`/`Gatherv` with computed counts and displacements; the first `m % P` ranks take one extra row |
| `P > m` | trailing ranks get zero rows — they still join every collective, and their buffers are allocated at size 1 so the pointers stay valid |
| non-square, skewed (`m ≫ n`, `n ≫ m`) | no dimension is assumed equal to another; `tall` and `wide` cases are in the benchmark sweep |
| `m = 1`, `n = 1`, `p = 1` | covered by the fixed edge cases; `n = 1` makes `C` an outer product |
| large sizes | matrices are flat `std::vector<int>`, so the collectives operate on contiguous memory with no packing; the whole run is three collective calls |
| integer entries | `MPI_INT` throughout |

No debug output is printed — stdout carries only the requested result, as the
assignment's general instructions require.

## Timing

Each phase is timed between barriers, and the per-phase figure reported is the
**maximum across ranks** (`MPI_Reduce`/`MPI_MAX`), since a phase is only over
when its slowest rank is done. The program reports distribution, computation and
gather separately, which is what feeds the report's communication-vs-computation
metric.

Expect speedup to hold up well on the large sizes and to fall away on the small
ones: computation is `O(m·n·p / P)` and shrinks with `P`, but broadcasting `B`
costs `O(n·p)` per rank no matter how many ranks there are. Once that broadcast
dominates, adding processes stops helping.
