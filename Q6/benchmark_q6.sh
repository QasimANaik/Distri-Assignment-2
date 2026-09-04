#!/bin/bash
# Timing sweep for Q6.  Emits one CSV row per (size, P) into results/.
# Columns: label,V,E,P,rounds,t_dist,t_solve,t_comm,t_total,comm_pct,verify
#
# The P=0 rows are the standalone sequential binary -- the true T_1 baseline for
# the speedup table.  (The P=1 MPI rows are the same algorithm under MPI, which
# is a different and also useful baseline, since it isolates parallel overhead
# from the algorithmic difference between BFS and label propagation.)
#
# Run after building, on a machine/allocation with at least 8 slots.

set -u
BIN=./connected_components
SEQ=./components_seq
OUT=results/q6_timings.csv
REPS=${REPS:-3}
PROCS=${PROCS:-"1 2 4 8"}
mkdir -p results

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not built."
    exit 1
fi

# Launcher.  HPC-X / OpenMPI's own mpirun is used in every case: inside a
# SLURM allocation it picks the node list up from the environment on its own.
# srun is deliberately NOT used -- it launches through SLURM's PMI, which the
# HPC-X build cannot attach to, and MPI_Init then aborts on a NULL communicator.
launch() {
    local np=$1; shift
    mpirun -np "$np" --oversubscribe "$@"
}

echo "label,V,E,P,rounds,t_dist,t_solve,t_comm,t_total,comm_pct,verify" > "$OUT"

# label          V        E        -- the four report sizes, then the shapes
#                                     that stress different parts of the method
CASES=(
  "small       10000    50000"
  "medium      50000   250000"
  "large      100000  1000000"
  "verylarge  100000  4000000"    # E above the stated cap, to show edge scaling
  "sparse     100000    50000"    # many small components
  "dense       20000  1000000"    # one giant component, high degree
  "nondiv      99991   499979"    # V divides evenly by no P in the sweep
)

for row in "${CASES[@]}"; do
    set -- $row
    label=$1; V=$2; E=$3

    if [ -x "$SEQ" ]; then
        echo "running $label V=$V E=$E sequential" >&2
        $SEQ --gen "$V" "$E" --reps "$REPS" --csv "$label" >> "$OUT"
    fi

    for P in $PROCS; do
        echo "running $label V=$V E=$E P=$P" >&2
        launch "$P" $BIN --gen "$V" "$E" --reps "$REPS" --csv "$label" >> "$OUT"
    done
done

echo >&2
echo "wrote $OUT" >&2
column -s, -t "$OUT" >&2
