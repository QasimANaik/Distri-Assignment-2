#!/bin/bash
# Timing sweep for Q1.  Emits one CSV row per (size, P) into results/.
# Columns: label,m,n,p,P,t_dist,t_comp,t_gather,t_total,comm_pct,verify
#
# Run after building, on a machine/allocation with at least 8 slots.

set -u
BIN=./matmul_rowrow
OUT=results/q1_timings.csv
REPS=${REPS:-3}
PROCS=${PROCS:-"1 2 4 8"}
mkdir -p results

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not built."
    exit 1
fi

# Launcher: srun inside a SLURM allocation, mpirun otherwise.
launch() {
    local np=$1; shift
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --ntasks="$np" --cpu-bind=cores "$@"
    else
        mpirun -np "$np" --oversubscribe "$@"
    fi
}

echo "label,m,n,p,P,t_dist,t_comp,t_gather,t_total,comm_pct,verify" > "$OUT"

# label            m     n     p       -- the four report sizes, then the
#                                         shape cases the PDF asks for
CASES=(
  "small       200   200   200"
  "medium      600   600   600"
  "large      1000  1000  1000"
  "verylarge  2000  2000  2000"
  "tall       8000    64    64"     # m >> n
  "wide         64  8000    64"     # n >> m
  "nondiv     1001   997   503"     # m not divisible by any P in the sweep
)

for row in "${CASES[@]}"; do
    set -- $row
    label=$1; m=$2; n=$3; p=$4
    for P in $PROCS; do
        echo "running $label ${m}x${n}x${p} P=$P" >&2
        launch "$P" $BIN --gen "$m" "$n" "$p" --reps "$REPS" --csv "$label" >> "$OUT"
    done
done

echo >&2
echo "wrote $OUT" >&2
column -s, -t "$OUT" >&2
