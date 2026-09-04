#!/bin/bash
# Correctness checks for Q6.  Four layers:
#   1. the sample from the assignment PDF, plus structural edge cases
#      (V=1, no edges at all, one long chain, several components), compared
#      against expected output files;
#   2. random graphs compared against a sequential BFS on rank 0 (--verify),
#      including V not divisible by P and P > V;
#   3. invariance -- the same input must give byte-identical labels at every P;
#   4. agreement with the standalone sequential binary, byte for byte.
#
# Run after building:  bash verify_q6.sh

set -u
BIN=./connected_components
SEQ=./components_seq
IN=inputs
pass=0; fail=0

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not built.  Run: mpicxx -O2 -std=c++17 -o connected_components connected_components.cpp"
    exit 1
fi

ok()  { echo "  PASS  $1"; pass=$((pass+1)); }
bad() { echo "  FAIL  $1"; fail=$((fail+1)); }

# --- layer 1: fixed cases with known expected output ----------------------
echo "== fixed cases (PDF sample + edge cases) =="
run_fixed() {
    local case=$1; shift
    for P in "$@"; do
        got=$(mpirun -np "$P" --oversubscribe $BIN --file "$IN/${case}.txt" --print 2>/dev/null)
        exp=$(cat "$IN/${case}_expected.txt")
        if [ "$got" = "$exp" ]; then
            ok "$case  P=$P"
        else
            bad "$case  P=$P"
            echo "        expected: $(echo "$exp" | tr '\n' '|')"
            echo "        got:      $(echo "$got" | tr '\n' '|')"
        fi
    done
}
run_fixed sample        1 2 3 4 8   # the PDF's 5-vertex sample
run_fixed edge_single   1 2 4       # V=1: P>V, most ranks own nothing
run_fixed edge_noedges  1 2 4 8     # E=0: every vertex is its own component
run_fixed edge_chain    1 2 4 8     # one path -- the slowest-converging shape
run_fixed edge_multi    1 2 3 4     # three components incl. two isolated vertices

# --- layer 2: random graphs vs. the sequential BFS ------------------------
echo "== random graphs vs. sequential reference =="
# V E, chosen to cover sparse (many components), dense (one giant component),
# and sizes that do not divide evenly by the process counts
for dims in "1000 500" "1000 5000" "5000 2000" "5000 20000" "10007 30011" \
            "50000 200000" "100 50" "97 41" "20000 10000"; do
    set -- $dims
    for P in 1 2 3 4 8; do
        if mpirun -np "$P" --oversubscribe $BIN --gen "$1" "$2" --verify \
             2>/dev/null | grep -q "verify=PASS"; then
            ok "gen V=$1 E=$2  P=$P"
        else
            bad "gen V=$1 E=$2  P=$P"
        fi
    done
done

# --- layer 3: result must not depend on P --------------------------------
echo "== P-invariance (same input, every P must agree) =="
for dims in "9973 20011" "30000 60000"; do
    set -- $dims
    ref=""
    for P in 1 2 3 4 8; do
        cur=$(mpirun -np "$P" --oversubscribe $BIN --gen "$1" "$2" --seed 7 --print \
                2>/dev/null | md5sum)
        if [ -z "$ref" ]; then
            ref=$cur
        elif [ "$cur" != "$ref" ]; then
            bad "invariance V=$1 E=$2  P=$P differs from P=1"
            continue
        fi
        ok "invariance V=$1 E=$2  P=$P"
    done
done

# --- layer 4: MPI output must equal the standalone sequential binary ------
if [ -x "$SEQ" ]; then
    echo "== MPI vs. standalone sequential binary =="
    for dims in "5000 12000" "20000 40000"; do
        set -- $dims
        exp=$($SEQ --gen "$1" "$2" --seed 11 --print 2>/dev/null | md5sum)
        for P in 1 2 4 8; do
            got=$(mpirun -np "$P" --oversubscribe $BIN --gen "$1" "$2" --seed 11 --print \
                    2>/dev/null | md5sum)
            if [ "$got" = "$exp" ]; then
                ok "seq-match V=$1 E=$2  P=$P"
            else
                bad "seq-match V=$1 E=$2  P=$P"
            fi
        done
    done
else
    echo "== MPI vs. standalone sequential binary == (skipped: $SEQ not built)"
fi

echo
echo "passed=$pass  failed=$fail"
[ "$fail" -eq 0 ] || exit 1
