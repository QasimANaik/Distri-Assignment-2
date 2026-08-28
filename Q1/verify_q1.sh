#!/bin/bash
# Correctness checks for Q1.  Three layers:
#   1. the two worked examples from the assignment PDF, plus m=1 / n=1 / p=1
#      edge cases, compared against expected output files;
#   2. random matrices compared against a serial multiply on rank 0 (--verify),
#      including shapes where m is not divisible by P and where P > m;
#   3. invariance -- the same input must give byte-identical C at every P.
#
# Run after building:  bash verify_q1.sh

set -u
BIN=./matmul_rowrow
IN=inputs
OUT=results
mkdir -p "$OUT"
pass=0; fail=0

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not built.  Run: mpicxx -O2 -std=c++17 -o matmul_rowrow matmul_rowrow.cpp"
    exit 1
fi

ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail+1)); }

# --- layer 1: fixed cases with known expected output ----------------------
echo "== fixed cases (PDF examples + edge cases) =="
# case name, and the process counts to try it at
run_fixed() {
    local case=$1; shift
    for P in "$@"; do
        got=$(mpirun -np "$P" --oversubscribe $BIN \
                --file "$IN/${case}_A.txt" "$IN/${case}_B.txt" --print 2>/dev/null \
              | tail -n +4)
        exp=$(tail -n +2 "$IN/${case}_C_expected.txt")
        if [ "$(echo "$got" | tr -s ' \n' ' ')" = "$(echo "$exp" | tr -s ' \n' ' ')" ]; then
            ok "$case  P=$P"
        else
            bad "$case  P=$P"
            echo "        expected: $(echo "$exp" | tr '\n' '|')"
            echo "        got:      $(echo "$got" | tr '\n' '|')"
        fi
    done
}
run_fixed ex1     1 2 3 4      # 3x2 * 2x3, P=3 is the PDF's even split
run_fixed ex2     1 2 3 4      # 4x2 * 2x2, P=3 is the PDF's uneven 2/1/1 split
run_fixed edge_m1 1 2 4        # m=1: P>m, most ranks get zero rows
run_fixed edge_n1 1 2 3        # n=1: C is an outer product
run_fixed edge_p1 1 2 3        # p=1: single-column result

# --- layer 2: random matrices vs. a serial multiply -----------------------
echo "== random matrices vs. serial reference =="
# m n p, chosen to cover divisible, non-divisible, skewed and degenerate shapes
for dims in "64 64 64" "100 40 70" "101 37 53" "1 200 200" "200 1 200" \
            "200 200 1" "3 3 3" "7 5 11" "512 128 64" "128 512 64"; do
    set -- $dims
    for P in 1 2 3 4 8; do
        if mpirun -np "$P" --oversubscribe $BIN --gen "$1" "$2" "$3" --verify \
             2>/dev/null | grep -q "verify=PASS"; then
            ok "gen $1x$2x$3  P=$P"
        else
            bad "gen $1x$2x$3  P=$P"
        fi
    done
done

# --- layer 3: result must not depend on P --------------------------------
echo "== P-invariance (same input, every P must agree) =="
for dims in "97 41 23" "256 64 32"; do
    set -- $dims
    ref=""
    for P in 1 2 3 4 8; do
        cur=$(mpirun -np "$P" --oversubscribe $BIN --gen "$1" "$2" "$3" --seed 7 --print \
                2>/dev/null | tail -n +4 | md5sum)
        if [ -z "$ref" ]; then
            ref=$cur
        elif [ "$cur" != "$ref" ]; then
            bad "invariance $1x$2x$3  P=$P differs from P=1"
            continue
        fi
        ok "invariance $1x$2x$3  P=$P"
    done
done

echo
echo "passed=$pass  failed=$fail"
[ "$fail" -eq 0 ] || exit 1
