#!/usr/bin/env bash
# zstar EXACT restart gate (FESOM3_RESTART_ZSTAR_EXACT=1). Under which_ALE=zstar the default restart
# set resumes INEXACTLY: the SSH stiffness values accumulate -dhe every step, dhe is lagged one step,
# and zbar_3d_n/Z_3d_n/helem/hnode_new derive from hnode, but a resume starts from the cold matrix,
# dhe = 0 and the cold geometry. The switch adds the stores ssh_stiff + dhe and rebuilds the rest
# (src/io/mod_io_restart.F90 header; fesom_lifecycle_native_mr). The gate runs the production driver
# with FESOM3_WHICH_ALE=zstar and asserts:
#
#   exact   for np in NPS, with the switch: STRAIGHT N steps vs SPLIT K steps -> checkpoint -> FRESH
#           process -> N-K steps. #1 the end-of-run checkpoints, ALL stores incl. ssh_stiff and dhe,
#           and #2 the live FESOM_DUMP_ALL node state of the last step: max|Δ| = 0.
#   cross   the np-NPW checkpoint (K steps, switch on) restored at each np in NPR with ZERO steps and
#           re-written equals the original, ssh_stiff included (partition-independent layout).
#           Cross-np EVOLUTION is not bitwise in FESOM_F at all (CG allreduce order), so a cross-np
#           case can only be the restore round trip (as tools/run_restart_gate_multirank.sh).
#   neg     the same split WITHOUT the switch (np 1) is NOT bitwise: the old behaviour, documented.
#   miss    the switch on while resuming a checkpoint written without it stops with a clear message.
#   linfs   the switch under linfs is a no-op with one root line and no extra stores.
#   badval  FESOM3_RESTART_ZSTAR_EXACT=yes stops (only 1 and 0 are accepted).
#   unset   only if BASE_BUILD is set (a build of the base commit): with the switch unset, the
#           straight N-step zstar and linfs checkpoints of this build equal the base build's
#           (same store list, max|Δ| = 0).
#
#   tools/run_restart_gate_zstar.sh [run_dir]      # run_dir must not exist (nothing is deleted)
#   env: MESH  pi (default) | core2 | a mesh directory with dist_<np>/
#        NPS   exact-case rank counts (default "1 2 8");  NPW / NPR  cross case (default 2 / "1 8")
#        N, K  straight steps and split point (default 4, 2);  F3, BUILD, PY, BASE_BUILD
#
# Measured 2026-10-24 (pi and CORE2): see TESTING.md §8.
set -uo pipefail
F3="${F3:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD="${BUILD:-$F3/build_intel_dp}"
PY="${PY:-/work/ab0995/a270088/mambaforge/bin/python3}"
MESH="${MESH:-pi}"
case "$MESH" in
    pi)    MESH_DIR=/home/a/a270088/port2/fesom2/tests/data/MESHES/pi ;;
    core2) MESH_DIR=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2 ;;
    *)     MESH_DIR="$MESH" ;;
esac
ICFILE=/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc
POOL=/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0
NCAR=/home/a/a270088/port2/fesom2/test/input/global
RUN="${1:-/scratch/${USER:0:1}/$USER/restart_gate_zstar.$(basename "$MESH").$(date +%Y%m%d-%H%M%S)}"
NPS="${NPS:-1 2 8}"
NPW="${NPW:-2}"
NPR="${NPR:-1 8}"
N="${N:-4}"
K="${K:-2}"
START="0 1 1948"

source "$F3/env.sh" intel >/dev/null 2>&1
BIN="$BUILD/bin/fesom_lifecycle_native_mr"
[ -x "$BIN" ] || { echo "missing $BIN - build first"; exit 2; }
[ -e "$RUN" ] && { echo "run dir $RUN exists - pass a new one (this gate deletes nothing)"; exit 2; }
mkdir -p "$RUN"

export FESOM3_MESH_DIR="$MESH_DIR" FESOM3_IC_FILE="$ICFILE" FESOM3_FORCING_DIR="$NCAR"
export FESOM3_RUNOFF_FILE="$POOL/CORE2_runoff.nc" FESOM3_SSS_FILE="$POOL/PHC2_salx.nc"
export FESOM3_WHICHEVP=0 FESOM3_START_CLOCK="$START"
unset FESOM3_RESTART_ZSTAR_EXACT
ulimit -s unlimited
MPIRUN=(mpirun --mca pml ob1 --mca btl self,vader --mca btl_vader_single_copy_mechanism none --oversubscribe)
GREP='RESTART RUN|state restored|registered|ZSTAR_EXACT|checkpoint written|final checkpoint'
rc=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; rc=1; }

# run one segment: seg <bin> <np> <restart_dir> <restart_in> <nsteps> <log> [dump_prefix dump_step] [-- env...]
seg() {
    local bin="$1" np="$2" rdir="$3" rin="$4" nst="$5" log="$6"; shift 6
    local -a e=( -u FESOM_DUMP_ALL -u FESOM_DUMP_FILE -u FESOM_DUMP_MINSTEP -u FESOM_DUMP_MAXSTEPS
                 FESOM3_RESTART="$rdir" FESOM3_RESTART_IN="$rin" FESOM3_RESTART_UNIT=off
                 FESOM3_RESTART_LENGTH=1 FESOM3_NSTEPS="$nst" )
    if [ $# -ge 2 ] && [ "$1" != -- ]; then
        mkdir -p "$(dirname "$1")"
        e+=( FESOM_DUMP_ALL=1 FESOM_DUMP_FILE="$1" FESOM_DUMP_MINSTEP="$2" FESOM_DUMP_MAXSTEPS="$2" )
        shift 2
    fi
    [ "${1:-}" = -- ] && shift
    env "${e[@]}" "$@" "${MPIRUN[@]}" -n "$np" "$bin" >"$log" 2>&1
    local s=$?
    grep -E "$GREP" "$log" | sort -u | sed 's/^/      /'
    return $s
}
latest() { cat "$1/restart.latest" 2>/dev/null; }
cmp_ckpt() {   # cmp_ckpt <dirA> <dirB> -> zarr_diff exit status (strict)
    "$PY" "$F3/tools/zarr_diff.py" --output-cmp "$1/$(latest "$1")" "$2/$(latest "$2")" --rel-floor 0
}
stores() { (cd "$1/$(latest "$1")" && ls -d *.zarr) | tr '\n' ' '; }

# straight + split with the given switch value; sets STRT/SPLT
split_case() {   # split_case <np> <label> <switch 0|1>
    local np="$1" label="$2" sw="$3" rem=$(( N - K ))
    STRT="$RUN/$label/np$np/straight"; SPLT="$RUN/$label/np$np/split"
    mkdir -p "$STRT" "$SPLT"
    echo "--- $label np=$np switch=$sw: straight $N steps / split $K + $rem ---"
    seg "$BIN" "$np" "$STRT" "$STRT" "$N" "$STRT.log" "$STRT/dump/node" "$N" -- FESOM3_WHICH_ALE=zstar FESOM3_RESTART_ZSTAR_EXACT="$sw"
    local s1=$?
    seg "$BIN" "$np" "$SPLT" "$SPLT" "$K" "$SPLT.seg1.log" -- FESOM3_WHICH_ALE=zstar FESOM3_RESTART_ZSTAR_EXACT="$sw"
    local s2=$?
    seg "$BIN" "$np" "$SPLT" "$SPLT" "$rem" "$SPLT.seg2.log" "$SPLT/dump/node" "$rem" -- FESOM3_WHICH_ALE=zstar FESOM3_RESTART_ZSTAR_EXACT="$sw"
    local s3=$?
    [ "$s1$s2$s3" = 000 ] || { fail "$label np=$np: a segment exited non-zero ($s1 $s2 $s3)"; return 1; }
    grep -q 'THIS IS A RESTART RUN' "$SPLT.seg2.log" || { fail "$label np=$np: seg-2 did not resume"; return 1; }
    return 0
}

echo "=== zstar exact restart gate: mesh=$MESH_DIR  N=$N K=$K  run=$RUN ==="
echo "=== build: $BUILD ==="

# ---------------------------------------------------------------- exact
for np in $NPS; do
    split_case "$np" exact 1 || continue
    for d in "$STRT" "$SPLT"; do
        s=$(stores "$d")
        [[ " $s " == *" ssh_stiff.zarr "* && " $s " == *" dhe.zarr "* ]] \
            || fail "exact np=$np: ssh_stiff/dhe missing in $d"
    done
    if cmp_ckpt "$STRT" "$SPLT"; then pass "exact np=$np #1 checkpoint max|Δ|=0 (all stores incl. ssh_stiff, dhe, ice, sigma, AB)"
    else fail "exact np=$np #1 checkpoint diverged"; fi
    if "$PY" "$F3/tools/dump_diff.py" --glob --ignore-step "$STRT/dump/node" "$SPLT/dump/node"; then
        pass "exact np=$np #2 live dump max|Δ|=0 (node dyn/tracer state, step $N)"
    else fail "exact np=$np #2 live dump diverged"; fi
done

# ---------------------------------------------------------------- cross
C="$RUN/cross/np$NPW"; mkdir -p "$C"
echo "--- cross: np=$NPW writes after $K steps (switch on) ---"
if seg "$BIN" "$NPW" "$C" "$C" "$K" "$C.log" -- FESOM3_WHICH_ALE=zstar FESOM3_RESTART_ZSTAR_EXACT=1; then
    for npr in $NPR; do
        R="$RUN/cross/np${NPW}to$npr"; mkdir -p "$R"
        echo "--- cross: np=$npr restores the np=$NPW checkpoint, 0 steps, re-writes ---"
        seg "$BIN" "$npr" "$R" "$C" 0 "$R.log" -- FESOM3_WHICH_ALE=zstar FESOM3_RESTART_ZSTAR_EXACT=1 \
            || { fail "cross np$NPW->np$npr: exit non-zero"; continue; }
        grep -q 'THIS IS A RESTART RUN' "$R.log" || { fail "cross np$NPW->np$npr: did not resume"; continue; }
        if cmp_ckpt "$C" "$R"; then pass "cross np$NPW->np$npr restore round trip max|Δ|=0 (ssh_stiff, dhe included)"
        else fail "cross np$NPW->np$npr restore round trip differs"; fi
    done
else fail "cross: np=$NPW write exited non-zero"; fi

# ---------------------------------------------------------------- neg (switch off)
if split_case 1 neg 0; then
    s=$(stores "$SPLT")
    [[ " $s " == *" ssh_stiff.zarr "* || " $s " == *" dhe.zarr "* ]] && fail "neg: ssh_stiff/dhe written without the switch"
    cmp_ckpt "$STRT" "$SPLT" >"$RUN/neg.cmp" 2>&1; s=$?
    sed 's/^/    /' "$RUN/neg.cmp"             # indented: its expected "FAIL:" line is not a gate FAIL
    if [ "$s" = 0 ]; then fail "neg: the resume WITHOUT the switch is bitwise (expected a difference)"
    else pass "neg: zstar resume without the switch is not bitwise (old behaviour): $(grep 'eta_n.zarr:eta_n' "$RUN/neg.cmp" | sed 's/^ *BAD *//')"; fi
fi

# ---------------------------------------------------------------- miss (switch on, store-less checkpoint)
M="$RUN/miss"; mkdir -p "$M/ck"
echo "--- miss: resume a checkpoint written WITHOUT the switch, switch on ---"
seg "$BIN" 1 "$M/ck" "$M/ck" "$K" "$M.seg1.log" -- FESOM3_WHICH_ALE=zstar FESOM3_RESTART_ZSTAR_EXACT=0
seg "$BIN" 1 "$M/ck" "$M/ck" 1 "$M.seg2.log" -- FESOM3_WHICH_ALE=zstar FESOM3_RESTART_ZSTAR_EXACT=1
s=$?
if [ "$s" != 0 ] && grep -q 'has no ssh_stiff/dhe stores' "$M.seg2.log"; then
    pass "miss: stops with: $(grep -m1 'has no ssh_stiff/dhe stores' "$M.seg2.log" | cut -c1-120)..."
else fail "miss: exit $s, message not found (see $M.seg2.log)"; fi

# ---------------------------------------------------------------- linfs (no-op)
L="$RUN/linfs"; mkdir -p "$L"
echo "--- linfs: switch on under linfs, 1 step ---"
if seg "$BIN" 1 "$L" "$L" 1 "$L.log" -- FESOM3_WHICH_ALE=linfs FESOM3_RESTART_ZSTAR_EXACT=1; then
    s=$(stores "$L")
    if grep -q 'FESOM3_RESTART_ZSTAR_EXACT=1 ignored, which_ALE=linfs' "$L.log" \
       && [[ " $s " != *" ssh_stiff.zarr "* && " $s " != *" dhe.zarr "* ]]; then
        pass "linfs: switch ignored with a root line, no ssh_stiff/dhe stores"
    else fail "linfs: message or store list wrong ($s)"; fi
else fail "linfs: exit non-zero"; fi

# ---------------------------------------------------------------- badval
B="$RUN/badval"; mkdir -p "$B"
seg "$BIN" 1 "$B" "$B" 1 "$B.log" -- FESOM3_WHICH_ALE=zstar FESOM3_RESTART_ZSTAR_EXACT=yes
s=$?
if [ "$s" != 0 ] && grep -q 'FESOM3_RESTART_ZSTAR_EXACT must be 1 or 0' "$B.log"; then pass "badval: =yes stops"
else fail "badval: exit $s, message not found"; fi

# ---------------------------------------------------------------- unset vs the base build
if [ -n "${BASE_BUILD:-}" ]; then
    for ale in zstar linfs; do
        A="$RUN/unset/$ale/new"; Bb="$RUN/unset/$ale/base"; mkdir -p "$A" "$Bb"
        echo "--- unset: $ale, $N steps, this build vs $BASE_BUILD (switch unset) ---"
        seg "$BIN" 1 "$A" "$A" "$N" "$A.log" -- FESOM3_WHICH_ALE=$ale
        seg "$BASE_BUILD/bin/fesom_lifecycle_native_mr" 1 "$Bb" "$Bb" "$N" "$Bb.log" -- FESOM3_WHICH_ALE=$ale
        if [ "$(stores "$A")" = "$(stores "$Bb")" ] && cmp_ckpt "$Bb" "$A"; then
            pass "unset $ale: checkpoint identical to the base build (same $(stores "$A" | wc -w) stores, max|Δ|=0)"
        else fail "unset $ale: checkpoint differs from the base build"; fi
    done
fi

echo "==================================================================================="
[ "$rc" = 0 ] && echo "run_restart_gate_zstar: GATE GREEN (mesh $MESH, exact np: $NPS; cross np$NPW -> $NPR)" \
              || echo "run_restart_gate_zstar: GATE FAILED - see FAIL lines above"
exit "$rc"
