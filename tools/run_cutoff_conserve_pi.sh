#!/usr/bin/env bash
# CONSERVING SEA-ICE CUT_OFF GATE (FESOM3_ICE_CUTOFF_CONSERVE, src/ice/mod_ice_thermo.F90).
#
# FESOM2's cut_off sets a_ice = m_ice = m_snow = 0 where a_ice < 1e-9 or m_ice < 1e-9 and drops that
# ice and snow without a flux. The ice FCT limits a_ice and m_ice separately and leaves a_ice ~ 0
# with m_ice 1e-5..2e-4 m at open-water nodes ahead of a moving ice edge, so under zstar (real
# salt flux, levitating ice) ocean + ice salt loses the deleted ice's salt: -2.3e11 psu m^3 in a
# CORE2 year. FESOM3_ICE_CUTOFF_CONSERVE=1 melts it into the ocean in the same step instead.
#
# What it runs: the production driver fesom_lifecycle_native_mr with the production physics of
# tools/prod_env.sh (SSH at the production defaults), on pi_w from a cold start on 1 June 1958
# (southern winter: the Antarctic ice edge advances and cut_off fires every step), NSTEPS steps
# (default 192 = 4 days), FESOM3_STEP_DIAG on (checkpoints every 48 steps only with REF_BIN). Under zstar the driver
# prints one "salt step" line per step (salt_budget): the ocean + ice salt content and the step's
# residual resid = change - dt*sum(areasvol*relax_salt), and cut = the ice salt the conserving
# cut_off returned.
#
#   zstar, switch on,  np 1 and 2   GATE: max |resid| per step <= TOL_STEP*content and
#                                   |sum resid| <= TOL_RUN*content; cut > 0 (cut_off fired)
#   zstar, switch off, np 1         GATE: sum resid < -TOL_RUN*content (the loss is visible);
#                                   reports the loss against the on run's returned cut
#   linfs, switch on,  np 1         runs to the end (the virtual-salt form; no salt line)
#   REF_BIN set (a build of the base commit): zstar and linfs with the switch unset, BIN against
#   REF_BIN: GATE every checkpoint file byte-identical (the switch off is the old code)
#
# Measured 2026-10-24 (Intel dp, pi_w, 192 steps): see TESTING.md (the cut_off row).
#
# Usage:  [OUT=<new dir>] [REF_BIN=<old fesom_lifecycle_native_mr>] bash tools/run_cutoff_conserve_pi.sh [NSTEPS]
# Never deletes: OUT must not exist or be empty; unset, a new cutoff_conserve.XXXXXX in the current directory
# (the ctest fesom_cutoff_conserve_pi_w runs 48 steps without REF_BIN: logs only, ~0.5 MB).
set -euo pipefail

F3="${F3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BIN="${BIN:-$F3/build_intel_dp/bin/fesom_lifecycle_native_mr}"
REF_BIN="${REF_BIN:-}"
MESH="${MESH:-/work/ab0995/a270029/meshes/pi_w}"
NSTEPS="${1:-192}"
OUT="${OUT:-$(mktemp -d "$PWD/cutoff_conserve.XXXXXX")}"
TOL_STEP="${TOL_STEP:-1e-15}"
TOL_RUN="${TOL_RUN:-1e-14}"
MPIRUN="${MPIRUN:-mpirun}"
MPIFLAGS="${MPIFLAGS:---mca pml ob1 --mca btl self,vader --mca btl_vader_single_copy_mechanism none --oversubscribe}"  # LESSONS L61

[ -x "$BIN" ] || { echo "run_cutoff_conserve_pi: missing $BIN (build first)"; exit 1; }
[ -d "$MESH" ] || { echo "run_cutoff_conserve_pi: missing mesh $MESH"; exit 1; }
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "run_cutoff_conserve_pi: $OUT is not empty; refusing"; exit 1
fi
mkdir -p "$OUT"

# shellcheck source=tools/prod_env.sh
source "$F3/tools/prod_env.sh"
prod_env 1e-5
unset FESOM3_SSH_SOLTOL FESOM3_SSH_MAXITER FESOM3_ICE_CUTOFF_CONSERVE
export FESOM3_MESH_DIR="$MESH" FESOM3_NSTEPS="$NSTEPS" FESOM3_START_CLOCK="0 152 1958" \
    FESOM3_RESTART_UNIT=s FESOM3_RESTART_LENGTH=48
echo "run_cutoff_conserve_pi: $BIN, $NSTEPS steps, runs in $OUT"

fail=0

# run <name> <np> <bin> [VAR=VALUE ...]: one run in $OUT/<name> (log; ckpt/ with REF_BIN)
run() {
    local name=$1 np=$2 bin=$3; shift 3
    local d="$OUT/$name" ck=()
    mkdir -p "$d/restart_in"
    [ -n "$REF_BIN" ] && { mkdir -p "$d/ckpt"; ck=(FESOM3_RESTART="$d/ckpt"); }
    ( cd "$d" && env "$@" "${ck[@]}" FESOM3_RESTART_IN="$d/restart_in" \
        $MPIRUN $MPIFLAGS -n "$np" "$bin" < /dev/null > "$d/log" 2>&1 )
}

# budget <name>: content, max |resid|, sum resid, sum cut, number of salt lines
budget() {
    awk '/native_mr: salt step/ {
             for (i = 1; i <= NF; i++) {
                 if ($i == "ocean+ice=") c = $(i+1)
                 if ($i == "resid=") { r = $(i+1); s += r; if (r < 0) r = -r; if (r > m) m = r }
                 if ($i == "cut=") k += $(i+1)
             }
             n++
         }
         END { printf "%.16e %.4e %.4e %.4e %d\n", c, m, s, k, n }' "$OUT/$1/log"
}

for np in 1 2; do
    name="zstar_on_np$np"
    echo "=== zstar, FESOM3_ICE_CUTOFF_CONSERVE=1, np=$np, $NSTEPS steps (closure gate) ==="
    if ! run "$name" "$np" "$BIN" FESOM3_WHICH_ALE=zstar FESOM3_ICE_CUTOFF_CONSERVE=1; then
        echo "run_cutoff_conserve_pi: FAILED (run $name)"; fail=1; continue
    fi
    read -r content maxr sumr sumcut nl <<< "$(budget "$name")"
    echo "  steps $nl  content $content  max|resid| $maxr  sum resid $sumr  returned cut $sumcut"
    if ! awk -v c="$content" -v m="$maxr" -v s="$sumr" -v k="$sumcut" -v n="$nl" -v ns="$NSTEPS" \
            -v ts="$TOL_STEP" -v tr="$TOL_RUN" \
            'BEGIN { s = (s < 0) ? -s : s; exit !(n == ns && m <= ts*c && s <= tr*c && k > 0) }'; then
        echo "run_cutoff_conserve_pi: FAILED ($name: closure or cut_off never fired)"; fail=1
    fi
done

echo "=== zstar, switch unset (FESOM2 cut_off), np=1, $NSTEPS steps (the loss) ==="
if run zstar_off_np1 1 "$BIN" FESOM3_WHICH_ALE=zstar; then
    read -r content maxr sumr sumcut nl <<< "$(budget zstar_off_np1)"
    read -r _ _ _ oncut _ <<< "$(budget zstar_on_np1)"
    echo "  steps $nl  content $content  max|resid| $maxr  sum resid $sumr  (on run returned $oncut)"
    if ! awk -v c="$content" -v s="$sumr" -v tr="$TOL_RUN" 'BEGIN { exit !(s < -tr*c) }'; then
        echo "run_cutoff_conserve_pi: FAILED (zstar_off_np1: no salt loss; does cut_off fire?)"; fail=1
    fi
else
    echo "run_cutoff_conserve_pi: FAILED (run zstar_off_np1)"; fail=1
fi

echo "=== linfs, FESOM3_ICE_CUTOFF_CONSERVE=1, np=1, $NSTEPS steps (runs; virtual-salt form) ==="
if run linfs_on_np1 1 "$BIN" FESOM3_WHICH_ALE=linfs FESOM3_ICE_CUTOFF_CONSERVE=1 \
        && grep -q 'native_mr: done' "$OUT/linfs_on_np1/log"; then
    echo "  done"
else
    echo "run_cutoff_conserve_pi: FAILED (linfs_on_np1)"; fail=1
fi

echo "=== bad value: FESOM3_ICE_CUTOFF_CONSERVE=2 must stop ==="
if run badvalue 1 "$BIN" FESOM3_WHICH_ALE=zstar FESOM3_ICE_CUTOFF_CONSERVE=2 FESOM3_NSTEPS=1; then
    echo "run_cutoff_conserve_pi: FAILED (FESOM3_ICE_CUTOFF_CONSERVE=2 accepted)"; fail=1
else
    grep -q 'FESOM3_ICE_CUTOFF_CONSERVE must be 0 or 1' "$OUT/badvalue/log" \
        && echo "  stopped" || { echo "run_cutoff_conserve_pi: FAILED (badvalue: wrong error)"; fail=1; }
fi

if [ -n "$REF_BIN" ]; then
    for ale in zstar linfs; do
        echo "=== $ale, switch unset, np=1, $NSTEPS steps: BIN vs REF_BIN checkpoints (byte-identical) ==="
        [ -d "$OUT/${ale}_off_np1" ] || run "${ale}_off_np1" 1 "$BIN" FESOM3_WHICH_ALE=$ale || fail=1
        run "${ale}_ref_np1" 1 "$REF_BIN" FESOM3_WHICH_ALE=$ale || fail=1
        nf=$(cd "$OUT/${ale}_ref_np1/ckpt" && find . -type f | wc -l)
        if [ "$nf" -gt 0 ] && diff -r -q "$OUT/${ale}_off_np1/ckpt" "$OUT/${ale}_ref_np1/ckpt" > /dev/null; then
            echo "  $nf files, none differ"
        else
            echo "run_cutoff_conserve_pi: FAILED ($ale: checkpoints differ from REF_BIN, or none)"; fail=1
        fi
    done
fi

if [ "$fail" -ne 0 ]; then
    echo "run_cutoff_conserve_pi: GATE FAILED"
    exit 1
fi
echo "run_cutoff_conserve_pi: GATE OK"
