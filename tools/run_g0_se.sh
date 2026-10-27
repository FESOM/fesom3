#!/usr/bin/env bash
# G0 BYTE-DIFF GATE for the split-explicit SSH work (docs/plans/completed/2026-10-17-split-explicit.md, Task 0.1).
# With the SE switch unset the CG path must stay BYTE-IDENTICAL to frozen main references: this script
# reruns the reference cases with a new build and compares every output at max|Δ| = 0.
#
# References: $REF/g0/<case>/, written ONCE by this script in MAKE_REF=1 mode with the frozen main
# binaries $REF/bin_main_6cd6449 (main 814e5d0 + docs; SHA256SUMS there). Cases:
#   linfs     CORE2, fesom_lifecycle_native_mr, np 2, 20 steps from a cold start, the production physics
#             of tools/prod_env.sh (prod_env 1e-5: JRA55-do from 1958-01-01, linfs, EVP, TKE, GM+Redi,
#             splines, vector-invariant momentum with rvo_upwind 1, w split, SW penetration, step
#             diagnostics; SSH CG soltol 1e-5 = production), an end-of-run checkpoint (FESOM3_RESTART,
#             unit off) and FESOM_DUMP_ALL node dumps of the final step (one step: ~0.3 GB at CORE2)
#   zstar     the same with FESOM3_WHICH_ALE=zstar
#   conserve  the log of tools/run_conserve_pi.sh (pi, fesom_conserve, 20 steps, ~25 configs). The
#             script is run from the FROZEN copy in $REF/g0/conserve/, so SE blocks added to the repo's
#             run_conserve_pi.sh later do not change the case set.
# Comparisons (each prints one PASS/FAIL line; the script exits 0 only if every case passes):
#   CORE2  env    the case's FESOM* environment (run dir masked) equals the reference's: the case
#                 definitions did not drift (a FAIL here means the gate itself changed, not the model)
#          ckpt   identical restart.latest folder and store list, zarr_diff.py --output-cmp --rel-floor 0
#                 (max|Δ| = 0 over every variable of every store), checkpoint.json equal except fesom_git
#          dump   dump_diff.py --glob --rel-floor=0: identical record keys, every record max|Δ| = 0
#          steps  the per-step FESOM3_STEP_DIAG lines (max|eta|, max|uv|, ice, Tmax/Smax, divsum, copy
#                 spreads) are textually equal
#   conserve      the log equals the reference's after dropping timing/host lines (LOGFILTER below);
#                 the exit status of run_conserve_pi.sh must match the reference's too
#
# Nothing is ever deleted: each invocation writes into a FRESH run dir (RUN, default a time-stamped
# dir under /scratch, which the scratch purge cleans) and refuses to start if it exists; MAKE_REF=1
# refuses to overwrite an existing reference case.
#
# Run it inside an allocation (no compute on the login node), e.g. -p shared --ntasks=8 --mem=60G;
# ~4.5 min for all three cases (2026-10-18: linfs 82 s, zstar 82 s, conserve 50 s, plus the diffs).
#
#   NEW_BIN=<bin dir> tools/run_g0_se.sh [case ...]          # compare; default cases: linfs zstar conserve
#   MAKE_REF=1 NEW_BIN=$REF/bin_main_6cd6449 tools/run_g0_se.sh   # write the references (once)
# Env overrides: REF (/work/ab0995/a270088/fesom3_se_ref), RUN (fresh run dir), COMPILER (intel; the
# toolchain env.sh loads for NEW_BIN), PY (python with xarray + zarr), NP (2), NSTEPS (20).
set -euo pipefail

F3="${F3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REF="${REF:-/work/ab0995/a270088/fesom3_se_ref}"
NEW_BIN="${NEW_BIN:?set NEW_BIN=<bin dir with fesom_lifecycle_native_mr and fesom_conserve>}"
MAKE_REF="${MAKE_REF:-0}"
COMPILER="${COMPILER:-intel}"
PY="${PY:-/work/ab0995/a270088/mambaforge/bin/python3}"
NP="${NP:-2}"
NSTEPS="${NSTEPS:-20}"
SSH_SOLTOL=1e-5                       # production value (= the CG default); part of the case definition
COREMESH=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
MPIRUN=(mpirun --mca pml ob1 --mca btl self,vader --mca btl_vader_single_copy_mechanism none --oversubscribe)
# conserve log: drop lines that may legitimately differ between identical runs (timings, hosts, MPI noise)
LOGFILTER='wall|elapsed|ms/step|s/step|timing|seconds|^\[[^]]*:[0-9]+\]|^--------|mpirun|hwloc|UCX'
CASES=("$@"); [ ${#CASES[@]} -eq 0 ] && CASES=(linfs zstar conserve)

NEW_BIN="$(cd "$NEW_BIN" && pwd)"
for b in fesom_lifecycle_native_mr fesom_conserve; do
    [ -x "$NEW_BIN/$b" ] || { echo "run_g0_se: missing $NEW_BIN/$b"; exit 2; }
done
if [ "$MAKE_REF" = 1 ]; then
    RUN="$REF/g0"
else
    RUN="${RUN:-/scratch/a/a270088/g0_se/$(date +%Y%m%d-%H%M%S)-$$}"
    [ -e "$RUN" ] && { echo "run_g0_se: run dir $RUN exists; pass a fresh RUN="; exit 2; }
    [ -d "$REF/g0" ] || { echo "run_g0_se: no references in $REF/g0 (MAKE_REF=1 writes them)"; exit 2; }
fi
mkdir -p "$RUN"

# the case env is exactly what this script sets: drop every FESOM* variable of the caller
while read -r v; do unset "$v"; done < <(compgen -e | grep -E '^FESOM' || true)
set +eu                               # env.sh runs `module` (not -e/-u clean); check the result instead
source "$F3/env.sh" "$COMPILER" >/dev/null 2>&1
set -eu
command -v mpirun >/dev/null || { echo "run_g0_se: env.sh $COMPILER did not provide mpirun"; exit 2; }
source "$F3/tools/prod_env.sh"
ulimit -s unlimited 2>/dev/null || true   # a batch job may cap the hard limit (see env/*/shell.intel)

echo "run_g0_se: NEW_BIN=$NEW_BIN  REF=$REF  RUN=$RUN  MAKE_REF=$MAKE_REF  cases: ${CASES[*]}"
echo "  fesom_lifecycle_native_mr sha256 $(sha256sum "$NEW_BIN/fesom_lifecycle_native_mr" | cut -c1-16)"
nfail=0
summary=()
note() { summary+=("$1"); echo "$1"; }

# one CORE2 lifecycle case: $1 = label, $2... = extra env on top of prod_env
run_core2() {
    local label="$1"; shift
    local rd="$RUN/$label" ref="$REF/g0/$label"
    if [ -e "$rd" ]; then echo "run_g0_se: $rd exists, refusing to overwrite"; return 2; fi
    mkdir -p "$rd/restart" "$rd/dump"
    local t0 rc=0
    t0=$(date +%s)
    (
        cd "$rd"
        prod_env "$SSH_SOLTOL"
        export FESOM3_MESH_DIR="$COREMESH" FESOM3_NSTEPS="$NSTEPS" \
            FESOM3_RESTART="$rd/restart" FESOM3_RESTART_IN="$rd/restart" \
            FESOM3_RESTART_UNIT=off FESOM3_RESTART_LENGTH=1 \
            FESOM_DUMP_ALL=1 FESOM_DUMP_FILE="$rd/dump/node" \
            FESOM_DUMP_MINSTEP="$NSTEPS" FESOM_DUMP_MAXSTEPS="$NSTEPS"
        for v in "$@"; do export "$v"; done
        { env | grep -E '^FESOM' | sed "s#$rd#@RUN@#g"; echo "np=$NP"; } | sort > env.txt
        sha256sum "$NEW_BIN/fesom_lifecycle_native_mr" > bin.sha256
        "${MPIRUN[@]}" -n "$NP" "$NEW_BIN/fesom_lifecycle_native_mr" > log 2>&1
    ) || rc=$?
    echo "  $label: model exit $rc, wall $(( $(date +%s) - t0 )) s"
    if [ "$MAKE_REF" = 1 ]; then
        if [ "$rc" = 0 ] && [ -s "$rd/restart/restart.latest" ]; then
            note "G0 REF  $label  written ($(cat "$rd/restart/restart.latest"), $(grep -c 'fesom_lifecycle_native_mr: step ' "$rd/log") step lines)"
        else
            note "G0 FAIL $label  reference run failed (exit $rc), see $rd/log"; nfail=$((nfail + 1))
        fi
        return 0
    fi
    local bad=()
    [ "$rc" = 0 ] || bad+=("exit=$rc")
    # env: the case definition
    diff "$ref/env.txt" "$rd/env.txt" > "$rd/env.diff" 2>&1 || bad+=("env")
    # ckpt: folder, store list, every variable max|Δ| = 0, checkpoint.json minus fesom_git
    local fr fn
    fr=$(cat "$ref/restart/restart.latest" 2>/dev/null || true)
    fn=$(cat "$rd/restart/restart.latest" 2>/dev/null || true)
    if [ -z "$fn" ] || [ "$fr" != "$fn" ]; then
        bad+=("ckpt-folder('$fr'/'$fn')")
    else
        if ! diff <(cd "$ref/restart/$fr" && ls) <(cd "$rd/restart/$fn" && ls) > "$rd/ckpt_ls.diff"; then
            bad+=("ckpt-stores")
        fi
        if ! diff <(grep -v fesom_git "$ref/restart/$fr/checkpoint.json") \
                  <(grep -v fesom_git "$rd/restart/$fn/checkpoint.json") > "$rd/ckpt_json.diff"; then
            bad+=("ckpt-json")
        fi
        "$PY" "$F3/tools/zarr_diff.py" --output-cmp "$ref/restart/$fr" "$rd/restart/$fn" --rel-floor 0 \
            > "$rd/ckpt.cmp" 2>&1 || bad+=("ckpt($(grep -c '^  BAD\|^  !' "$rd/ckpt.cmp" || true) vars)")
    fi
    # dump: final-step node dumps, record keys and values identical
    "$PY" "$F3/tools/dump_diff.py" --glob --rel-floor=0 "$ref/dump/node" "$rd/dump/node" \
        > "$rd/dump.cmp" 2>&1 || bad+=("dump")
    # steps: the per-step diagnostics
    diff <(grep 'fesom_lifecycle_native_mr: step [0-9]' "$ref/log") \
         <(grep 'fesom_lifecycle_native_mr: step [0-9]' "$rd/log") > "$rd/steps.diff" 2>&1 || bad+=("steps")
    local nst nst_ref nrec
    nst=$(grep -c 'fesom_lifecycle_native_mr: step [0-9]' "$rd/log" || true)
    nst_ref=$(grep -c 'fesom_lifecycle_native_mr: step [0-9]' "$ref/log" || true)
    [ "$nst" = "$nst_ref" ] && [ "$nst" -gt 0 ] || bad+=("nsteplines($nst/$nst_ref)")
    nrec=$(sed -n 's/^records compared: \([0-9]*\).*/\1/p' "$rd/dump.cmp")
    if [ ${#bad[@]} -eq 0 ]; then
        note "G0 PASS $label  (env; ckpt $fn: $(grep -c '^  ok ' "$rd/ckpt.cmp") vars max|Δ|=0; dump ${nrec:-?} records max|Δ|=0; $nst step lines equal)"
    else
        note "G0 FAIL $label  (${bad[*]}; see $rd/*.cmp *.diff)"; nfail=$((nfail + 1))
    fi
}

run_conserve() {
    local rd="$RUN/conserve" ref="$REF/g0/conserve"
    if [ -e "$rd" ]; then echo "run_g0_se: $rd exists, refusing to overwrite"; return 2; fi
    mkdir -p "$rd"
    local script="$ref/run_conserve_pi.sh"
    if [ "$MAKE_REF" = 1 ]; then
        cp "$F3/tools/run_conserve_pi.sh" "$rd/run_conserve_pi.sh"     # freeze the case set
        script="$rd/run_conserve_pi.sh"
    fi
    local t0 rc=0
    t0=$(date +%s)
    ( cd "$rd" && BIN="$NEW_BIN/fesom_conserve" bash "$script" > log 2>&1 ) || rc=$?
    echo "$rc" > "$rd/exit_status"
    echo "  conserve: run_conserve_pi.sh exit $rc, wall $(( $(date +%s) - t0 )) s"
    if [ "$MAKE_REF" = 1 ]; then
        note "G0 REF  conserve  written (run_conserve_pi.sh exit $rc, $(grep -c '^===' "$rd/log") configs; $(tail -1 "$rd/log"))"
        return 0
    fi
    local bad=()
    [ "$rc" = "$(cat "$ref/exit_status")" ] || bad+=("exit=$rc(ref $(cat "$ref/exit_status"))")
    diff <(grep -vE "$LOGFILTER" "$ref/log") <(grep -vE "$LOGFILTER" "$rd/log") > "$rd/log.diff" 2>&1 \
        || bad+=("log($(grep -c '^[<>]' "$rd/log.diff" || true) lines differ)")
    if [ ${#bad[@]} -eq 0 ]; then
        note "G0 PASS conserve  (log equal without timing lines: $(grep -vcE "$LOGFILTER" "$rd/log") lines, $(grep -c '^===' "$rd/log") configs; $(tail -1 "$rd/log"))"
    else
        note "G0 FAIL conserve  (${bad[*]}; see $rd/log.diff)"; nfail=$((nfail + 1))
    fi
}

for c in "${CASES[@]}"; do
    case "$c" in
        linfs)    run_core2 linfs ;;
        zstar)    run_core2 zstar FESOM3_WHICH_ALE=zstar ;;
        conserve) run_conserve ;;
        *) echo "run_g0_se: unknown case '$c' (linfs zstar conserve)"; exit 2 ;;
    esac
done

echo "==================================================================================="
printf '%s\n' "${summary[@]}"
if [ "$nfail" -ne 0 ]; then
    echo "run_g0_se: G0 FAILED ($nfail of ${#CASES[@]} cases; NEW_BIN=$NEW_BIN, RUN=$RUN)"
    exit 1
fi
echo "run_g0_se: G0 $([ "$MAKE_REF" = 1 ] && echo 'REFERENCES WRITTEN' || echo 'GREEN: byte-identical to the frozen main references') (${CASES[*]}; NEW_BIN=$NEW_BIN)"
