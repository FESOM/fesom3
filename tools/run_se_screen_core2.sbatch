#!/bin/bash
# Split-explicit 3000-step physics SCREENS (docs/plans/completed/2026-10-17-split-explicit.md, Task 5.1): CORE2 with
# the production physics of tools/prod_env.sh (prod_env 1e-5: PHC winter IC, JRA55-do from 1958-01-01, EVP,
# TKE, GM + Redi, splines, vector-invariant momentum with rvo_upwind 1, w split, SW penetration, STEP_DIAG
# on), Δt 1800 s, a cold start, one compute node, np 128, the cases of $CASES one after the other:
#   si     the semi-implicit CG solve, linfs (the baseline)
#   se_a   SE at the defaults (chi 0.25 in Task 5.1, 0.1 since Task 5.8; M 50, se_visc_ah 0.01), linfs, SE_CHECK=1
#   se_b   SE with M = M_min of the CFL guard (FESOM3_SE_M=48 at chi 0.25 on CORE2; at the default chi 0.1 that
#          is M = 48 above M_min 40: set FESOM3_SE_CHI=0.25 to repeat the Task 5.1 case), linfs, SE_CHECK
#   se_c   SE without the barotropic viscosity (FESOM3_SE_VISC_AH=0), linfs, SE_CHECK
#   se_z   SE at the defaults under zstar (the zstar headline), SE_CHECK (+ SE_CHECK_TAIL)
#   si_z   the semi-implicit CG solve under zstar (the baseline of se_z)
# SE_CHECK costs 2.8 % of the loop at np 128 (job 28218251: 150.3 vs 146.2 ms/step), so it stays on.
# Binaries: a FROZEN copy (BIN, a dir with fesom_lifecycle_native_mr and SHA256SUMS), checked before the
# first case, so a rebuild of the tree cannot change them under a running job.
# Per case: run dir $RUN/core2_<case> (log, env.txt, status); the summary below prints the guard line, the
# max|eta| series per 250-step window (the value at the window's last step, the window max), the cold-start
# peak (max over steps 1-96, the first two days), the run maxima of max|eta|, max|uv|, divsum, max CFL_z,
# the SE_CHECK maxima, NaN / non-finite lines, wall time and exit status, and the TIMING loop / SSH lines.
# Nothing is ever deleted: RUN must not exist.
#   BIN=<frozen bin dir> CASES="si se_a se_b se_c" sbatch -J s5_A_screen_linfs tools/run_se_screen_core2.sbatch
# Env overrides: BIN (required), CASES (default "si se_a"), NS (3000), RUN (/scratch/a/a270088/se_s5a/
# screen_<jobid>).
#SBATCH -J se_screen
#SBATCH -A ab0995
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH -o /work/ab0995/a270088/fesom3_se_ref/jobs/%x.%j.out
set -u
F3=/work/ab0995/a270088/fesom3_se
BIN="${BIN:?set BIN=<frozen bin dir with SHA256SUMS>}"
CASES="${CASES:-si se_a}"
NS="${NS:-3000}"
RUN="${RUN:-/scratch/a/a270088/se_s5a/screen_${SLURM_JOB_ID}}"
COREMESH=/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2
NP=128
MPIRUN=(mpirun --mca pml ob1 --mca btl self,vader --mca btl_vader_single_copy_mechanism none)
[ -e "$RUN" ] && { echo "ABORT: $RUN exists"; exit 1; }
mkdir -p "$RUN"
cd "$F3"
echo "host $(hostname)  HEAD $(git rev-parse --short HEAD)  BIN $BIN  CASES $CASES  NS $NS  RUN $RUN  $(date)"
( cd "$BIN" && sha256sum -c --quiet SHA256SUMS ) || { echo "ABORT: frozen binaries changed"; exit 1; }
sha256sum "$BIN/fesom_lifecycle_native_mr" | cut -c1-16
while read -r v; do unset "$v"; done < <(compgen -e | grep -E '^FESOM' || true)
set +u; source "$F3/env.sh" intel >/dev/null 2>&1; set -u
source "$F3/tools/prod_env.sh"
ulimit -s unlimited

case_env() {   # the case's env assignments on top of prod_env 1e-5
    case "$1" in
        si)   echo "FESOM3_WHICH_ALE=linfs" ;;
        se_a) echo "FESOM3_WHICH_ALE=linfs FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1" ;;
        se_b) echo "FESOM3_WHICH_ALE=linfs FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1 FESOM3_SE_M=48" ;;
        se_c) echo "FESOM3_WHICH_ALE=linfs FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1 FESOM3_SE_VISC_AH=0" ;;
        se_z) echo "FESOM3_WHICH_ALE=zstar FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1" ;;
        si_z) echo "FESOM3_WHICH_ALE=zstar" ;;
        *)    return 1 ;;
    esac
}
for c in $CASES; do case_env "$c" > /dev/null || { echo "ABORT: unknown case $c"; exit 1; }; done

core2_case() {   # core2_case <label>
    local label=$1 rd=$RUN/core2_$1 t0=$(date +%s) rc=0
    mkdir -p "$rd"
    ( cd "$rd"
      prod_env 1e-5
      export FESOM3_MESH_DIR=$COREMESH FESOM3_NSTEPS=$NS
      for v in $(case_env "$label"); do export "$v"; done
      { env | grep -E '^FESOM' | sort; echo "np=$NP"; } > env.txt
      "${MPIRUN[@]}" -n "$NP" "$BIN/fesom_lifecycle_native_mr" < /dev/null > log 2>&1 ) || rc=$?
    echo "core2 $label np $NP $NS steps: exit $rc, wall $(( $(date +%s) - t0 )) s" > "$rd/status"
    cat "$rd/status"
}
for c in $CASES; do core2_case "$c"; done

series() {   # max|eta| per 250-step window: the value at the window's last step and the window max (+ max|uv|, CFL_z)
    awk '/fesom_lifecycle_native_mr: step [0-9]+ / { s = $3 + 0; w = int((s - 1)/250)
             for (i = 4; i < NF; i++) { if ($i == "max|eta|=") e = $(i+1) + 0; if ($i == "max|uv|=") u = $(i+1) + 0 }
             if (!(e <= we[w])) { we[w] = e; ws[w] = s }; if (!(u <= wu[w])) wu[w] = u; ee[w] = e; ls[w] = s; if (w > nw) nw = w }
         /fesom_lifecycle_native_mr: CFL_z step / { w = int(($4 - 1)/250); v = $7 + 0; if (!(v <= wc[w])) wc[w] = v }
         END { printf "  window     at-end max|eta|   window max|eta| (step)   window max|uv|   window max CFL_z\n"
               for (w = 0; w <= nw; w++) if (w in ls)
                   printf "  %4d-%-4d  %10.4f m      %10.4f m (%4d)      %8.4f m/s     %8.4f\n", 250*w + 1, ls[w], ee[w], we[w], ws[w], wu[w], wc[w] }' "$1"
}
stepmax() {   # run maxima of the STEP_DIAG fields, the cold-start peak and max CFL_z
    awk '/fesom_lifecycle_native_mr: step [0-9]+ / { n++; s = $3 + 0
             for (i = 4; i < NF; i++) if ($i ~ /=$/) { v = $(i+1) + 0; if (!(v <= mx[$i])) { mx[$i] = v; at[$i] = s } }
             for (i = 4; i < NF; i++) if ($i == "max|eta|=" && s <= 96) { v = $(i+1) + 0; if (!(v <= cp)) { cp = v; cps = s } }
             last = $0 }
         /fesom_lifecycle_native_mr: CFL_z step / { m++; v = $7 + 0; if (!(v <= cmax)) { cmax = v; cat_ = $4 } }
         END { printf "  %d step lines: max|eta|=%.4e (step %s) cold-start peak (steps 1-96)=%.4e (step %s) max|uv|=%.4e (step %s)\n",
                      n, mx["max|eta|="], at["max|eta|="], cp, cps, mx["max|uv|="], at["max|uv|="]
               printf "  divsum=%.3e (step %s) dUV=%.3e dAB=%.3e dsig=%.3e Tmax=%.4e Smax=%.4e m_ice=%.4e\n",
                      mx["divsum="], at["divsum="], mx["dUV="], mx["dAB="], mx["dsig="], mx["Tmax="], mx["Smax="], mx["m_ice="]
               printf "  %d CFL_z lines: max CFL_z=%.4e (step %s)\n", m, cmax, cat_
               print "  last: " last }' "$1"
}
semax() {   # run maxima of the SE_CHECK / SE_CHECK_TAIL tokens (|value|), non-finite tokens counted
    awk '/^SE_CHECK / { n++; for (i = 3; i <= NF; i++) { split($i, kv, "="); if (kv[2] ~ /[Nn]a[Nn]|[Ii]nf/) bad++
                                                         v = kv[2] + 0; if (v < 0) v = -v
                                                         if (!(v <= mx[kv[1]])) { mx[kv[1]] = v; at[kv[1]] = $2 } } }
         /^SE_CHECK_TAIL / { t++; for (i = 3; i <= NF; i++) { split($i, kv, "="); if (kv[2] ~ /[Nn]a[Nn]|[Ii]nf/) bad++
                                                         v = kv[2] + 0; if (v < 0) v = -v
                                                         if (!(v <= tx[kv[1]])) tx[kv[1]] = v } }
         END { printf "  SE_CHECK %d lines; max etacomp=%.3e (%s) trim=%.3e (%s) |msl|=%.3e |fw|=%.3e spreadU=%.3e spreadUm=%.3e maxeta=%.4e (%s) maxU=%.4e; non-finite tokens %d\n",
                      n, mx["etacomp"], at["etacomp"], mx["trim"], at["trim"], mx["msl"], mx["fw"], mx["spreadU"], mx["spreadUm"],
                      mx["maxeta"], at["maxeta"], mx["maxU"], bad
               if (t > 0) printf "  SE_CHECK_TAIL %d lines; max laysum=%.3e w1=%.3e hlaw=%.3e\n", t, tx["laysum"], tx["w1"], tx["hlaw"] }' "$1"
}
for c in $CASES; do
    rd=$RUN/core2_$c
    echo "==== CORE2 np $NP $c ($(case_env "$c")): $(cat "$rd/status")"
    grep -E "oce_ssh_se:   (M =|CFL|WARNING)" "$rd/log" | head -3
    stepmax "$rd/log"
    case "$c" in se_*) semax "$rd/log" ;; esac
    series "$rd/log"
    grep -E "LOOP TOTAL|dynamics ssh|SE barotropic|ssh solve" "$rd/log" | tail -4
    echo "  NaN / non-finite / error lines: $(grep -acE "NaN|Infinity|non-finite|rror|VIOLATED|overflow|signal" "$rd/log")"
    grep -aE "NaN|Infinity|non-finite|rror|VIOLATED|overflow|signal" "$rd/log" | sort | uniq -c | head -5
done
echo "done $(date)"
