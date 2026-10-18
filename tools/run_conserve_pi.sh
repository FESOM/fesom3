#!/usr/bin/env bash
# CONSERVATION + NO-LEAKAGE GATE for the bottom-at-vertices change (docs/plans/completed/
# 20260910-fesom3-bottom-at-vertices.md, Task 2). This is the numerical regression net
# that replaces the retired FESOM2 byte-gates: it is FESOM_F-internal, needs no oracle,
# and must stay green through every task of that plan.
#
# What it checks, via src/drivers/fesom_conserve.F90:
#   zstar + Redi, np 1 -> the Redi path. diff_ver_part_redi_expl is the ONLY caller of
#       the tr_xynodes node-average whose denominator the depth-independent area changed,
#       and it never runs unless Redi is on. Note that conservation alone does NOT validate
#       that denominator -- the Redi tendency is a telescoping flux divergence, so it
#       conserves whatever tr_xynodes contains. The denominator is pinned by test_bottom's
#       'Redi: node-averaged gradient is exact on the WET area' check instead.
#   zstar, np 1 and 2 -> HARD conservation gate. Total heat and salt content must not
#       drift beyond TOL. This is the sharp test of the bottom change: if the flux areas
#       and the cell volumes ever disagree -- the central risk when area() becomes
#       depth-independent and nlevels() flips to vertex-defined -- it shows up here as
#       drift long before it shows up anywhere else.
#   linfs, np 1 -> INVARIANTS ONLY (T10 velocity leakage, helem == mean(hnode),
#       zbar_e_bot, finiteness). Drift is reported but NOT gated, because the linear free
#       surface is not tracer-conserving by construction: hnode is frozen, so the surface
#       vertical advective flux -w*T*area at nzmin is a real source/sink with no thickness
#       change to balance it. Measured baseline on pi over 20 steps: heat -2.2e-04,
#       salt -1.2e-06, non-monotone (a free-surface adjustment transient).
#
#   zstar + wsplit, np 1 and 8 (+ momadv-vinv-wsplit and GM+wsplit, np 1) -> the smooth
#       w = w_e + w_i split (use_wsplit, oce_wsplit) with ALL its consumers live: the
#       momentum TDMA (w_i, flux form at momadv_opt=2 / advective form at 1), momentum
#       advection (w_e), the tracer-diffusion TDMA (do_wimpl, w_i) and the FCT
#       adv_tra_vert_impl (w_i, then the LO flux on the full w); GM+wsplit adds the bolus
#       fer_w, which is added to w AND w_e at owned+halo right where the split's halo
#       contract lives and bypasses the cap. The 20-step cold start never reaches
#       CFL_z ~ 1 (measured last-step max 2.76e-2 on pi, split off), so the cap is set
#       to FESOM3_WSPLIT_MAXCFL=0.005 (onset 0.0025 = the driver's 0.5*maxcfl default;
#       the triple is WSPLIT_CFG below, the same as test/CMakeLists.txt's _wsplit_cfg):
#       measured 8632 owned faces with w_i /= 0, 500 with f = |w_i|/|w| >= 0.5, max f
#       0.82 (np 1/2 identical to +-1 face). FESOM3_WSPLIT_EXPECT_SPLIT=1000 makes the
#       driver error-stop unless >= 1000 faces split and one has f >= 0.5, so these
#       configs cannot pass vacuously. The driver prints the split statistics line for
#       EVERY config (the line before the drift), which is why every tail below is one
#       line longer than the drift block alone.
#
#   linfs + sea ice, np 2 and 8 -> SHARED-ELEMENT gate (docs/plans/completed/20261012-shared-element-sync.md).
#       A border element is owned by every rank that owns one of its vertices; update_vel syncs
#       uv to the designated owner's copy. FESOM3_DIVSUM_TOL=1e-12 makes the driver stop unless
#       the divergence-sum ratio |sum ssh_rhs_old| / sum|ssh_rhs_old| stays below 1e-12 in every
#       step (a linfs mass leak raises it) and the copy spreads of uv, uv_rhsAB and sigma are 0.
#       Vector-invariant momentum with rvo_upwind 1 (the CORE2 leak's trigger) at np 2 and 8,
#       the scalar advection (momadv 2, the default) at np 2. The same vinv case with
#       FESOM3_SHARED_SYNC=0 is report-only: it shows whether the copies drift without the sync.
#       Measured 2026-10-12 (pi, 20 steps, cold start): max ratio 1.6e-16 (vinv np 2), 3.1e-16
#       (vinv np 8), 1.8e-16 (scalar np 2), every spread 0. Sync off: uv spread 6.9e-18 -- the
#       roundoff drift -- and uv_rhsAB spread 5.8e-11, i.e. the rvo_upwind blend has amplified it
#       by ~1e7 within 20 steps (sigma 0; the ratio stays 1.6e-16 this early at these small
#       velocities). The other configs report ratios <= 3.8e-16 and zero spreads.
#   Every driver run prints one shared-element line after the drift block, so every tail
#   below is one line longer than before.
#
#   linfs + split-explicit (SE), np 1 -> the split-explicit barotropic solver (FESOM3_SSH_SE=1,
#       docs/plans/2026-10-17-split-explicit.md, Tasks 1.3-1.5) with FESOM3_SE_CHECK=1. The driver's
#       linfs invariants as above (drift reported, not gated) plus the per-step SE_CHECK line of
#       src/oce/oce_ssh_se.F90, whose run maxima are gated: eta-compatibility (etacomp, m), trim
#       (m^2/s) and |mean-sea-level change| (msl, m) below SE_TOL_* = the Task 1.5 levels x 100,
#       the copy spreads of U and <<U>> exactly 0, and one SE_CHECK line per step. Two configs: the
#       default physics, and vector-invariant momentum with rvo_upwind 1 + sea ice (the SHARED
#       configuration below). The np 2/8 SE configs come with Task 2.2. Measured 2026-10-18 (pi,
#       1000 steps, np 1, M = 50; Task 1.5, job 28217370): max etacomp 5.0e-15 m, trim 5.7e-13 m^2/s,
#       |msl| 3.5e-16 m (default; vinv+ice 4.8e-15, 4.5e-13, 2.9e-16), spreads 0. Heat/salt drift
#       -3.68e-4/-1.59e-6 (SI linfs: -3.69e-4/-1.60e-6).
#   The SE blocks run at the solver's DEFAULT knobs (no FESOM3_SE_CHI etc.), so since 2026-10-18 (plan Task 5.0)
#   at chi = 0.25 instead of 0.1; the levels quoted here are the chi = 0.1 ones, and the chi = 0.25 20-step maxima
#   are at the same level (job 28220536: etacomp <= 2.1e-15 m, trim <= 2.3e-13 m^2/s, |msl| <= 1.3e-16 m).
#
#   zstar + split-explicit (SE), np 1/2/8 -> the split-explicit solver under zstar (Task 3.1/3.2): the HARD
#       conservation gate of the zstar cases (FESOM3_CONSERVE_TOL=$TOL; heat, salt and volume) plus the SE_CHECK
#       gate of the linfs SE block (etacomp, trim, msl net of the freshwater -- 0 here, the run is unforced --,
#       spreads exactly 0) and the run maxima of the SE_CHECK_TAIL line the zstar run prints after the ALE tail:
#       the layer sum max|sum hnode - (H0 + eta)| (laysum, m), max|w(1)| (w1, m/s, 0 to rounding under SE) and the
#       depth law max|sum helem^{n+1} - H_e(eta^{n+1})| (hlaw, m), each below SE_Z_TOL_* = the Task 3.2 levels
#       x 100; one SE_CHECK and one SE_CHECK_TAIL line per step. Configs: default np 1/2/8, vector-invariant
#       momentum with rvo_upwind 1 + sea ice np 1/8, GM + Redi + TKE + w split np 1. Measured 2026-10-18 (pi,
#       1000 steps, M = 50, chi = 0.1; Task 3.2, job 28218186), max over the six configs: etacomp 4.9e-15 m, trim
#       5.7e-13 m^2/s, |msl| 6.9e-16 m, laysum 1.8e-11 m, w1 3.1e-18 m/s, hlaw 7.3e-12 m, spreads 0; heat/salt/volume
#       drift <= 6e-15 / 1.6e-14 / 4.3e-14 (SI zstar: 1.1e-14 / 1.0e-14 / 1.4e-14).
#
# Measured zstar baseline on pi, 20 steps, before any bottom change:
#   np=1  heat -4.87e-15  salt -1.66e-14
#   np=2  heat  0.00e+00  salt -7.80e-15
# TOL=1e-12 leaves roughly two orders of headroom over that.
#
# Usage:  bash tools/run_conserve_pi.sh [NSTEPS]
set -euo pipefail

F3="${F3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BIN="${BIN:-$F3/build_intel_dp/bin/fesom_conserve}"
PIMESH="${PIMESH:-/home/a/a270088/port2/fesom2/tests/data/MESHES/pi}"
ICFILE="${ICFILE:-/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc}"
NSTEPS="${1:-20}"
TOL="${TOL:-1e-12}"
MPIFLAGS="${MPIFLAGS:---mca pml ob1 --mca btl self,vader --mca btl_vader_single_copy_mechanism none --oversubscribe}"  # LESSONS L61
# the tuned wsplit triple (header); re-tune the cap here AND in test/CMakeLists.txt
WSPLIT_CFG="${WSPLIT_CFG:-FESOM3_WSPLIT=1 FESOM3_WSPLIT_MAXCFL=0.005 FESOM3_WSPLIT_EXPECT_SPLIT=1000}"

[ -x "$BIN" ] || { echo "run_conserve_pi: missing $BIN (build first)"; exit 1; }
[ -d "$PIMESH" ] || { echo "run_conserve_pi: missing pi mesh $PIMESH"; exit 1; }

export FESOM3_MESH_DIR="$PIMESH" FESOM3_IC_FILE="$ICFILE" FESOM3_NSTEPS="$NSTEPS"

fail=0

# np=8 matters beyond conservation: at np=1 nNodL == mesh%nod2D, so a routine that
# over-runs a LOCAL-sized array using a dummy declared with the GLOBAL nod2D is exactly
# in-bounds and cannot fail. Only a rank count where nNodL << nod2D exposes it.
for np in 1 2 8; do
    echo "=== zstar, np=$np, $NSTEPS steps (conservation gate, tol=$TOL) ==="
    if FESOM3_WHICH_ALE=zstar FESOM3_CONSERVE_TOL="$TOL" \
         mpirun $MPIFLAGS -n "$np" "$BIN" 2>&1 | tail -8; then
        :
    else
        echo "run_conserve_pi: FAILED (zstar np=$np)"; fail=1
    fi
done

# Physics combinations. The area/areasvol consumers are spread across the vertical
# diffusion TDMA, the Redi isoneutral flux and the KPP non-local / shortwave terms, and
# each is only reachable with its own scheme switched on -- a gate that runs only the
# default configuration proves nothing about the others.
# NOTE read from fd 3: mpirun inherits stdin and CONSUMES the heredoc, so a plain
# `while read ... done <<CFG` silently runs only the FIRST config and drops the rest.
# The heredoc is unquoted so $WSPLIT_CFG expands (no other line contains $ or `).
while read -r -u 3 label vars; do
    [ -z "$label" ] && continue
    echo "=== zstar + $label, np=1, $NSTEPS steps (conservation gate, tol=$TOL) ==="
    if env $vars FESOM3_WHICH_ALE=zstar FESOM3_CONSERVE_TOL="$TOL" \
         mpirun $MPIFLAGS -n 1 "$BIN" 2>&1 | tail -9; then
        :
    else
        echo "run_conserve_pi: FAILED (zstar+$label np=1)"; fail=1
    fi
done 3<<CFG
Redi FESOM3_REDI=1
KPP FESOM3_MIX_KPP=1
TKE FESOM3_MIX_TKE=1
GM+Redi+TKE FESOM3_FER_GM=1 FESOM3_REDI=1 FESOM3_MIX_TKE=1
GM+KPP FESOM3_FER_GM=1 FESOM3_MIX_KPP=1
splines FESOM3_SHEAR_SPLINES=1
splines+KPP FESOM3_SHEAR_SPLINES=1 FESOM3_MIX_KPP=1
splines+TKE FESOM3_SHEAR_SPLINES=1 FESOM3_MIX_TKE=1
N2splines FESOM3_N2_SPLINES=1
momadv-vinv FESOM3_MOMADV_OPT=1
momadv-vinv-upw FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=0.7
wsplit $WSPLIT_CFG
momadv-vinv-wsplit FESOM3_MOMADV_OPT=1 $WSPLIT_CFG
GM+wsplit FESOM3_FER_GM=1 $WSPLIT_CFG
bothsplines+GM+Redi+TKE FESOM3_SHEAR_SPLINES=1 FESOM3_N2_SPLINES=1 FESOM3_FER_GM=1 FESOM3_REDI=1 FESOM3_MIX_TKE=1
CFG

# Vector-invariant momentum advection at HIGH rank. The new code adds two nodal scatters
# and two halo exchanges, so np=8 (where nNodL << nod2D) is the configuration that matters
# most -- the same reason the zstar loop above runs it.
echo "=== zstar + momadv-vinv, np=8, $NSTEPS steps (conservation gate, tol=$TOL) ==="
if FESOM3_MOMADV_OPT=1 FESOM3_WHICH_ALE=zstar FESOM3_CONSERVE_TOL="$TOL" \
     mpirun $MPIFLAGS -n 8 "$BIN" < /dev/null 2>&1 | tail -8; then
    :
else
    echo "run_conserve_pi: FAILED (zstar+momadv-vinv np=8)"; fail=1
fi

# upwind blend at np=8: its exchange_elem of omega_e is the one new MR communication.
echo "=== zstar + momadv-vinv-upw, np=8, $NSTEPS steps (conservation gate, tol=$TOL) ==="
if FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=0.7 FESOM3_WHICH_ALE=zstar FESOM3_CONSERVE_TOL="$TOL" \
     mpirun $MPIFLAGS -n 8 "$BIN" < /dev/null 2>&1 | tail -8; then
    :
else
    echo "run_conserve_pi: FAILED (zstar+momadv-vinv-upw np=8)"; fail=1
fi

# w split at np=8: compute_Wvel_split produces w_e/w_i at owned+HALO with no trailing
# exchange (FESOM2's layout) and the NEXT step's momentum TDMA reads them at the halo
# vertices of owned elements, so the rank count where nNodL << nod2D is the one that runs
# that consumer at scale (the halo values themselves are pinned by test_wsplit S5).
echo "=== zstar + wsplit, np=8, $NSTEPS steps (conservation gate, tol=$TOL) ==="
if env $WSPLIT_CFG FESOM3_WHICH_ALE=zstar FESOM3_CONSERVE_TOL="$TOL" \
     mpirun $MPIFLAGS -n 8 "$BIN" < /dev/null 2>&1 | tail -8; then
    :
else
    echo "run_conserve_pi: FAILED (zstar+wsplit np=8)"; fail=1
fi

echo "=== linfs, np=1, $NSTEPS steps (invariants only; drift reported, not gated) ==="
if FESOM3_WHICH_ALE=linfs mpirun $MPIFLAGS -n 1 "$BIN" 2>&1 | tail -7; then
    :
else
    echo "run_conserve_pi: FAILED (linfs invariants np=1)"; fail=1
fi

# SHARED: linfs + sea ice, the shared-element gate (header). Own blocks, not the heredoc loop:
# that loop runs np 1 only. The sync-off case is report-only (no FESOM3_DIVSUM_TOL).
SHARED_ENV="FESOM3_WHICH_ALE=linfs FESOM3_CONSERVE_ICE=1"
while read -r -u 3 label np vars; do
    [ -z "$label" ] && continue
    echo "=== linfs + ice + $label, np=$np, $NSTEPS steps (shared-element gate, divsum tol=$TOL) ==="
    if env $SHARED_ENV $vars FESOM3_DIVSUM_TOL="$TOL" \
         mpirun $MPIFLAGS -n "$np" "$BIN" < /dev/null 2>&1 | tail -8; then
        :
    else
        echo "run_conserve_pi: FAILED (linfs+ice+$label np=$np)"; fail=1
    fi
done 3<<CFG
momadv-vinv-upw1 2 FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=1.0
momadv-vinv-upw1 8 FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=1.0
momadv-scalar 2 FESOM3_MOMADV_OPT=2
CFG
echo "=== linfs + ice + momadv-vinv-upw1, np=2, sync OFF, $NSTEPS steps (report only) ==="
if env $SHARED_ENV FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=1.0 FESOM3_SHARED_SYNC=0 \
     mpirun $MPIFLAGS -n 2 "$BIN" < /dev/null 2>&1 | tail -8; then
    :
else
    echo "run_conserve_pi: FAILED (linfs+ice+momadv-vinv-upw1 np=2 sync off)"; fail=1
fi

# SE: linfs + split-explicit with the SE_CHECK gate (header). Levels measured over 1000 steps on
# pi (Task 1.5: 5.0e-15 m, 5.7e-13 m^2/s, 3.5e-16 m); the tolerances are those levels x 100.
SE_TOL_ETACOMP="${SE_TOL_ETACOMP:-5e-13}"   # m
SE_TOL_TRIM="${SE_TOL_TRIM:-6e-11}"         # m^2/s
SE_TOL_MSL="${SE_TOL_MSL:-4e-14}"           # m
SE_ENV="FESOM3_WHICH_ALE=linfs FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1"
while read -r -u 3 label np vars; do
    [ -z "$label" ] && continue
    echo "=== linfs + SE + $label, np=$np, $NSTEPS steps (SE_CHECK gate) ==="
    rc=0
    out=$(env $SE_ENV $vars mpirun $MPIFLAGS -n "$np" "$BIN" < /dev/null 2>&1) || rc=$?
    { echo "$out" | grep -v '^SE_CHECK ' || true; } | tail -8
    # run maxima of the SE_CHECK tokens (|value|); fails unless every step printed a line
    echo "$out" | awk -v ns="$NSTEPS" -v te="$SE_TOL_ETACOMP" -v tt="$SE_TOL_TRIM" -v tm="$SE_TOL_MSL" '
        /^SE_CHECK / { n++
                       for (i = 3; i <= NF; i++) { split($i, kv, "="); v = kv[2] + 0; if (v < 0) v = -v
                                                   if (v > mx[kv[1]]) mx[kv[1]] = v } }
        END { printf "run_conserve_pi: SE_CHECK max over %d steps: etacomp=%.3e trim=%.3e |msl|=%.3e " \
                     "spreadU=%.3e spreadUm=%.3e maxeta=%.3e maxU=%.3e (tol %s %s %s)\n", n, mx["etacomp"], \
                     mx["trim"], mx["msl"], mx["spreadU"], mx["spreadUm"], mx["maxeta"], mx["maxU"], te, tt, tm
              exit !(n == ns && mx["etacomp"] <= te + 0 && mx["trim"] <= tt + 0 && mx["msl"] <= tm + 0 &&
                     mx["spreadU"] == 0 && mx["spreadUm"] == 0) }' || rc=1
    if [ "$rc" -ne 0 ]; then
        echo "run_conserve_pi: FAILED (linfs+SE+$label np=$np)"; fail=1
    fi
done 3<<CFG
default 1
momadv-vinv-upw1+ice 2 FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=1.0 FESOM3_CONSERVE_ICE=1 FESOM3_DIVSUM_TOL=$TOL
momadv-vinv-upw1+ice 8 FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=1.0 FESOM3_CONSERVE_ICE=1 FESOM3_DIVSUM_TOL=$TOL
momadv-vinv-upw1+ice 1 FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=1.0 FESOM3_CONSERVE_ICE=1
CFG

# SE ZSTAR: zstar + split-explicit with the hard conservation gate and the SE checks (header). Its own
# block (the linfs SE list above gets the Stage 2 np 2/8 lines). Levels: plan Task 3.2; the tolerances
# are those levels x 100.
SE_Z_TOL_ETACOMP="${SE_Z_TOL_ETACOMP:-5e-13}"   # m
SE_Z_TOL_TRIM="${SE_Z_TOL_TRIM:-6e-11}"            # m^2/s
SE_Z_TOL_MSL="${SE_Z_TOL_MSL:-7e-14}"               # m, net of the freshwater (0 here: unforced)
SE_Z_TOL_LAYSUM="${SE_Z_TOL_LAYSUM:-2e-9}"      # m
SE_Z_TOL_W1="${SE_Z_TOL_W1:-4e-16}"                  # m/s
SE_Z_TOL_HLAW="${SE_Z_TOL_HLAW:-8e-10}"            # m
SE_Z_ENV="FESOM3_WHICH_ALE=zstar FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1"
while read -r -u 3 label np vars; do
    [ -z "$label" ] && continue
    echo "=== zstar + SE + $label, np=$np, $NSTEPS steps (conservation gate, tol=$TOL; SE_CHECK gate) ==="
    rc=0
    out=$(env $SE_Z_ENV $vars FESOM3_CONSERVE_TOL="$TOL" mpirun $MPIFLAGS -n "$np" "$BIN" < /dev/null 2>&1) || rc=$?
    { echo "$out" | grep -v '^SE_CHECK' || true; } | tail -9
    # run maxima of the SE_CHECK and SE_CHECK_TAIL tokens (|value|); one line of each per step
    echo "$out" | awk -v ns="$NSTEPS" -v te="$SE_Z_TOL_ETACOMP" -v tt="$SE_Z_TOL_TRIM" -v tm="$SE_Z_TOL_MSL" \
                      -v tl="$SE_Z_TOL_LAYSUM" -v tw="$SE_Z_TOL_W1" -v th="$SE_Z_TOL_HLAW" '
        /^SE_CHECK / || /^SE_CHECK_TAIL / {
                       if ($1 == "SE_CHECK") n++; else nt++
                       for (i = 3; i <= NF; i++) { split($i, kv, "="); v = kv[2] + 0; if (v < 0) v = -v
                                                   if (v > mx[kv[1]]) mx[kv[1]] = v } }
        END { printf "run_conserve_pi: SE_CHECK max over %d steps: etacomp=%.3e trim=%.3e |msl|=%.3e " \
                     "spreadU=%.3e spreadUm=%.3e maxeta=%.3e maxU=%.3e |fw|=%.3e (tol %s %s %s)\n", n, mx["etacomp"], \
                     mx["trim"], mx["msl"], mx["spreadU"], mx["spreadUm"], mx["maxeta"], mx["maxU"], mx["fw"], te, tt, tm
              printf "run_conserve_pi: SE_CHECK_TAIL max over %d steps: laysum=%.3e w1=%.3e hlaw=%.3e (tol %s %s %s)\n", \
                     nt, mx["laysum"], mx["w1"], mx["hlaw"], tl, tw, th
              exit !(n == ns && nt == ns && mx["etacomp"] <= te + 0 && mx["trim"] <= tt + 0 && mx["msl"] <= tm + 0 &&
                     mx["spreadU"] == 0 && mx["spreadUm"] == 0 && mx["laysum"] <= tl + 0 && mx["w1"] <= tw + 0 &&
                     mx["hlaw"] <= th + 0) }' || rc=1
    if [ "$rc" -ne 0 ]; then
        echo "run_conserve_pi: FAILED (zstar+SE+$label np=$np)"; fail=1
    fi
done 3<<CFG
default 1
default 2
default 8
momadv-vinv-upw1+ice 1 FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=1.0 FESOM3_CONSERVE_ICE=1
momadv-vinv-upw1+ice 8 FESOM3_MOMADV_OPT=1 FESOM3_RVO_UPWIND=1.0 FESOM3_CONSERVE_ICE=1
GM+Redi+TKE+wsplit 1 FESOM3_FER_GM=1 FESOM3_REDI=1 FESOM3_MIX_TKE=1 $WSPLIT_CFG
CFG

if [ "$fail" -ne 0 ]; then
    echo "run_conserve_pi: GATE FAILED"
    exit 1
fi
echo "run_conserve_pi: GATE OK"
