# Single precision in FESOM_F — port of the upstream FESOM2 stack (handoff)

**Date:** 2026-10-19. **Branch:** `single-precision` (local, not pushed), worktree
`/work/ab0995/a270088/fesom3_sp`, based on `main` 814e5d0.
**Build:** `./configure.sh --compiler intel --precision sp --build` (→ `build_intel_sp/`). The
double-precision build (`--precision dp`) is bit-identical to `main` (gate 1 below).
**Purpose:** step 1 of FESOM2 SP → FESOM_F → fesom3_jax: cut the bytes the model moves while keeping
in real64 what upstream FESOM2 keeps in real64. The last section is the precision policy the JAX port
has to reproduce.

## 1. How the port is built

- `src/params/mod_precision.F90`: `WP` is `4` in an SP build (unchanged), `MP = max(WP,4)` = 4 (mesh
  geometry is real32, as upstream's mesh is WP), and new: **`WP_full = 8`** (always real64),
  **`MPI_WP_FULL = MPI_DOUBLE_PRECISION`**, **`use_tracer_carry = (WP /= WP_full)`**. Intel flags of the SP
  build: `-r4` instead of `-r8` (upstream does the same), so unsuffixed literals are real32 there.
- Two rules keep the DP build bit-identical:
  1. Everything that is *storage or accumulation* kind only is written as `real(x, WP_full)` /
     `real(kind=WP_full)`. In DP `WP_full == WP == MP`, so every such cast is the identity and the
     compiled arithmetic is the old one.
  2. Every *algorithmic* change (point-slope forcing, FCT flux form, stiffness shadow, tracer carry,
     pi in the PP profile, the SW/GM guards) is behind `#if defined(USE_SINGLE_PRECISION)`. Upstream
     applies three of them in double precision too (#940 point-slope form, #1056 flux form, #1007 pi),
     which changes upstream's DP rounding; FESOM_F keeps them single-precision-only, so its DP build
     stays bit-identical to `main`. The salt anomaly is a runtime option (off: `+ 0.0`, identity).
- Generic interfaces get real64 specifics only in SP (`allreduce_sum/max` in `mod_halo`,
  `write/read_bin_array`, `dump_node`), because in DP a WP and a WP_full specific would be ambiguous.

## 2. Mapping: upstream PR → FESOM_F change

The requested stack (#940, #995, #997, #1007, #1054, #1206, #1207) plus the other merged SP PRs it depends on (`git log --first-parent` of
upstream main: #991 #992 #993 #994 #1026 #1030 #1032 #1056 #986).

| upstream PR | what it does upstream | FESOM_F change (files / routines) |
|---|---|---|
| #940 Support SP | `WP` switch, `-r4`, MPI types, forcing time axis real64 + point-slope interpolation, KPP `epsln` 1e-20, CVMix (TKE) in real64, banner, I/O conversion, output-precision report, salt anomaly | `params/mod_precision.F90` (WP_full); `forcing/mod_forcing_read.F90` (`nc_time`, `rdate`, `delta_t` real64, `forcing_binarysearch_r8`, `forcing_rdate`, point-slope `coef_b + (rdate-time_t0)*coef_a` **SP only**, julday/calyear arithmetic in WP_full); drivers `fesom_lifecycle_native_mr.F90` (+ `fesom_lifecycle_native`, `fesom_forcingdump`, `fesom_forcing_core2`: `rdate` real64); `oce/oce_mixing_kpp.F90` (`epsln` 1e-20 SP); `oce/oce_mixing_tke.F90` (column arithmetic in `TP = WP_full`, WP↔real64 copy-in/out in `calc_cvmix_tke`); banner in the production driver; `io/mod_io_zarr.F90` (real32 writer wrappers), `forcing/mod_forcing_other.F90` and `oce/oce_initial_state.F90` (netCDF axes/fields read as double, narrowed to WP); `mod_halo` already uses `MPI_WP`. Salt anomaly: see #986 |
| #986 salt anomaly (runtime toggle) | store `S - 35` (`use_salt_anomaly`, `S_ref_anomaly`) | `params/mod_param_phys.F90` (`use_salt_anomaly`, `S_ref_anomaly`), env **`FESOM3_SALT_ANOMALY=1`** (production driver only); consumers: `oce_pressure_bv` (`densityJM_components`, `sw_alpha_beta_point`, s<0 screen), `mod_ice_dyn` (ocean→ice SSS), `mod_ice_oce_coupling` (`rsss`, SSS restoring), `oce_mixing_kpp` (Bo), `oce_ale_tracer` (`bc_surface` dilution term, clip bounds, KPP non-local `rsss`), driver (IC conversion after `do_ic3d`, restart detection/conversion, `sss`/`salt` output offset, step-diag Smax). Off: `S_ref = 0`, `x + 0.0` is the identity |
| #991 FCT generic min/max | `dmax1/dmin1` → `max/min` | nothing to do (FESOM_F already generic) |
| #992 WP from iso_fortran_env, WP_full | kind definitions | `mod_precision.F90` (`WP_full`, `MPI_WP_FULL`) |
| #993 global integrals in WP_full | `integrate_nod` real64 | `ice/mod_ice_oce_coupling.F90 integrate_nod_2D` (WP_full accumulator, WP_full reduction, rounded once); `mesh/mod_mesh_areas.F90` (`ocean_area` sums) |
| #994 SSH CG in WP_full | CG scalars + sparse-row sums real64 | `oce/oce_ssh_solve.F90 ssh_solve_cg` (`s_old, s_aux, sprod, al, be, rtol` WP_full; A·x rows summed in WP_full, rounded once into the WP vector) |
| #995 auto SP preconditioner variant | `precond_variant` = 1 in SP | **not applicable**: FESOM_F uses a diagonal (Jacobi) preconditioner, symmetric by construction (no MITgcm variant switch) |
| #997 stiffness in real64 | real64 shadow of `ssh_stiff%values` | `types/mod_mesh.F90` (`values_full`, SP only); `oce/oce_ssh_rhs.F90 update_stiff_mat_ale/add_edge_stiffness` (seed on first update, accumulate `real(fy*factor,8)`, `values = real(values_full,MP)` once per step). Only zstar calls it; production (linfs) does not |
| #1007 pi at machine precision | one `pi` | **SP only**: `oce_ale_mixing_pp.F90` `PI_QIANG` (the `3.1415926` literal is one real32 ulp off pi; DP keeps the v2.7.3 literal). `mod_constants` pi / ice-thermo PI already round to the same real32 as exact pi |
| #1026 / #1030 MPI datatype of WP_full reductions | `MPI_WP_FULL` | `infra/mod_halo.F90` (`allreduce_sum_f0/f1`, `allreduce_max_f0`, SP only) |
| #1032 output means in real64 | accumulate means in real64 for every stream | `io/mod_io_means.F90` (`acc_r8`: every stream in SP; DP unchanged, i.e. float32 streams still accumulate in float32 as v2.7.3) |
| #1054 / #1056 FCT low-order tendency in flux form | `fct_LO_tend` instead of `LO*hnode_new - ttf*hnode`; w-split implicit part adds `tr*hnode_new` | **SP only**: `oce/oce_adv_tra_driver.F90 do_oce_adv_tra`, `oce/oce_adv_tra_flux.F90 oce_tra_adv_flux2dtracer (lo_tend)`, `oce/oce_adv_tra_ver.F90 adv_tra_vert_impl (tend)` — production runs w-split, so the #1056 form is the relevant one |
| #1206 SP conservation (real64 accumulators + tracer carry) | `del_ttf*`, `fct_LO_tend` WP_full, `flux*dt/areasvol` in WP_full, column scratch WP_full, carry | `types/mod_tracer.F90` (`del_ttf`, `del_ttf_advhoriz/advvert`, `fct_LO_tend` WP_full; `carry(nl-1,nod,ntr)` WP); `oce_adv_tra_flux.F90` (WP_full products); `oce_adv_tra_ver.F90 adv_tra_vert_impl` and `oce_ale_tracer.F90 diff_ver_part_impl_ale` (column scratch WP_full); carry at both state writes (`diff_tracers_ale` ALE reconstruction, `diff_ver_part_impl_ale` update; SP only, allocated on first use); `fesom_conserve` budgets in WP_full incl. carry; binary dump through the WP format |
| #1207 layer thickness in real64 | `hnode`, `hnode_new` WP_full, `exchange_nod_h` | `types/mod_mesh.F90` (`hnode, hnode_new` WP_full); `infra/mod_halo.F90 exchange_nod_h` (hi/lo split, SP); `oce/oce_ale.F90 vert_vel_ale`; `io/mod_io_restart.F90` (thickness staged through WP on write, widened on read, `exchange_nod_h` on the halo); `mod_dump` / `mod_binary_arrays` WP_full specifics |
| #1029 coupled SP | OIFS/OASIS/XIOS | not ported (no coupling in FESOM_F) |

FESOM_F-only additions (no upstream counterpart, needed to run SP at all):
- `oce/oce_shortwave_pene.F90`: `exp(z/sc)` in WP_full (SP only). In real32 `exp(-88)` is subnormal
  above the `aux < 1e-5` cut-off and Intel's `expf` traps (`-fpe0`, FTZ on): pi crashed at step 1.
- `oce/oce_fer_gm.F90`: the Redi template `K_hor**(...)` (FESOM2's literal exponentiation) in WP_full,
  clamped to `huge(WP)` (SP only): real32 overflow trap on pi at step 1. Only the Redi-without-GM path
  reads it.
- step diagnostic `divsum` sums in WP_full; `fesom_conserve` helem tolerance scaled with `epsilon(WP)`.

## 3. What stays real64 (WP_full) in the SP build

| quantity | where | why |
|---|---|---|
| forcing time axis `nc_time`, model date `rdate`, bracket length `delta_t`, bracket start `time_t0`, Julian-day arithmetic | `mod_forcing_read` | a Julian day ~2.44e6 has a real32 ulp of 6 h > the 3-h JRA records |
| forcing slope `coef_a` is *formed* in real64 (`(data2-data1)/delta_t`) then stored WP; elapsed time `rdate - time_t0` formed in real64, narrowed once | `forcing_getcoeffld`, `forcing_timeinterp` | point-slope form keeps the 2.4e6 day out of WP |
| SSH CG scalars `s_old, s_aux, sprod(2), al, be, rtol` and the row sums of A·x, A·p | `oce_ssh_solve` | they steer the iteration |
| global integrals: `integrate_nod_2D` (freshwater / virtual-salt / relaxation balancing), `ocean_area` | `mod_ice_oce_coupling`, `mod_mesh_areas` | sum over all nodes used to balance fluxes to zero |
| SSH stiffness accumulator `ssh_stiff%values_full` (zstar) | `oce_ssh_rhs` | per-step increments ~1e-7 of the entry |
| TKE column solve (mixing length, diffusivities, forcing, tridiagonal) | `integrate_tke`, `solve_tridiag` | upstream runs CVMix in cvmix_r8 |
| tracer tendencies `del_ttf`, `del_ttf_advhoriz`, `del_ttf_advvert`, `fct_LO_tend`, and the `flux*dt/areasvol` products added into them | `mod_tracer`, `oce_adv_tra_*`, `oce_ale_tracer` | edge/interface fluxes must telescope |
| column scratch of the implicit vertical diffusion and of the w-split implicit advection (`a,b,c,tr,cp,tp,zbar_n,Z_n,m,zinv,dz,v_adv,...`) | `diff_ver_part_impl_ale`, `adv_tra_vert_impl` | TDMA for the increment of a real32 state |
| ALE reconstruction content `T*hnode + del_ttf + carry` | `diff_tracers_ale` | rounded once to WP; residual → carry |
| layer thickness `hnode`, `hnode_new` | `mod_mesh` | volume budget |
| conservation budgets (`fesom_conserve`), step-diag divsum | drivers | diagnostics |
| exp of the SW attenuation; GM Redi template | `oce_shortwave_pene`, `oce_fer_gm` | real32 range (FESOM_F-only) |
| on disk: checkpoints `<f8` (WP values widened); output `<f8` streams; output means accumulated in real64 | `mod_io_*` | double on disk, cast at the boundary |
| MPI: every reduction of a WP_full buffer uses `MPI_DOUBLE_PRECISION`; halos stay `MPI_WP`; `hnode_new` halo = hi + lo through the WP exchange (exact to 2^-48) | `mod_halo` | |

SP-only state in WP: the tracer **carry** `tracers%work%carry` (nl-1, nod, ntracer), not in the checkpoint
(a restarted SP run starts with zero carry, as upstream). *Since branch `sp-restart-exact` (2026-10-28) it
is: `carry_t`, `carry_s` (IMPLEMENTATION.md §10, LESSONS L79).*

Everything else is WP = real32: prognostic T, S, u, v, η, d_eta, hbar, w, all ice fields and the EVP
stresses, TKE state and Av/Kv, mesh geometry, the stiffness matrix values the CG reads, the CG vectors,
EOS/pressure/PGF, GM/Redi, KPP/PP, bulk formulas, forcing field values.

## 4. Gates

All jobs on Levante `compute`, account ab0995, Intel ifort 2021.5, OpenMPI 4.1.2, `env.sh intel`
(KNEM workaround). Run scripts: `/work/ab0995/a270088/fesom3_sp_runs/jobs/` (`runpair.sbatch` wraps
`~/fesom3_jax/scripts/fesom_prod_run.sh` = Dmitry's production physics, JRA55 1958, linfs, TKE, GM+Redi,
w-split, EVP, SW penetration; `ctest.sbatch`, `conserve.sbatch`, `restart_pair.sbatch`). Run output under
`/work/ab0995/a270088/fesom3_sp_runs/`. CORE2 production dt = 1800 s (48 steps/day, the driver default).

### Gate 1 — DP unchanged (bit-identical to main 814e5d0)

Reference: `/work/ab0995/a270088/fesom3_jax/fesom_f_main/build_intel_dp` (main), job 28256736 (pi) and
28256737 (CORE2). Comparison = `diff -rq` of the final checkpoint folders (356 files each: every
prognostic field of ocean + ice + EVP stresses + `checkpoint.json`).

| check | commit | job | result |
|---|---|---|---|
| pi, np 2, 144 steps (3 days), production physics | 4a7e181-pre (forcing/CG/integrals) | 28256736 | identical |
| CORE2 dist_512, 72 steps | same | 28256737 | identical |
| pi np 2 / CORE2 dist_512 | full port before 4a7e181 | 28256921 / 28256922 | identical |
| pi np 2 / CORE2 dist_512 | 2fbcc8f | 28257093 / 28257094 | identical |
| ctest DP | 2fbcc8f | 28257095 | 167/167 pass (also 28256929 on 4a7e181) |
| `tools/run_conserve_pi.sh` DP | 2fbcc8f | 28257096 | GATE OK |
| pi / CORE2 / ctest DP with the salt-anomaly code (toggle off) | 507eed0 | 28257163 / 28257164 / 28257165 | identical / identical / 167/167 (§4c) |

### Gate 2 — SP builds and runs

- **Build:** `build_intel_sp` compiles: library, all 33 drivers, all test programs.
- **pi, np 2, production physics:** 240 steps = 5 days stable (job 28257112; max|η| 2.10 m, max|u| 0.74
  m/s, Tmax 29.7, Smax 37.36 at step 240). A `-check all` debug build ran 72 steps clean (job 28256952).
- **SP restart:** straight 144 steps vs 72 + restart + 72 on pi (job 28257061): the resumed run reads
  the checkpoint and continues; the end states agree to ~1e-5 relative (max|u| 0.65117 vs 0.65116,
  m_ice 2.1564 vs 2.1565). Not bitwise, by design as upstream (carry and stiffness shadow are not in
  the checkpoint).
- **CORE2 dist_512, one model month (1488 steps = January 1958), production physics** (job 28257111 SP,
  28257097 DP, commit 2fbcc8f): stable, all fields finite.

  | per step (ms, mean over ranks) | DP | SP | SP/DP |
  |---|---|---|---|
  | LOOP TOTAL | 43.22 | 35.69 | **0.826** |
  | ocean step | 27.17 | 20.28 | 0.747 |
  | — mix, pressure, EOS | 3.16 | 2.32 | 0.73 |
  | — dynamics u,v,w | 4.86 | 3.35 | 0.69 |
  | — SSH solve (CG) | 6.77 | 6.40 | 0.95 |
  | — GM / Redi | 2.38 | 1.57 | 0.66 |
  | — tracers | 9.01 | 6.02 | 0.67 |
  | sea ice | 8.55 | 8.10 | 0.95 |
  | forcing (JRA55 read + interp) | 6.35 | 6.52 | 1.03 |
  | job wall time (incl. init) | 92 s | 79 s | 0.86 |

  The ocean kernels gain 25–35 %; the CG at 512 ranks is latency-bound (allreduces), the forcing is
  I/O-bound (netCDF-4 inflate) and does not change, and the EVP ice gains only 5 %.
- **Conservation** (`tools/run_conserve_pi.sh`, unforced pi, 20 steps, all 25 cases; job 28257113 SP at
  TOL=1e-6, 28257096 DP at the standard 1e-12): SP **GATE OK at 1e-6**. |relative drift| over 20 steps,
  range over the 20 zstar cases (np 1/2/8, Redi, KPP, TKE, GM, splines, vinv, w-split):

  | | heat | salt | volume |
  |---|---|---|---|
  | DP | 2.1e-16 … 1.7e-14 | 3.5e-16 … 2.4e-14 | 1.3e-15 … 4.7e-14 |
  | SP | 3.5e-14 … 4.2e-12 | 1.5e-14 … 8.3e-13 | 6.8e-14 … 4.9e-13 |

  linfs (not conserving by design, reported only): SP heat −2.17e-4 / salt −1.08e-6, DP −2.17e-4 /
  −1.09e-6. The production driver itself prints no budget.
- **ctest SP:** see §4b.

### Gate 3 — SP vs DP after one month (CORE2, Jan 1958, production physics)

`/work/ab0995/a270088/fesom3_sp_runs/tools/compare_ckpt.py` on the day-31 checkpoints (areas from the
mesh diagnostic `fesom.mesh.diag.zarr` `nod_area` surface level):

| quantity | DP | SP | SP−DP |
|---|---|---|---|
| area-mean SST [°C] | 18.264175 | 18.264163 | −1.16e-05 |
| area-mean SSS [psu] | 34.6609183 | 34.6609179 | −4.0e-07 |
| sea-ice area NH [10^6 km²] | 15.30882 | 15.30890 | +8.7e-05 |
| sea-ice area SH [10^6 km²] | 9.56067 | 9.56078 | +1.14e-04 |
| sea-ice volume NH [10^3 km³] | 19.57073 | 19.57083 | +1.0e-04 |
| sea-ice volume SH [10^3 km³] | 19.07452 | 19.07466 | +1.4e-04 |
| SSH rms (area-weighted, mean removed) [m] | 0.6443576 | 0.6443560 | −1.5e-06 |

Pointwise (ocean nodes): SST max|Δ| 0.64 °C, area-rms 2.5e-3 °C; SSS max 0.18 psu, rms 1.3e-3;
SSH max 2.7e-3 m, rms 1.1e-4 m; a_ice max 0.067, rms 4.7e-4. (A 46.5-day pair, jobs 28257031/28257033,
gave the same picture: SST mean +4e-6, SSS +3e-6, ice area ±4e-4, SSH rms +5e-6.) No FP64 noise
envelope was run, so these numbers carry no verdict.

### 4a. One model year (CORE2 dist_512, 1958, production physics; jobs 28257317 DP, 28257318 SP)

Both runs complete the year (17520 steps) with finite fields. Loop time per step DP 41.88 ms, SP 35.05
ms (**0.837**; ocean step 26.48 → 20.12 ms, SSH solve 6.52 → 6.39, ice 8.12 → 7.81, forcing 6.14 → 6.32);
job wall time 768 s vs 678 s (incl. setup). Loop-only throughput: DP 118 SYPD, SP 141 SYPD on 4 nodes.

State on 1959-01-01 00:00 (`compare_ckpt.py`):

| quantity | DP | SP | SP−DP |
|---|---|---|---|
| area-mean SST [°C] | 18.051411 | 18.051366 | −4.5e-05 |
| area-mean SSS [psu] | 34.5921783 | 34.5922062 | +2.8e-05 |
| sea-ice area NH / SH [10^6 km²] | 14.40767 / 7.35548 | 14.40766 / 7.36117 | −1.2e-05 / +5.7e-03 |
| sea-ice volume NH / SH [10^3 km³] | 23.92589 / 8.97554 | 23.92577 / 8.99177 | −1.2e-04 / +1.6e-02 |
| SSH rms [m] | 0.6230853 | 0.6231148 | +2.9e-05 |
| global-mean SSH [m] | 7.7e-17 | −9.1e-10 | |

Pointwise: SST max|Δ| 0.83 °C, area-rms 0.014 °C; SSS max 0.30 psu, rms 7.8e-3; SSH max 0.014 m, rms
7.5e-4 m; a_ice max 0.31, rms 1.2e-3. The SP global-mean SSH is −9e-10 m after one month and after one
year, so the real32 closure of the linfs divergence (open problem 1) does not accumulate into a mean
sea-level drift on this time scale (the freshwater balancing integral is WP_full).

### 4b. ctest in SP

Job 28257224 (`ctest -j 32 --timeout 300` in `build_intel_sp`, commit 507eed0): **80 of 167 pass,
87 fail**; all failures are DP-only by construction, none is a defect of the SP model:

- **Round-off assertions at double-precision tolerance (1e-12 … 1e-14 relative)** in the unit tests:
  `test_mesh`, `test_geom_mixed_*` (G1 node-area sum = element-area sum, 1.7e-6 in SP),
  `test_coupling_mixed_*`, `test_ice_mass_*`, `test_ice_dyn_mixed_*` (strain-rate adjoint),
  `test_pgf_mixed_*` (sigma_xy of a linear field, 1e-5 in SP), `test_vinv_*`, `test_updown_mixed_*`
  (exact gradients/MUSCL of linear fields), `test_vert_spline`, `test_vector_rotate`, `test_wsplit_*`
  (`|Cu*(1-f) - Cu_max| <= 1e-14*Cu`), `test_ivertvisc_*`, `test_wimpl_tra_np1` (constancy/conservation
  to round-off), `test_ssh_stiff_*` (row sums, symmetry), `test_shared_elem_*` (divergence-sum ratio
  < 1e-12; 6e-8 … 3.8e-9 in SP).
- **`fesom_conserve_*` ctest cases at TOL = 1e-12** (21 cases, plus `linfs_vinv_shsync` divsum
  1e-12): the same physics passes `tools/run_conserve_pi.sh` at TOL = 1e-6 in SP (§4, job 28257113).
- **Tests that pass `MPI_DOUBLE_PRECISION` on WP buffers** (`test_ivertvisc`, `test_geom_mixed`,
  `test_ssh_stiff`, `test_params`, `test_wimpl_tra`, `test_wsplit`, `test_vinv`): at np > 1 in SP they
  reduce garbage (`test_wimpl_tra_np2`: floating invalid inside `MPI_Allreduce`) or hang
  (`test_geom_mixed_pi_w_np2`, `test_ssh_stiff_mixed_np2`: 300 s timeout).
- **`meshdiag_mixed`, `meshdiag_pi_w`**: the M5 face-orientation check of `tools/check_meshdiag.py` on
  real32 mesh coordinates (DP tolerance).

What passes in SP includes the halo/partition/decomp/IO tests, the Zarr/restart smoke and round-trip
tests, the clock test, the meshes' read tests and `test_bottom`. Making the unit tests precision-aware
(tolerances in units of `epsilon(1.0_WP)`, `MPI_WP` in the test reductions) is open item 7.

### 4c. Salt anomaly runs (FESOM3_SALT_ANOMALY=1, commit 507eed0)

- DP with the anomaly code and the toggle off: pi and CORE2 checkpoints identical to main (jobs
  28257163, 28257164), ctest DP 167/167 (28257165).
- SP pi, np 2, 5 days with the anomaly: stable (job 28257166; IC converted, step-diag Smax absolute
  37.355).
- CORE2 dist_512, one month, SP and DP with the anomaly (jobs 28257167, 28257168), compared on day 31:

  | pair (B − A) | SST mean | SSS mean | ice area NH / SH [10^6 km²] | SSH rms [m] | SST area-rms | SSS area-rms |
  |---|---|---|---|---|---|---|
  | SP_off − DP_off (precision, §4 Gate 3) | −1.16e-05 | −4.0e-07 | +8.7e-05 / +1.14e-04 | −1.5e-06 | 2.5e-03 | 1.3e-03 |
  | SP_on − DP_on (precision, anomaly mode) | +2.35e-06 | −5.6e-06 | +1.09e-04 / +1.05e-04 | −6.7e-07 | 2.0e-03 | 1.2e-03 |
  | DP_on − DP_off (method, DP) | −1.28e-05 | −1.7e-06 | −7.3e-05 / −1.7e-05 | −4.0e-07 | 1.7e-03 | 1.1e-03 |

  After one month the three pairs are all at the same level: the pointwise differences are the
  divergence of nearby trajectories, so a month cannot separate precision error from method bias.
  Upstream's evidence for the anomaly is a multi-year salt drift (Suvarchal's 60-year CORE2 pairs).

## 5. Open problems

1. **linfs SSH divergence closes only to real32.** `ssh_rhs`/`ssh_rhs_old` (and the hbar divergence)
   accumulate edge fluxes in WP; the step-diagnostic `divsum` (|Σ ssh_rhs_old| / Σ|ssh_rhs_old|) is
   ~1e-7 in SP against ~1e-17 in DP. Upstream FESOM2 SP has the same accumulation (not part of its
   stack). A WP_full accumulator for `ssh_rhs` would be a FESOM_F extension (DP-neutral); not done.
2. **zstar volume closure** in SP is 1e-13 … 5e-13 per 20 steps on pi (upstream reports 3.5e-14 per day
   for its SP pi case). The thickness is real64 now, but its per-step increment
   `(zbar_3d_n(nz)-zbar_3d_n(nz+1))*dd` is formed from real32 `zbar_3d_n`/`dd`, as upstream. Not
   investigated further.
3. **Output of `<f8` streams in SP** passes the mean through the WP-typed redistribution
   (`decomp_redistribute`) before the Zarr writer widens it, so an 8-byte SP mean carries real32
   rounding (the accumulation itself is real64, #1032). Upstream writes the real64 mean. Needs a
   real64 path in `mod_io_decomp`.
4. **Restart in SP** holds `hnode` rounded to WP (as upstream) and drops the carry and the stiffness
   shadow (as upstream); a resumed SP run is close to, not identical with, a straight run.
   *Resolved 2026-10-28 (branch `sp-restart-exact`): real64 paths in `mod_io_decomp`/`mod_io_zarr`,
   `values_full` and the carry in the checkpoint; the SP restart gates are `max|Δ| = 0` (LESSONS L79).
   The real64 decomposition path is also what item 3 needs; the output writer does not use it yet.*
5. **#995 has no counterpart** (Jacobi preconditioner). CG iteration counts in SP were not measured; the
   SSH solve time is 0.95 of DP at dist_512.
6. **Salt anomaly** is wired into the production driver only (`FESOM3_SALT_ANOMALY`); other drivers keep
   `S_ref = 0`. No guard refuses unsupported combinations (upstream refuses cavities, icebergs, transient
   tracers, DVD/cmor diagnostics — none exist in FESOM_F; the KPP non-local `rsss` is corrected
   instead of refused).
7. **ctest in SP:** the unit tests carry DP tolerances (see §4b); an SP tolerance set or an SP-specific
   subset was not written.
8. **GNU SP build** (`--compiler gnu --precision sp`, probe build `build_gnu_sp`): the library and all
   33 drivers compile with gfortran; one test program does not (`test/test_coupling_mixed.F90:193`,
   `allocate(tracers%data(2), tracers%data(2)%values(...))` in one statement, a line unchanged from
   `main` and unrelated to precision). No GNU SP run was made.
9. **CG iteration counts** in SP vs DP were not measured (the production driver does not report them).

## 6. For the JAX port (fesom3_jax): precision policy to reproduce

The bitwise contract of fesom3_jax is against FESOM_F DP; an SP mode should reproduce **this** policy
(arrays float32 unless listed), so that FESOM_F-SP becomes its oracle.

**float64 (WP_full) storage:**
- `hnode`, `hnode_new` (layer thickness, nl-1 × nod). Their halo exchange in SP is the hi/lo split
  (`hi = f32(x)`, `lo = f32(x - hi)`, rebuild `x = f64(hi) + f64(lo)` on halo nodes) — on a single GPU
  without halos, simply keep float64.
- tracer tendency accumulators `del_ttf`, `del_ttf_advhoriz`, `del_ttf_advvert`, `fct_LO_tend`.
- forcing time axis and the model date (`nc_time`, `rdate`, `time_t0`, `delta_t`).
- SSH stiffness shadow `values_full` (zstar only).
- conservation/diagnostic integrals.

**float32 storage with float64 arithmetic inside (round once at the end):**
- CG: dot products (`r·z`, `r·r`, `p·Ap`, `b·b`) and the scalars α, β, rtol in float64; each row of A·x
  / A·p summed in float64 and rounded to float32; `x += α p`, `r -= α Ap`, `p = z + β p` computed with
  the float64 scalar and rounded to float32. Global reductions in float64.
- forcing: slope `coef_a = f32((d2 - d1)/delta_t_f64)`, `coef_b = d1`; per step
  `dt_el = f32(rdate_f64 - t0_f64)`, `atm = coef_b + dt_el*coef_a` (all float32).
- `integrate_nod_2D`: float64 accumulator of float32 products, float64 reduction, one rounding.
- TKE column solve entirely in float64 from widened float32 inputs (N², shear², dz, hnode, tke_old,
  surface stress), outputs `tke`, `Av`, `Kv` rounded to float32. Keep the unsuffixed literals (6.6,
  0.5, 3./2.) as float32 constants promoted to float64, as upstream compiles them.
- tracer advection: every `flux*dt/areasvol` added into the float64 tendencies (horizontal edge scatter
  and the vertical difference `(f64(flux_v(k)) - flux_v(k+1))*dt/areasvol`); FCT LO: `fct_LO_tend`
  (float64) = edge scatter + vertical difference, `fct_LO = f32((ttf*hnode + fct_LO_tend)/hnode_new)`;
  the w-split implicit step adds `tr*hnode_new` to `fct_LO_tend`; `del_ttf_advvert += fct_LO_tend`
  replaces `- ttf*hnode + LO*hnode_new`.
- implicit vertical diffusion and implicit w-split advection: the tridiagonal coefficients, rhs,
  Thomas sweep in float64.
- **tracer carry** (float32 array per tracer, zero at start): at the ALE reconstruction
  `acc = T*hnode + del_ttf + carry` (f64), `T' = f32(acc/hnode_new)`,
  `carry = f32(acc - T'*hnode_new)`; at the implicit-diffusion update `acc = T + dT` (f64),
  `T' = f32(acc)`, `carry += f32((acc - T')*hnode_new)`. The carry counts as content in any budget.
- shortwave penetration `exp(z/sc)` in float64.

**float32 everything else**, including all prognostic state (T, S, u, v, η, d_eta, hbar, w, ice
a/m/snow, EVP σ, TKE), geometry, EOS/pressure, momentum, GM/Redi, mixing coefficients, CG vectors and
matrix values.

**SP-only constants:** KPP `epsln = 1e-20` (1e-40 is subnormal in float32 and flushed); PP background
profile uses machine pi.

**Options:** salt anomaly (`S_ref = 35`, every absolute-S consumer adds `S_ref`; the list in §2/#986).
**Not in the policy (open):** linfs `ssh_rhs` accumulation stays float32 (open problem 1).

**Notes for the JAX implementation:** keep the reductions' summation *order* questions separate from
precision: FESOM_F-SP is deterministic per rank count, so a JAX SP oracle comparison will be roundoff-
level, not bitwise, unless the summation orders already matched in DP (they do in fesom3_jax's CPU gate).
XLA must not fuse float64 accumulations into float32 (watch `jnp` dtype promotion: a float32 array times
a Python float stays float32 — use explicit `.astype(jnp.float64)` at every WP_full point).

## 7. Rebased on split-explicit (fb92d87)

**Date:** 2026-10-20. **Branch:** `single-precision-se` (local, not pushed), worktree
`/work/ab0995/a270088/fesom3_sp_se`: the seven commits of `single-precision` replayed on
`origin/split-explicit` fb92d87 (PR #3), plus two commits for split-explicit (SE) in SP. The DP
reference is a build of fb92d87 in the detached worktree `/work/ab0995/a270088/fesom3_se_refbuild`.
Run output: `/work/ab0995/a270088/fesom3_sp_runs/se_rebase/`.

**Conflicts (two, both resolved by keeping both sides):**
- `src/drivers/fesom_lifecycle_native_mr.F90`: SE's `dstat(10)` = max CFL_z in the step diagnostic
  next to SP's `dsum` in WP_full (`real(ssh_rhs_old, WP_full)`, reduced as `MPI_WP_FULL`).
- `src/infra/mod_halo.F90`: SE's `allreduce_max_r1` (vector MAX for SE_CHECK) next to SP's
  `allreduce_max_f0`, `allreduce_sum_f0/f1` (SP only) and `exchange_nod_h`.

**What SE needed for SP.** Nothing to compile: library, all 33 drivers and all test programs
(including `test_ssh_se`) build in SP unchanged. The SE arithmetic stays in WP, i.e. real32 in SP:
the elevation and transport rings, the AB3/AM4 substeps, the mean transport, the trim, the depth
law and the CFL-guard power iteration (the JAX SP port keeps SE in float32 too). Two diagnostic
commits, DP-neutral under rule 1 (identity casts, unchanged operation order):
- `0aa8737` SE_CHECK integrals in WP_full: the mean-sea-level area sums of `se_check_values`,
  `dyn%se_msl0`, `dyn%se_fwmsl`, and the `hnode` layer sum of `se_check_tail_values` (a real32 sum
  of a 4000 m column rounds at 2.4e-4 m). The SP model state is unchanged (pi checkpoint
  identical before/after).
- `ff64992` sea-ice mass budgets of `fesom_conserve` in WP_full. The SP port had moved the tracer
  content and volume to WP_full but not the three ice masses; their real32 sum over the nodes rounds
  at ~1e-6 relative by itself, so the zstar + SE + sea-ice case that the SE branch adds to
  `tools/run_conserve_pi.sh` failed at TOL = 1e-6 (ice drift −1.5e-6; the same case with the CG
  solver gives −2.2e-6, job 28268546/28268547). With the fix the largest SP ice drift is 3.5e-8.

**Gates** (`compute`, Intel, `env.sh intel`; job scripts as in §4; checkpoint comparison = `diff -rq`
of the final checkpoint folders, 354 files with SE off, 464 with SE on):

| check | code | job | result |
|---|---|---|---|
| SP build (library, drivers, tests) | a59c961, ff64992 | interactive allocation | compiles, no SP changes needed |
| DP pi np 2, 144 steps, SE off: fb92d87 vs rebased | a59c961 | 28268373 | identical |
| DP pi np 2, 144 steps, `FESOM3_SSH_SE=1` | a59c961 | 28268374 | identical |
| DP CORE2 dist_512, 72 steps, SE off | a59c961 | 28268375 | identical |
| DP CORE2 dist_512, 72 steps, SE on | a59c961 | 28268376 | identical |
| ctest DP, fb92d87 / rebased | a59c961 | 28268377 / 28268378 | 176/176 / 176/176 |
| `run_conserve_pi.sh` DP, fb92d87 | fb92d87 | 28268379 | GATE OK |
| the same four checkpoint pairs at HEAD | ff64992 | 28268577 / 28268578 / 28268579 / 28268580 | identical / identical / identical / identical |
| ctest DP at HEAD | ff64992 | 28268581 | 176/176 |
| `run_conserve_pi.sh` DP at HEAD | ff64992 | 28268582 | GATE OK, log byte-identical to fb92d87's |
| SP pi np 2, 240 steps, SE on (+ STEP_DIAG, SE_CHECK) | a59c961; 0aa8737 code | 28268353; 28268438 | stable, finite; the two checkpoints identical |
| SP / DP CORE2 dist_512, 1488 steps, SE on | 0aa8737 code (= HEAD for this driver) | 28268439 / 28268441 | stable, all fields finite |
| `run_conserve_pi.sh` SP, TOL = 1e-6, SE tolerances report-only | before ff64992 / ff64992 | 28268442 / 28268583 | 1 failure (ice budget, above) / GATE OK |
| ctest SP | ff64992 | 28268584 | 84 of 176 pass (SP branch: 80 of 167), see below |

**SP with SE, pi** (np 2, 5 days, job 28268438): step 240 max|η| 3.5166 m (DP 3.5163; SP with the CG
solver gives 2.10 m, §4, so the higher maximum comes with the method, not the precision), max|u| 0.777 m/s (DP 0.766),
divsum 8.4e-9 (DP 4.3e-17; SP with CG 1.1e-7). CFL guard identical to DP (ω_max 5.9801e-3 1/s,
M_min 8 for M = 50). SE_CHECK run maxima SP / DP: etacomp 2.9e-6 / 4.9e-15 m, trim 2.4e-4 / 4.5e-13
m²/s (max|U| 915 m²/s), |msl| 5.0e-8 / 3.5e-16 m, copy spreads 0 in both.

**SP with SE, CORE2 one month** (dist_512, Jan 1958, production physics + `FESOM3_SSH_SE=1`, M = 50,
CFL guard M_min = 40 in both precisions). Per step (ms, mean over ranks), next to the CG pair of §4
(jobs 28257097 / 28257111, commit 2fbcc8f):

| | DP SE | SP SE | SP/DP | DP CG | SP CG |
|---|---|---|---|---|---|
| LOOP TOTAL | 40.18 | 32.23 | **0.802** | 43.22 | 35.69 |
| ocean step | 24.07 | 16.86 | 0.70 | 27.17 | 20.28 |
| — SSH part (dynamics ssh) | 4.55 | 3.62 | 0.80 | 7.76 | 7.02 |
| — of which SE barotropic substeps / CG solve | 3.79 | 3.07 | 0.81 | 6.77 | 6.40 |
| — mix, pressure, EOS | 3.19 | 2.31 | 0.73 | 3.16 | 2.32 |
| — dynamics u,v,w | 4.89 | 3.35 | 0.69 | 4.86 | 3.35 |
| — GM / Redi | 2.41 | 1.57 | 0.65 | 2.38 | 1.57 |
| — tracers | 9.04 | 6.01 | 0.67 | 9.01 | 6.02 |
| sea ice | 8.62 | 8.06 | 0.93 | 8.55 | 8.10 |
| forcing | 6.32 | 6.52 | 1.03 | 6.35 | 6.52 |
| job wall time | 87 s | 78 s | 0.90 | 92 s | 79 s |

At 512 ranks the SE substeps (per substep one 2-D node halo exchange and, with the default BT
viscosity, one element exchange; no reductions) cost 0.56 of the CG
solve in DP and 0.48 in SP; unlike the latency-bound CG, they gain ~20 % from SP.

State on day 31 (`compare_ckpt.py`), SP − DP with SE next to SP − DP with CG (§4 Gate 3):

| quantity | DP SE | SP SE − DP SE | SP CG − DP CG |
|---|---|---|---|
| area-mean SST [°C] | 18.264090 | −3.2e-06 | −1.16e-05 |
| area-mean SSS [psu] | 34.6611805 | −5.5e-06 | −4.0e-07 |
| sea-ice area NH / SH [10^6 km²] | 15.30033 / 9.54767 | +1.5e-05 / +1.7e-04 | +8.7e-05 / +1.14e-04 |
| sea-ice volume NH / SH [10^3 km³] | 19.55625 / 19.03312 | +1.4e-04 / +3.2e-04 | +1.0e-04 / +1.4e-04 |
| SSH rms [m] | 0.6444626 | −3.6e-06 | −1.5e-06 |
| global-mean SSH [m] | 8.6e-17 | SP: +3.8e-09 | SP: −9.8e-10 |
| pointwise area-rms SST / SSS / SSH / a_ice | | 2.5e-3 / 1.3e-3 / 1.3e-4 m / 4.6e-4 | 2.5e-3 / 1.3e-3 / 1.05e-4 m / 4.7e-4 |

The precision differences with SE are at the level of those with CG. For scale, the method
difference DP SE − DP CG after the same month (against the §4 DP run, job 28257097) is 5 to 55 times
larger (SST rms 1.2e-2, SSS 6.6e-3, SSH 7.2e-3 m, a_ice 3.4e-3; ice area NH −8.5e-3).

**Conservation in SP with SE** (`run_conserve_pi.sh`, 20 steps, |relative drift|, job 28268583):

| | heat | salt | volume |
|---|---|---|---|
| zstar, CG (21 cases) | 3.5e-14 … 4.2e-12 | 1.5e-14 … 8.3e-13 | 6.8e-14 … 4.9e-13 |
| zstar, SE (6 cases, np 1/2/8) | 1.7e-12 … 1.5e-11 | 6.6e-13 … 2.1e-12 | 8.9e-14 … 2.2e-12 |

SE_CHECK run maxima in SP (DP in brackets): etacomp 0.9–1.0e-6 m (2e-15), trim 6–9e-5 m²/s for
max|U| 425 m²/s (1.7e-13), |msl| 3–15e-9 m (8e-17), spreads 0; zstar tail: laysum 0.8–1.3e-7 m,
w1 5.2–6.1e-10 m/s (1.3e-18), hlaw 2.9–3.4e-3 m (6.4e-12; the real32 rounding of 4000 m column sums of
`helem`, about 7 ulp relative, in the model's `se_H` and in the check alike).

**Why SE in SP closes worse than CG in SP, and why it is left as is.** The elevation is advanced in
the M = 50 substeps, `eta^{m+1} = eta^m + I(U*)·dtbt/areasvol`, and every substep rounds `eta` to
real32 on every node; these per-node roundings do not telescope. With CG the elevation is rounded once
per baroclinic step. So in SP `eta^{n+1} − eta^n` differs from `tau·I(<<U>>)/areasvol` by the
accumulated substep rounding (etacomp ~1e-6 m), the mean sea level does a random walk (pi: 12,000
roundings per node over 240 steps, of order 3e-6 m per node and 3e-6 m/√3140 ≈ 6e-8 m for the area
mean, measured 5e-8 m), and
under zstar the same mismatch shows up as w(1) ≈ etacomp/tau ≈ 5e-10 m/s and a volume drift
≈ msl/depth ≈ 2e-12 per 20 steps. The CORE2 month gives a global-mean SSH of +3.8e-9 m in SP with SE
(−9.8e-10 with CG). All of this is real32 rounding at well below the precision differences above;
nothing here shows that SE needs real64. If a tighter closure is ever wanted, two DP-neutral options:
commit `eta^{n+1} = eta^n + tau·ssh_rhs_old/areasvol` at the end of the step (one rounding, as CG;
mathematically equal to eta^M), or carry the substep elevation in WP_full. Neither was done.

**ctest in SP** (job 28268584): 84 of 176 pass. Against the SP branch's SP ctest (§4b): every
test that passed there still passes. Of the nine SE tests, the four `fesom_conserve_se_*` cases for
linfs and for the knob and CFL-guard refusals pass. `fesom_conserve_se_zstar_np1/np2` fail at their
TOL = 1e-12 (the same physics passes `run_conserve_pi.sh` at 1e-6), and `test_ssh_se_np1` fails 42
round-off assertions (adjointness, closed-basin sums, trim and eta-compatibility to ~1e-14, as in
§4b). `test_ssh_se_np2/np8` are not run because they need the np1 fixture. None of these failures
is a crash. `test_geom_mixed_np2` now times out instead of failing; it is one of the §4b tests that
pass `MPI_DOUBLE_PRECISION` on WP buffers, and split-explicit does not touch it.

**For the JAX port:** SE stays float32 in the SP policy (§6); only the SE_CHECK diagnostic
integrals are float64.
