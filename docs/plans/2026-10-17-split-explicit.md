# Split-explicit barotropic subcycling (SE) for FESOM_F

> ✅/⬜ Tick the `- [ ]` boxes (and add a ✅ to a Task header) when a sub-task is **gated DONE** (memory
> `feedback-tick-plan-checkboxes`). Read the actual code (FESOM_F, port_kokkos, the Zenodo SE file) and Sergey's
> notes, not summaries (memory `stay-close-to-fortran`). Design input: `docs/plans/2026-10-17-split-explicit-handoff.md`.
> The decisions below come from the brainstorm with Nikolay on 2026-10-17; the plan was revised after an automated
> plan review on the same day.

## Overview

Add a split-explicit (SE) barotropic solver to FESOM_F as an alternative to the semi-implicit CG SSH solve. Per
baroclinic step the 2-D system (η, Ū) is integrated in M explicit substeps with the Shchepetkin–McWilliams (2005)
AB3–AM4 dissipative scheme of Sergey's notes (`~/port_kokkos/ssh_sergey/subcycling.tex`; Banerjee et al. 2024,
GMD 17, 7051). Afterwards the 3-D velocities are trimmed to the time-mean transport ⟨⟨Ū⟩⟩. The existing ALE tail
(`vert_vel_ale`, tracers, `update_thickness_ale`) consumes the result through hbar/hbar_old, unchanged.

- **Why:** the SE solver has no global reductions; each substep needs two small 2-D halo exchanges. The main payoff is
  expected on the GPU in the later JAX port (stage B of the JAX GPU speed plan), which will be gated bitwise against
  this Fortran code. On the CPU it is measured, not assumed.
- **Scope:** linfs first (Dmitry's production configuration). zstar is designed in and gated on this branch,
  including restart, so that switching `FESOM3_WHICH_ALE` is the only step needed to use it.
- **Default unchanged:** with `FESOM3_SSH_SE` unset the CG path runs, and its outputs are **byte-identical to frozen
  main references** (G0, Task 0.1).
- **Not bitwise to CG** (the notes say so). SE is judged by invariants, determinism, restart exactness and a
  1-year SE-vs-SI twin.

## Context (from discovery)

### Where SE plugs in
- `src/step/mod_step_oce.F90:227-244`: `update_stiff_mat_ale` (non-linfs) → `compute_ssh_rhs_ale` → `solve_ssh_ale`
  (TMR_OCE_SOLVE) → `update_vel` → `compute_hbar_ale` → `update_eta_n`, all under TMR_OCE_SSH. SE replaces this
  block. The predictor (`compute_vel_rhs` :212, `viscosity_filter`, `impl_vert_visc_ale` :220) and the tail (GM,
  `vert_vel_ale` :270, tracers, `update_thickness_ale` :295) stay untouched.
- `step_oce(n, dt, lfirst, dynamics, tracers, mesh, Ki, heat_flux, water_flux, …, partit, stress_node_surf)`
  (:81-99). lfirst = `(n == 1) .and. .not. r_restart` (`fesom_lifecycle_native_mr.F90:965`) and `(n == 1)` in
  `fesom_conserve.F90:565/569`.
- Predictor facts (`src/oce/oce_dyn_velrhs.F90:94-148`):
  - UV_rhs = dt·(…)/elem_area is a velocity increment;
  - the per-layer −g∇ηⁿ comes from `gradient_sca` on `pre = -g*eta_n` (:108-112);
  - Coriolis is stored as (v·ff, −u·ff) = −f e_z×u in `uv_rhsAB(1,:,:,:)` (:119-120), **mixed with momentum
    advection** (which only runs for `momadv_opt` 1 or 2, :132-136);
  - ab1 = −(0.5+ε), ab2 = 1.5+ε, ε = 0.1; ff = 1 on lfirst; dynamics is AB2 only.
- `update_vel` (`src/oce/oce_ale.F90:115-176`): owned-element update, then `exchange_elem_shared(UV)` (if
  `partit%sync_shared_elem`, TMR_OCE_SHSYNC), then `exchange_elem(UV)`.
- `compute_hbar_ale` (:179-279):
  - an inflow edge-scatter over owned edges into `ssh_rhs_old` (:216-246; positive = convergence);
  - non-linfs subtracts `water_flux*areasvol` on nodes with `ulevels_nod2D == 1` (:253-259);
  - then `hbar = hbar_old + ssh_rhs_old*dt/areasvol` (:267) and `exchange_nod(hbar)`.
- zstar tail:
  - `vert_vel_ale` :433-452 spreads Δη = hbar − hbar_old over vertex layers `1 .. nlevels_nod2D−2`, with
    Htilde = zbar_3d_n(1) − zbar_3d_n(nlevels_nod2D−1), and applies `Wvel(1) −= water_flux`;
  - `update_thickness_ale` :582-637 commits hnode, rebuilds zbar_3d_n/Z_3d_n bottom-up (nNodL) and helem over the
    element's full range (owned), then runs `exchange_elem(helem)`.
- `mesh%area == mesh%areasvol` (vertex bottom, no cavities; `src/mesh/mod_mesh_areas.F90:515`). So the zstar surface
  w closes against the SE η equation without a correction.
- Other readers of the SSH state (none needs new code, but all must see consistent values):
  - the ice reads `mesh%hbar` over owned+halo (`src/ice/mod_ice_dyn.F90:95-100`);
  - output 'ssh' accumulates eta_n (`fesom_lifecycle_native_mr.F90:975`);
  - `FESOM3_STEP_DIAG` reads eta_n, uv and ssh_rhs_old (divsum);
  - `fesom_conserve.F90:644` reads ssh_rhs_old (the DIVSUM gate);
  - d_eta is read only by SI, restart and dumps; dhe only by `update_stiff_mat_ale`; ssh_rhs only by SI and dumps.

### Parallel infrastructure
- Ownership rule (FESOM2): a rank owns every element with at least one owned vertex, and every edge with at least one
  owned vertex. Both elements of an owned edge are therefore owned (`oce_ssh_rhs.F90:255-256`), so the inflow
  scatter reads only owned Ū. Border edges are computed on both ranks, which is why shared-element copies must agree
  (LESSONS L70).
- `elem2D_nodes` is stored **for owned elements only**. Halo centroids exist only after an exchange
  (`compute_geometry`, `mod_mesh_areas.F90:80-110`, which deallocates its centroid arrays afterwards). `elem_area` is
  full-halo valid (:101).
- `mesh%elem_neighbors(MAX_NV, nElemO)` (`mod_mesh_areas.F90:358-416`):
  - slot k is the edge between vertex slots k and k+1;
  - a boundary edge stores the element itself;
  - a triangle's slot 4 repeats slot 1;
  - neighbours are always in the eDim ring, so the standard `exchange_elem` suffices (test_vinv A5).
- `src/infra/mod_halo.F90`:
  - `exchange_nod` / `exchange_elem` accept a real (2, n) array (the `*_3D_r` cores).
  - `exchange_elem_shared` (:79-81) exists for integer rank-1 and real (d1, d2, elem) arrays only.
  - `shared_copy_max_elem` (:87-89, :436-481) exists for real rank-1 and (d1, d2, elem) arrays.
  - A real (2, elem) variant of both is needed.

### Knobs, drivers, timers
- **There are no namelists at run time:** `read_param_phys` is never called. Every run switch is an env variable read
  in the driver (`fesom_lifecycle_native_mr.F90:215-250`) or inside a module (`oce_ssh_solve.F90:80-86`). The SE
  knobs are read inside the SE module.
- Production driver: `fesom_lifecycle_native_mr` (`tools/prod_env.sh`). The conservation driver is `fesom_conserve`
  (dt = 1800 fixed, pi; its linfs checks are T10 no-leakage, helem consistency and finiteness, `fesom_conserve.F90:17-22,
  72-78`; its linfs drift is report-only). SE is wired into these two drivers; `step_oce` refuses SE anywhere else.
- Cold init ends just before the restart block (`fesom_lifecycle_native_mr.F90:909-925`: `restart_register_state`,
  then `restart_read` when `r_restart`).
- Timers (`src/infra/mod_timer.F90:26-62`): ids in pre-order; to add a nested timer, insert the id, bump the later ids
  and NTIMER, and extend TNAME / TPARENT / TSUBSET.
- Sources are globbed (`CMakeLists.txt:100-123`, CONFIGURE_DEPENDS).

### Restart
- `restart_register_state` (`src/io/mod_io_restart.F90:280-354`) registers fields depending on the config. A
  registered field missing from the store hits `zarr_check` → `error stop` (:730-789). RESTART_MAXF = 64 (~28 used).
- **Gap found in review:** a restart restores `hnode` only. `helem`, `zbar_3d_n`, `Z_3d_n` and `hnode_new` are never
  rebuilt after `restart_read`, so a zstar restart is not exact today, with or without SE. The restart gates have never
  run zstar. Task 4.1 closes this for the derived thickness fields. The SI-zstar restart has further gaps (dhe, the
  accumulated stiffness values, `oce_ssh_rhs.F90:229-248`); these are out of scope under D6 and are documented.

### Tests and gates
- ctest:
  - registration with `add_fesom_test(name np)` (CMakeLists.txt:168-183) and `add_mesh_test(test target meshdir np
    [ENV…])` (`test/CMakeLists.txt:66-72`);
  - the helper is `setup_test_mesh` (`test/test_util.F90:28-54`); each caller sets the default mesh;
  - **pi must be rotated** (`FESOM3_TEST_ROTATE=1`): unrotated pi has polar elements with no planar embedding;
  - 167 tests now (`TESTING.md:113`; :22/:385 still say 153).
- Build: `./configure.sh --compiler intel|gnu --precision dp --clean --build` → `build_intel_dp/`, `build_gnu_dp/`;
  `ctest -j 8 --output-on-failure`; `ulimit -s unlimited`.
- Diff tools: `tools/zarr_diff.py`, `tools/dump_diff.py` (`FESOM_DUMP_ALL` dumps).
- Gate scripts:
  - `tools/run_conserve_pi.sh` (invariants and tolerances, not a byte diff);
  - `tools/run_restart_gate_core2.sh` and `tools/run_restart_gate_multirank.sh` (split vs straight within one binary;
    **stale `F3` default → always pass `F3=`/`BUILD=`**; `run_seg` inherits the environment);
  - `tools/run_tri_regression.sh` (pass `NEW_BIN=`);
  - `tools/prod_env.sh` (production physics: vinv, rvo_upwind 1, TKE, GM+Redi, wsplit, JRA55-do from 1958, linfs,
    EVP).
- Cost reference: CORE2 `prod_env` ≈ 144 ms/step on one node (np 128); ≈ 1 s/step at np 8; several s/step at np 1.

### Prior art
- port_kokkos M12 (`~/port_kokkos/docs/SSH_SE_M12.md`, `docs/report/M12_SERGEY_PACKET.md`,
  `src/fesom_ssh_se.{h,cpp}`): the same scheme in C++ on FESOM2's element bottom. It provides the measured χ = 0.1,
  the CFL guard (`fesom_ssh_se.cpp:255-296`: node `mesh_resolution`, node depth, 0.5 safety, MPI min; CORE2 dt1800 →
  M_min 35), and the invariant levels used as orientation. Its G0 was a byte diff (`diff_snap rc=0`).
- Zenodo FESOM2.5 SE (`~/port_kokkos/ssh_sergey/zenodo_se/.../oce_ale_ssh_splitexpl_subcycl.F90`): the FB variant;
  used only for routine layout and names.

## Solution Overview (brainstorm decisions, 2026-10-17)

| # | fork | decision |
|---|---|---|
| D1 | time stepping | SM AB3–AM4 as in the notes, no filtering, χ = 0.1, M per mesh with a startup CFL guard |
| D2 | shared elements | sync the barotropic forcing F **once per step** (one owner). Every per-substep element operator is a per-element **gather** in fixed slot order, so the copies stay bitwise equal by construction; the copy spreads of Ū and ⟨⟨Ū⟩⟩ (= 0) prove it. Per substep: `exchange_nod(η)` + `exchange_elem(Ū)`. The trim keeps update_vel's uv sync |
| D3 | BT depth | H_e(η) = Hⁿ_e + Σ_v elem_w·c_{e,v}·(η_v − ηⁿ_v): the zstar law under the vertex bottom. c is recomputed once per step from zbar_3d_nⁿ; linfs is c ≡ 0. One depth function, no mode branch in the substep loop |
| D4 | Coriolis in r | stored sum (port_kokkos D4): šⁿ = Σhelemⁿ·uv^{n−1/2}, r_cor = f×(a₂šⁿ + a₁š^{n−1}). Exact under linfs, second order under zstar. `compute_vel_rhs` untouched |
| D5 | BT viscosity | the notes' **linear** harmonic, A_h = `se_visc_ah`·l_e (0.01 m/s; 0 = off), flux form H_edge·A_h·∇(Ū/Hⁿ) with Hⁿ frozen → V(Ūᵐ) − V(Ūⁿ) = V(Ūᵐ − Ūⁿ). BT bottom drag off |
| D6 | restart | strict: SE fields only when SE is on; a checkpoint without them aborts with a clear message; no SI↔SE switching; M and χ must stay fixed across a restart chain (documented) |
| D7 | order | linfs np1 → linfs np>1 → zstar → restart → physics (G3) |

Not in scope: NEMO SE runs, the zstar 1-year twin, wide halo, JAX dump stages, the γ1 flow-dependent viscosity term
(added only if the screens show it is needed), SE→SI and SI→SE restarts, the SI-zstar restart gaps (dhe, stiffness).

## Development Approach

- **Testing approach: regular.** In each task, write the code and then its unit test. The gate for each stage runs on
  pi/CORE2.
- Complete each task fully before the next; small focused changes; all tests green before moving on.
- **G0 after every stage** (SE off): ctest Intel + GNU green, **and** a byte diff against the frozen references of
  Task 0.1 (`zarr_diff.py`, `dump_diff.py --rel-floor 0`, conserve_pi log diff excluding timings), **and** the restart
  gates green.
- **No compute on the login node** (memory `feedback-no-compute-on-login-node`). Builds, ctest and gates run inside an
  interactive/shared allocation. Every sbatch gets `#SBATCH --hint=nomultithread` (plus `--mem=0` for whole nodes).
  `env.sh` provides the knem workaround. Waiting on Slurm uses a background shell watcher on `sacct`, never
  model-driven polling.
- **Cost, Stages 0–4:** every gate run is at most ~15–20 min:
  - pi;
  - CORE2 np 1 for ≤ 48 steps;
  - CORE2 np 8 for 480 steps (~8–10 min);
  - CORE2 np 128 on one node.

  Total < 2 node-hours. Stage 5 has its own estimate and needs a separate yes.
- **Measure, do not guess:** invariant thresholds come from the measured levels with a margin, and perf is claimed only
  from timers.
- **SE silent when off:** no output from `se_init` or SE_CHECK when SE is off. `run_conserve_pi.sh` and
  `run_tri_regression.sh` parse fixed output lines.
- **Commits:** self-contained (the module plus small hooks), so that the branch rebases onto `jax-oracle`. No session
  links or tool markers. Nothing is pushed without Nikolay's approval of the exact text.
- Update this plan when scope changes (➕ new tasks, ⚠️ blockers).

## Testing Strategy

- **Unit (ctest):** a new `test/test_ssh_se.F90` on **rotated** pi (`add_mesh_test … FESOM3_TEST_ROTATE=1`) at np 1,
  2 and 8. Each test targets a specific failure: literal coefficient values, operator/r-term consistency, dtbt
  convergence, exact identities, copy spreads, and the zstar depth law through the real tail.
- **Invariant suite `FESOM3_SE_CHECK`** (in the module, rank-0 line every step, NaN-loud abort):
  - in `se_trim_finish`, while helem is still at n:
    - η-compatibility: max|η^{n+1} − ηⁿ − τ(I(⟨⟨Ū⟩⟩) − W·areasvol)/areasvol|, where I is the inflow scatter;
    - trim: max|Σₖhelemⁿ·uv − ⟨⟨Ū⟩⟩|;
    - mean-sea-level change in metres, Σareasvol·(η − η_start)/Σareasvol (zstar: also against the integrated
      freshwater);
    - copy spreads of Ū and ⟨⟨Ū⟩⟩;
  - in a second hook after the tail (zstar only): layer sum max|Σhnode − (H0 + η)| and max|w(1)|.
- **Gates:** G0 (above), G1 invariants, G2 determinism and restart, G3 physics and perf (Stage 5).
- No UI / e2e.

## Progress Tracking

- Mark `[x]` immediately; add ✅ to a Task header when it is gated; ➕ new tasks; ⚠️ blockers.
- Record measured numbers (invariant levels, copy spreads, M_min, timings) in the task line that gates them.

## What Goes Where

- **Implementation Steps** (checkboxes): code, unit tests, gate runs, docs in this repo.
- **Post-Completion** (no checkboxes): the Sergey/Dmitry question packet, a production trial in Dmitry's runs, the
  JAX port, NEMO.

## Implementation Steps

### Stage 0 — Baseline references, infrastructure, module skeleton, knobs
*Everything SE needs to exist and be switched off byte-identically before any physics goes in.*

#### Task 0.1: Baseline build and frozen G0 references ✅

**Files:** Create: `tools/run_g0_se.sh` (build dirs; references under `/work/ab0995/a270088/fesom3_se_ref/`; job
scripts and logs in its `jobs/`)

- [x] build Intel and GNU at HEAD (code = main 814e5d0 plus docs only) inside an allocation; ctest Intel + GNU,
  record the count (expect 167/167). 2026-10-18, job 28214504 (shared, 32 cores; no `--clean`, fresh build dirs):
  **Intel 167/167** (ctest 27 s). **GNU 156/167**: 11 failures that already exist on main, all deterministic
  (a serial `ctest -j 1` rerun, job 28215528, fails the same way):
  - ⚠️ `test/test_coupling_mixed.F90:193` does not compile with gfortran 11.2 (`allocate(tracers%data(2),
    tracers%data(2)%values(nl-1, nn))`: "Allocate-object at (1) is subobject of object at (2)"). The 5
    `test_coupling_mixed_*` tests have no executable; every other GNU target builds.
  - ⚠️ `test_wimpl_tra_np1/np2`: 4 × "FAIL: C6 (i) the limiter leaves the antidiffusive flux unchanged" /
    "every limiting factor applied to a nonzero face == 1", w > 0 cases only (GNU prints fct_minus = -0.000).
  - ⚠️ `meshdiag_mixed/quad/pi_w`: "FAIL M1 face_nodes / face_edges / face_links / gradient_sca_x / gradient_sca_y
    dims ['n4', 'el']" at np 1 and 2.
  - ⚠️ `test_shared_elem_np1`: SIGSEGV after "S2: exchange and probes done" (the np 2/8 variants pass).

  Until main fixes these, GNU G0 on this branch means "the same 11 failures and no new one".
- [x] freeze the Intel binaries (copy plus `SHA256`) and write the references:
  `/work/ab0995/a270088/fesom3_se_ref/bin_main_6cd6449/` (read-only; `SHA256SUMS`; `PROVENANCE`: src/test/CMake
  identical to 814e5d0), `fesom_lifecycle_native_mr` sha256 `2b6d1ab1c3fa4303…`, `fesom_conserve` `15eb8fd05d93d1cf…`.
  The references are in `…/fesom3_se_ref/g0/`, written by `MAKE_REF=1 tools/run_g0_se.sh` with the frozen binaries
  (job 28214520), so the reference runs and the gate runs share one case definition:
  - CORE2 with the full production physics `prod_env 1e-5` (JRA55-do from 1958-01-01, linfs, EVP, TKE, GM + Redi,
    splines, vector-invariant momentum with rvo_upwind 1, w split, SW penetration, step diagnostics; SSH soltol 1e-5
    = production), np 2, 20 steps (dt 1800). It has an end-of-run checkpoint `fesom.1958.001.36000` (28 stores, 99
    variables) and `FESOM_DUMP_ALL` node dumps of **step 20 only** (1 649 154 records; all 20 steps would be ~6 GB
    per case). 83 s; 1.3 GB.
  - the same with `FESOM3_WHICH_ALE=zstar`: 85 s, 1.3 GB.
  - the `run_conserve_pi.sh` log: 26 configs, GATE OK, 50 s. A frozen copy of the script sits next to the log, so
    the SE blocks that later tasks add to `run_conserve_pi.sh` do not change the G0 case set.
- [x] write `tools/run_g0_se.sh`: rerun those cases with a new build (SE off) and compare against the references
  (`zarr_diff.py` max|Δ| = 0, `dump_diff.py --rel-floor 0`, conserve log diff without timing lines). It passes on
  the frozen binary itself. The script also compares the case environment (catches drift of the case definitions),
  the checkpoint store list and `checkpoint.json` (without `fesom_git`), and the 20 `FESOM3_STEP_DIAG` lines.
  It never deletes: each run goes into a fresh dir (default `/scratch/a/a270088/g0_se/<stamp>`). The conserve log
  has no timing lines today (the filter drops 0 of 249 lines). Self-tests (job 28214520):
  - frozen bin: **G0 GREEN**, 276 s;
  - `build_intel_dp/bin`: **G0 GREEN**, 271 s;
  - negative control, the GNU build: **FAIL** in all three cases, as it should (83 resp. 84 of 99 checkpoint
    variables differ, dumps and step lines differ, 222 conserve log lines differ), 619 s.
- [x] `F3=<worktree> tools/run_restart_gate_core2.sh` (np 1/2) green; record the wall times. Both GREEN, max|Δ| = 0
  (checkpoint and live dump, mid-run and month-boundary splits): **np 1 306 s** (job 28214521), **np 2 144 s**
  (job 28214522). Run dirs: `/scratch/a/a270088/fesom3_se_rgate/np{1,2}_<jobid>`.

#### Task 0.2: Shared-element exchange for real (2, elem) arrays ✅

**Files:**
- Modify: `src/infra/mod_halo.F90`
- Modify: `test/test_shared_elem.F90`

- [x] add `exchange_elem_shared_3D_r` (real (:, elem); the `*_3D_r` naming of `exchange_elem_3D_r`) and a rank-2
  `shared_copy_max_elem` variant with its own probe, in the generic interfaces (:79-89). Done:
  `exchange_elem_shared_3D_r` (wrapper over `core_3D_r` on `com_elem2D_shared`), `shared_copy_max_elem_3D_r` and
  `probe_3D` (additions only)
- [x] test: on pi np 2/8, fill a (2, elem) array with rank-dependent values on shared elements. After the sync the
  copy spread is exactly 0, and the designated owner's values are kept. Before the sync the probe reports a non-zero
  spread. Added to S2 of `test_shared_elem.F90`; the pre-sync probe must equal the spread computed from the designated
  owners; after the sync every receive slot equals the owner's value, all other slots are bitwise unchanged, probe = 0.
  Global pre-sync spread printed: 1.00E+07 (pi np 2), 5.00E+07 (pi np 8), 0 (np 1). Existing registrations cover it
- [x] ctest green (Intel + GNU). **Intel 167/167.** GNU needed the allocate fix of `test_coupling_mixed.F90:193`
  (the same hunk as jax-oracle 7e6ceac, its own commit); with it **GNU 161/167**, and the base 6cd6449 plus only that
  fix fails the same 6 (`test_wimpl_tra_np1/np2`, `meshdiag_mixed/quad/pi_w`, `test_shared_elem_np1`: SIGSEGV in S3
  after S2, GNU np 1 only). GNU pi np 2/8 of `test_shared_elem` pass incl. the new checks. Jobs 28214499, 28215492,
  28216020 (~0.05 node-h). From here on the GNU baseline is **these 6 failures, no new one**

#### Task 0.3: SE state in t_dyn, env knobs, `se_init`, CFL guard, driver guard, timer ✅

**Files:**
- Create: `src/oce/oce_ssh_se.F90`
- Modify: `src/types/mod_dyn.F90`
- Modify: `src/infra/mod_timer.F90`
- Modify: `src/drivers/fesom_lifecycle_native_mr.F90`, `src/drivers/fesom_conserve.F90`
- Modify: `src/step/mod_step_oce.F90`
- Create: `test/test_ssh_se.F90`; Modify: `test/CMakeLists.txt`

- [x] `t_dyn` (`mod_dyn.F90:104-146`): add the scalars and arrays of Technical Details §1. Extend
  `write_t_dyn`/`read_t_dyn` symmetrically only if test_types needs it. test_types does not need it, so they are
  not serialized (the carried arrays go to the Zarr checkpoint in Task 4.2). ➕ One scalar beyond §1:
  `se_m_force` (FESOM3_SE_M_FORCE), so that the parser writes every knob into `t_dyn` and the test can check it
- [x] `oce_ssh_se`:
  - `se_env_requested()` (parses `FESOM3_SSH_SE` once, 0/1, any other value dies);
  - `se_read_env` for `FESOM3_SE_M` (int ≥ 1), `FESOM3_SE_CHI` (0 ≤ χ ≤ 0.25), `FESOM3_SE_VISC_AH` (≥ 0),
    `FESOM3_SE_CHECK` (0/1), `FESOM3_SE_M_FORCE` (0/1). Values are checked as for `FESOM3_SHARED_SYNC`.
    The parsing is `se_parse_knob(name, str, dyn, ok, msg)` (no output, no stop; switches exactly `0`/`1`, M digits
    only, the reals plain-number characters and a finite value, so `nan`, `inf` and `0.1 0.2` are rejected); the
    driver-facing wrappers stop through `se_die` (MPI_Abort)
- [x] `se_init(dyn, mesh, partit, dt)`:
  - silent no-op when SE is off;
  - guards: `AB_order == 2`; no cavity (`ulevels_nod2D > 1` anywhere → die);
  - allocates the state and computes the SM coefficients;
  - cold start: η ring ← eta_n, Ū ring ← Σhelem·uv (nElemL), š_old ← 0;
  - CFL guard as in port_kokkos: owned surface nodes, `mesh_resolution`, depth from the vertex column, 0.5 safety;
    **allreduce MIN** of the limit before deciding; print M_min on rank 0; die if M < M_min unless FORCE;
  - sets `se_ready`

  The guard is factored into `se_cfl_dtbt_limit` (limit, MIN-reduced), `se_cfl_m_min` and the pure
  `se_cfl_decision` (OK / FORCED / DIE). **pi (dt 1800 s): limit 233.653 s, M_min = 8** (np 1 and np 2).
- [x] call `se_init` in `fesom_lifecycle_native_mr` right before the restart block (:909) and in `fesom_conserve`
  after its cold init (after the sea-ice setup, before `check_invariants(0)`)
- [x] `step_oce`, first call: if `se_env_requested()` and not `se_ready`, die ("FESOM3_SSH_SE=1 but this driver does
  not initialise split-explicit"). Without this check, other drivers would silently run SI. ➕ Until Task 1.3 adds
  the SE branch, `se_step_guard` also stops an *initialised* SE run ("step_oce has no split-explicit step yet"), for
  the same reason; **Task 1.3 removes that second stop**
- [x] timer `TMR_OCE_SE_BT` nested under TMR_OCE_SSH (ids, NTIMER, TNAME/TPARENT/TSUBSET). ➕ A `TQUIET` flag
  (one extra MPI_MAX of the counts in `timer_report`) drops the line when the timer never ran, so SI runs print
  no zero SE line (checked in the G0 linfs log). No other code uses the ids by number
- [x] tests (`test_ssh_se`, rotated pi, np 1):
  - the coefficients at χ = 0.1 equal **literal numbers** worked out by hand from the notes (ζ = −0.003689497,
    γ = 0.0320866, δ = 0.724707606), which catches a typo in the constants;
  - M_min on pi is finite, identical on all ranks at np 2, and the guard fires for M = 1;
  - env parsing rejects bad values

  Registered as `test_ssh_se_np1` and `test_ssh_se_np2` (`FESOM3_TEST_ROTATE=1 FESOM3_SSH_SE=1 FESOM3_SE_M=40`):
  C1 coefficients (also χ = 0), C2 32 rejected / 11 accepted knob values, C3 the limit against an independent
  per-rank recomputation and the decision at M = 1, M_min − 1, M_min, 50, C4 `se_init` from the environment
  (extents, cold-start rings, coefficients, `se_ready`). Plus three regex ctests of the driver-facing stops in
  `fesom_conserve` (`fesom_conserve_se_{badswitch,badm,cfl}_np1`)
- [x] **GATE G0:** ctest Intel + GNU; `tools/run_g0_se.sh` byte-identical; restart gate green. 2026-10-18, job
  28216244 (shared, 32 cores, 12.7 min, 0.16 node-h): **Intel 172/172** (167 + 5 new). **GNU 161/172**, the same 11
  pre-existing failures as Task 0.1 and no new one; the 5 new tests pass on GNU. **G0 GREEN** (linfs, zstar,
  conserve byte-identical to the frozen main references; 281 s). **Restart gate GREEN, max|Δ| = 0: np 1 203 s,
  np 2 149 s.** New `fesom_lifecycle_native_mr` sha256 `103151b26b8445df…`. Job scripts and logs:
  `/work/ab0995/a270088/fesom3_se_ref/jobs/t03_*` (verbose test output: job 28216531)

### Stage 1 — linfs, one rank
*The whole scheme on the simplest decomposition, with every invariant visible.*

#### Task 1.1: Barotropic forcing (`se_bt_forcing`) and the element depth function ✅

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`
- Modify: `test/test_ssh_se.F90`

- [x] `se_elem_depth`: Hⁿ_e = Σₖhelemⁿ over `ulevels(e)..nlevels(e)−1` for **owned + eDim** elements (nElemL;
  helem is halo-valid). c_{e,v} ≡ 0 under linfs (the zstar branch comes in Task 3.1). ➕ Until Task 3.1 a
  non-linfs run stops in `se_elem_depth` ("split-explicit supports which_ALE = linfs only so far"), so a zstar SE
  run cannot silently use c = 0 once Task 1.3 wires SE in; **Task 3.1 replaces that stop**
- [x] `se_bt_forcing`, on owned elements:
  - G = Σₖhelemⁿ·UV_rhs;
  - šⁿ = Σₖhelemⁿ·uv;
  - F = G/τ + g·Hⁿ·∇ηⁿ + f·e_z×(a₂šⁿ + a₁š^{n−1}), with ∇ through `gradient_sca` exactly as
    `oce_dyn_velrhs.F90:105-111`, a₂ = 1.5+ε (1 on lfirst), a₁ = −(0.5+ε);
  - copy Ūⁿ (ring level 0, nElemL) into `se_ubtn` for the viscosity;
  - then š_old ← šⁿ

  In code: F(1) = G(1)/τ − Hⁿ·Fx − f·S(2), F(2) = G(2)/τ − Hⁿ·Fy + f·S(1), with Fx, Fy = gradient_sca·(−gηⁿ)
  (the compute_vel_rhs expression) and S = a₂šⁿ + a₁š^{n−1}. ➕ The one-owner sync of F
  (`exchange_elem_shared(se_force)` under `partit%sync_shared_elem`, timer TMR_OCE_SHSYNC, a no-op at np 1) is
  already the last statement of `se_bt_forcing` (§2 pseudo-code); Task 2.1 keeps only its tests
- [x] test (pressure part): uv = 0, pgf = 0, a non-trivial eta_n, the **real** `compute_vel_rhs` → F = 0 to rounding.
  `test_ssh_se` C5 (np 1 and np 2): **max|F| = 1.4e-16 m²/s² against max|gHⁿ∇ηⁿ| = 0.39 m²/s² (ratio 3.6e-16
  Intel, 3.5e-16 GNU)**; tolerance 1e-12 × the pressure term. C5 also checks se_H == Σₖhelem bitwise on owned + eDim
- [x] test (Coriolis part): **momentum advection off** (`momadv_opt = 0`, so neither branch at :132-136 runs),
  UV_rhsAB built by a previous real `compute_vel_rhs` call from a known uv^{n−3/2}, š_old consistent, helem constant
  → F = 0 to rounding; plus the lfirst coefficient. `test_ssh_se` C6: step 1 (lfirst, cold uv_rhsAB, š_old = 0)
  **max|F| = 7.4e-18**, step 2 (AB2 blend of uv^{n−1/2} and uv^{n−3/2}) **max|F| = 1.3e-17 m²/s²**, against
  max|f·š| = 3.8e-3 (ratios 1.9e-15 and 3.4e-15; GNU 5.4e-18 / 1.2e-17). The lfirst weight: the step-1 UV_rhs taken
  as a non-lfirst step leaves exactly (0.5+ε)·f e_z×šⁿ (residual 7.4e-18). š_old == šⁿ and se_ubtn == ring level 0
  bitwise. Tolerance 1e-12 × max|f·š|
- [x] ctest green. Job 28216760 (shared, 2 min): **Intel 172/172; GNU 166/172, the 6 known failures**
  (`test_wimpl_tra_np1/np2`, `meshdiag_mixed/quad/pi_w`, `test_shared_elem_np1`), no new one; `test_ssh_se_np1/np2`
  pass on both. Script `/work/ab0995/a270088/fesom3_se_ref/jobs/t11_build_ctest.sbatch`

#### Task 1.2: Substep loop (`se_subcycle`), inflow scatter, viscosity ✅

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`
- Modify: `test/test_ssh_se.F90`

- [x] `se_inflow(U, out)`: the inflow edge-scatter of `compute_hbar_ale` (:216-246) over owned edges, with the 2-D Ū
  in place of Σhelem·uv (same edge order; positive = convergence, the `ssh_rhs_old` sign). Signature
  `se_inflow(mesh, partit, U, out)`; `out` is zeroed on nNodL and complete on owned nodes (halo entries partial,
  as `ssh_rhs_old` in `compute_hbar_ale`)
- [x] `se_kappa` (built in `se_init` for SE runs, moved here next to its only consumer):
  - owned centroids via `elem_center`, then `exchange_elem` for the halo neighbours (as `compute_geometry`
    :80-89; `elem2D_nodes` is owned-only);
  - edge length l from the element's own vertices; centroid distance d; κ_{e,k} = A_h·l/(d·A_e);
  - κ = 0 on boundary slots and on a triangle's slot 4

  Routine `se_kappa_build`, called by `se_init` after the cold start. The longitude metric of l and d is the mean
  elem_cos of the two elements (edge_len's, `compute_edge_geometry`), so l = edge_len on owned edges and l, d,
  H_edge are bitwise symmetric between the two elements of an edge. pi (A_h = 0.01·l): max κ = 2.1e-6 s⁻¹
- [x] `se_visc(dU, out)`: a gather over `k = 1..elem2D_nnodes(e)` of `elem_neighbors`, H_edge = ½(Hⁿ_e + Hⁿ_nb),
  V_e = Σ_k κ_{e,k}·H_edge·(dU_nb/Hⁿ_nb − dU_e/Hⁿ_e)
- [x] `se_subcycle` for m = 0..M−1, following Technical Details §2 with the halo extents of §1:
  - the tendency is a separate routine `se_tendency(Ustar, etaA, HA, …)` so that it can be tested;
  - η levels are kept on nNodL;
  - rings are shifted **by copying**, never by index rotation (the ring offset must not become restart state);
  - `exchange_elem(Ū)` is skipped when `se_visc_ah = 0`;
  - TMR_OCE_SE_BT

  ➕ H^A comes from `se_depth_at(eta, HA)` = Hⁿ + Σ_v elem_w·c·(η − ηⁿ) on owned elements, the one depth function
  of D3 (exactly Hⁿ under linfs, c ≡ 0). η⁰ = eta_n is copied into `se_etam` at entry, ⟨⟨Ū⟩⟩ is zeroed there; U*,
  U^{m+1}, the tendency, dU, η^A, I(U*) and H^A are local allocatables of `se_subcycle` (not state, so not in §1).
  On exit η^M is in `se_etam` (η^{M−1}, η^{M−2} in `se_eta(:,1:2)`, all nNodL and halo-valid), Ū^M in ring level 0,
  ⟨⟨Ū⟩⟩ in `se_umean` (owned); eta_n/hbar untouched (Task 1.3). ⚠️ With `se_visc_ah = 0` the eDim entries of the
  Ū ring are never refreshed after the cold start; nothing reads them then, but Task 4.2/4.3 must not expect
  halo-valid Ū levels in that mode (the partition-independent checkpoint stores owned values)
- [x] tests (`test_ssh_se` C7-C9, np 1 and np 2, Intel and GNU; the existing registrations cover them):
  - **operator/r-term consistency:** with U* = šⁿ = š_old, η^A = ηⁿ, H^A = Hⁿ and dU = 0, `se_tendency` = G/τ to
    rounding. This catches a sign or factor mismatch between F and the operator, e.g. in Coriolis. C7 (two real
    `compute_vel_rhs` calls on the same uv, non-trivial eta_n and pgf): **max|tend − G/τ| = 2.8e-17 m²/s² of the
    largest term 0.39 (7.1e-17; GNU 1.4e-16)**, tolerance 1e-12;
  - **dtbt convergence:** a linear gravity wave on rotated pi, one baroclinic step with M, 2M and 4M against an 8M
    reference at χ = 0; observed order ≥ 2. This catches swapped ring levels or misplaced coefficients. C9d: a 1 m
    Gaussian bump (e-folding 1500 km) at rest, F = 0, viscosity on, max|η^M − η⁰| = 0.128 m; M = 16/32/64 vs 128:
    **η errors 1.32e-4 / 3.23e-5 / 6.54e-6 m, observed orders 2.04 and 2.30; Ū orders 2.13 and 2.23** (identical
    at np 1/2 and Intel/GNU). Second order is what a cold start gives: with flat rings the first two substeps
    extrapolate U* and η^A to first order only (a pure second-order error reads 2.07/2.32 against an 8M
    reference). In production only the first step starts cold;
  - rest stays at rest bitwise (C9b, M = 40: every ring level, η^M and ⟨⟨Ū⟩⟩ exactly 0);
  - Σ_v areasvol·(η^M − η⁰) = 0 to rounding (closed basin, W = 0). C9c (hashed η, three Ū levels and F, M = 40,
    viscosity on): **6.6e-5 m³ of Σ areasvol·|Δη| = 2.9e13 m³ (2.3e-18)**, np 2 1.5e-5, GNU ≤ 2.7e-4 m³;
  - η-compatibility η^M − η⁰ = τ·I(⟨⟨Ū⟩⟩)/areasvol to rounding. C9c: **max residual 3.4e-16 m** of
    max(|η⁰|, |Δη|) = 0.41 m (GNU 3.1e-16); tolerance 1e-12;
  - Σ_e A_e·V_e = 0 to rounding (this also catches a double-counted triangle slot); V of a uniform velocity over a
    flat bottom is 0. C8: **Σ A_e·V_e = (6.1e-7, 3.5e-8) of Σ A_e·|V_e| = 1.2e11 (5e-18)**, each element counted
    once (on its first vertex's owner) at np 2; V(uniform, flat bottom) == 0 exactly, and also 0 over the real
    bottom; κ = 0 on boundary slots and triangle slot 4, > 0 elsewhere, and equal to an independent recomputation
    from edge_len and the centroid distance to 4.3e-16
  - ➕ C9a, M = 1: the rings shift by copying (bitwise), ⟨⟨Ū⟩⟩ = U*, η¹ and Ū¹ equal the operators called directly
    (to 1e-13: GNU's `-march=znver3` contracts `a + b*c` into FMAs differently in test and module); C9c at np 2:
    η levels and Ū^M halo-valid after the substeps (stale-halo probes 0)
- [x] ctest green. Job 28216792 (shared, 2 min): **Intel 172/172; GNU 166/172, the 6 known failures**, no new one.
  GATE G0 (SE off): job 28216833 (5.2 min), **G0 GREEN**: linfs and zstar CORE2 (99 checkpoint variables,
  1 649 154 dump records max|Δ| = 0, 20 step lines equal) and conserve (249 log lines, 26 configs) byte-identical
  to the frozen main references; new `fesom_lifecycle_native_mr` sha256 `5ccb664fe4ed9d37…`. Scripts
  `/work/ab0995/a270088/fesom3_se_ref/jobs/t11_build_ctest.sbatch` (TAG=t12), `t12_g0.sbatch`

#### Task 1.3: Trim, finish and the `step_oce` branch

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`
- Modify: `src/step/mod_step_oce.F90`
- Modify: `test/test_ssh_se.F90`

- [ ] `se_trim_finish`:
  - u** = uv + UV_rhs;
  - uvₖ = u**ₖ − (Σⱼhelemⁿⱼu**ⱼ − ⟨⟨Ū⟩⟩)/Hⁿ_e on owned elements;
  - `exchange_elem_shared(uv)` (under `sync_shared_elem`, TMR_OCE_SHSYNC), then `exchange_elem(uv)`, exactly as in
    `update_vel`;
  - on **nNodL**: hbar_old = hbar; hbar = η^M; eta_n = η^M (the ice and `compute_vel_rhs` read halo vertices);
  - ssh_rhs_old = I(⟨⟨Ū⟩⟩) − W·areasvol (non-linfs);
  - Ū^M stays in the ring
- [ ] `step_oce`: `if (dynamics%use_ssh_se) then` call the three SE routines under TMR_OCE_SSH, `else` the existing
  block **verbatim**. Add `dump_node_2d` for hbar/eta_n at the same substep ids
- [ ] ➕ remove the temporary second stop in `se_step_guard` ("step_oce has no split-explicit step yet", Task 0.3)
- [ ] test: Σₖhelemⁿ·uv = ⟨⟨Ū⟩⟩ to rounding after the trim; the correction is depth-uniform in velocity
- [ ] ctest green; G0

#### Task 1.4: Invariant suite `FESOM3_SE_CHECK`

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`, `src/step/mod_step_oce.F90` (tail hook)
- Modify: `test/test_ssh_se.F90`

- [ ] `se_check_core`, called inside `se_trim_finish`:
  - η-compatibility, trim, mean-sea-level change in metres since the start of the run;
  - copy spreads of Ū and ⟨⟨Ū⟩⟩ (meaningful from Stage 2 on);
  - max|η|, max|Ū|;
  - NaN/Inf → die with the step number
- [ ] `se_check_tail`, called in `step_oce` after `update_thickness_ale` when SE and SE_CHECK are on: the zstar layer
  sum and max|w(1)| (filled in by Task 3.1)
- [ ] reductions via `allreduce_max`/`allreduce_sum`; nothing is computed or printed when off
- [ ] test: on a valid state the checks report rounding-level values; with a deliberately perturbed uv on one element
  the trim check reports the perturbation
- [ ] ctest green

#### Task 1.5: GATE Stage 1 — linfs np 1

- [ ] pi, `fesom_conserve`, `FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1 FESOM3_WHICH_ALE=linfs`, 1000 steps: SE_CHECK
  invariants at rounding (record the levels); `fesom_conserve`'s linfs checks green (T10 no-leakage, helem
  consistency, finiteness). Heat and salt drift reported only, as for SI linfs
- [ ] CORE2 `prod_env`, np 1, 48 steps (~6–10 min) with SE and with SI: no NaN, invariants at rounding, max|η| and
  max CFL_z comparable
- [ ] record the measured levels here and set the gate thresholds (margin ×100)

### Stage 2 — linfs, many ranks
*The risk port_kokkos never had: shared border elements under the vertex bottom.*

#### Task 2.1: One-owner forcing sync and copy-spread checks

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`
- Modify: `test/test_ssh_se.F90`, `test/CMakeLists.txt`

- [ ] `exchange_elem_shared(se_force)` after `se_bt_forcing` (under `sync_shared_elem`)
- [ ] tests (`test_ssh_se` np 2/8 on rotated pi):
  - before the F sync the spread is > 0 (R̄ roundoff exists);
  - after K steps of the barotropic loop the copy spreads of Ū and ⟨⟨Ū⟩⟩ are exactly 0;
  - the inflow-sum ratio is ≤ 1e-14;
  - np 1 vs np 2 vs np 8: η agrees to roundoff (record the level)
- [ ] ctest green

#### Task 2.2: GATE Stage 2 — linfs np > 1

- [ ] add an SE block to `tools/run_conserve_pi.sh` (linfs + ice, np 2 and 8, `FESOM3_SSH_SE=1`) next to the
  SHARED block; the existing blocks stay unchanged
- [ ] CORE2 `prod_env`, np 8 (480 steps) and np 128 on one node (480 steps), `FESOM3_STEP_DIAG=1 FESOM3_SE_CHECK=1`:
  - divsum ≤ 1e-13;
  - dUV = dAB = dsig = 0;
  - SE copy spreads = 0;
  - invariants at the Stage 1 level
- [ ] determinism: the same binary at the same np, run twice, is bitwise identical (`dump_diff.py --rel-floor 0`);
  np 1 vs np 8 difference recorded
- [ ] G0

### Stage 3 — zstar
*Make "switch zstar on" a tested fact.*

#### Task 3.1: zstar depth law and freshwater

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`
- Modify: `test/test_ssh_se.F90`

- [ ] `se_elem_depth` under zstar, from **zbar_3d_n differences** exactly as `vert_vel_ale` (:438-444) does:
  - Htilde_v = zbar_3d_n(1,v) − zbar_3d_n(nlev_v−1,v);
  - c_{e,v} = (zbar_3d_n(ule(e),v) − zbar_3d_n(K_{e,v}+1,v))/Htilde_v, with K_{e,v} = min(nle(e)−1, nlev_v−2);
  - recomputed once per step;
  - the substep uses H^A = Hⁿ + Σ_v elem_w·c·(η^A − ηⁿ)
- [ ] W in the η equation when `which_ALE /= 'linfs'` (mask `ulevels_nod2D == 1`, as `compute_hbar_ale` :253-259)
- [ ] `se_check_tail`: layer sum max|Σhnode − (H0 + η)|, max|w(1)|; `se_check_core`: mean-sea-level change against
  the integrated freshwater
- [ ] tests:
  - c = 1 at the element's shallowest vertex and 0 < c ≤ 1 elsewhere;
  - H_e(ηⁿ) = Σhelemⁿ to rounding;
  - **decisive:** apply a synthetic Δη through the real `vert_vel_ale` + `update_thickness_ale`; then
    Σhelem^{n+1} = H_e(η^{n+1}) to rounding on every element;
  - a uniform W with Ū = 0: η^M − η⁰ = −τW and ΔV = −τΣW·areasvol to rounding;
  - under linfs c ≡ 0
- [ ] ctest green

#### Task 3.2: GATE Stage 3 — zstar np 1 and np > 1

- [ ] `run_conserve_pi.sh` zstar cases with SE (np 1/2/8): hard heat/salt conservation as in the existing zstar
  cases; layer sum and w(1) at rounding
- [ ] CORE2 zstar np 8 (480 steps) and np 1 (48 steps), SE_CHECK: all invariants and copy spreads; record the levels
- [ ] G0

### Stage 4 — Restart

#### Task 4.1: zstar derived thickness after a restart

**Files:**
- Modify: `src/drivers/fesom_lifecycle_native_mr.F90`

- [ ] after `restart_read` (:923), when `which_ALE == 'zstar'`: set hnode_new = hnode on nNodL, then call
  `update_thickness_ale(mesh, partit)`. This rebuilds zbar_3d_n and Z_3d_n bottom-up from the fixed bottom anchor,
  and helem (owned plus `exchange_elem`), with the same loops as the straight run, so the result is bitwise. linfs is
  untouched (G0)
- [ ] verify: `run_restart_gate_core2.sh` with `FESOM3_WHICH_ALE=zstar` and **SI** shows whether the remaining
  SI-zstar gaps (dhe, stiffness) still break SI; record the result, do not fix it (D6). The split-explicit zstar
  restart is gated in Task 4.3

#### Task 4.2: SE fields in the checkpoint

**Files:**
- Modify: `src/io/mod_io_restart.F90`

- [ ] in `restart_register_state` (:280-354), `if (dyn%use_ssh_se)`:
  - `se_eta1`, `se_eta2` (node, default halo);
  - `se_ubt{0,1,2}_{u,v}` (elem, `RESTART_HALO_ELEM`);
  - `se_sbar_old_{u,v}` (elem, `RESTART_HALO_NONE`);
  - the rank-0 count of registered fields grows by 10
- [ ] pre-check on read: SE on and the checkpoint has no `se_eta1/.zgroup` → `die('checkpoint has no split-explicit
  state: written with FESOM3_SSH_SE off; SI<->SE restarts are not supported')`
- [ ] verify that both legs of the restart gates inherit `FESOM3_SSH_SE`, `FESOM3_SE_*` and `FESOM3_WHICH_ALE`
  (`run_seg` uses `env "${e[@]}"`)

#### Task 4.3: GATE Stage 4 — restart exactness

- [ ] `F3=<worktree> run_restart_gate_core2.sh` with SE on (default N = 4, K = 2, plus the month-boundary split):
  linfs np 1 and np 2 → max|Δ| = 0 over the full state including the SE rings; the same for zstar
- [ ] `run_restart_gate_multirank.sh` (np 2 → np 8 round trip) with SE: max|Δ| = 0
- [ ] an SI checkpoint read with SE on dies with the message above (one-off; record it)
- [ ] G0 (SE off: the registered set is unchanged, so old checkpoints still read)

### Stage 5 — Physics and performance (G3)
*Costs node-hours: launch only after Nikolay's yes on the estimate below.*

Estimate: CORE2 year ≈ 42 min on one node (SI, 144 ms/step); 3000 steps ≈ 7 min. Total ≈ 3–5 node-hours, one
sbatch at a time or in a short chain; each wait is a background `sacct` watcher.

#### Task 5.1: 3000-step screens

**Files:**
- Create: `tools/run_se_screen_core2.sbatch` (prod_env + SE knobs; `--hint=nomultithread --mem=0`)

- [ ] SI baseline and SE screens, 3000 steps each, CORE2 `prod_env`:
  - (a) χ = 0.1, M = 50;
  - (b) M = M_min from the guard;
  - (c) `se_visc_ah = 0`;
  - (d) zstar headline
- [ ] per screen, record: max|η| over time, cold-start peak, max CFL_z, max|uv|, divsum, SE_CHECK, NaN. Decide
  whether the linear viscosity suffices (⚠️ if (a) misbehaves: try the γ1 term in the same gather operator, as a
  new task)

#### Task 5.2: 1-year SE vs SI twin (linfs)

**Files:**
- Modify: `tools/zarr_diff.py` (field-statistics mode) if it fits; otherwise create `tools/se_twin_diff.py`

- [ ] two 1-year runs from the same cold start (SI, SE χ = 0.1 / M from 5.1), monthly means on
- [ ] annual-mean SSH difference (rms, max; port_kokkos reference 0.2 mm / 2.9 mm), global-mean SSH and salt drift,
  sea-ice area; record the numbers here

#### Task 5.3: SSH phase cost, SI vs SE

- [ ] one node, np 128, CORE2 `prod_env`, 300 steps, a same-day pair, min of 2: the TMR_OCE_SSH and TMR_OCE_SE_BT
  phase times for SI and SE (M = 50 and M = M_min); record ms/step

### Stage 6 — Verify and document

#### Task 6.1: Acceptance check

- [ ] every Overview requirement met:
  - SE off is byte-identical to the frozen references;
  - linfs and zstar invariants at np 1/2/8;
  - copy spreads 0;
  - restart max|Δ| = 0 (linfs and zstar);
  - G3 numbers recorded
- [ ] full ctest Intel + GNU, `run_g0_se.sh`, conserve_pi, restart gates, `run_tri_regression.sh` (`NEW_BIN=` this
  build)

#### Task 6.2: Documentation

- [ ] `IMPLEMENTATION.md`: an SE section covering the scheme, the vertex-bottom depth law, the shared-element design,
  the knobs, and restart (M and χ fixed across a chain; the SI-zstar restart gaps)
- [ ] `TESTING.md`: the new tests and gates; fix the stale 153 → current count
- [ ] `README.md`: the run switches; `docs/HANDOFF.md`: status and next steps; `docs/LESSONS.md` if anything was learnt
- [ ] draft the Sergey/Dmitry question packet in `docs/` (not sent)
- [ ] move this plan to `docs/plans/completed/`; tag and memory update after Nikolay's yes

## Technical Details

### 1. Notation, state, halo extents
τ = dt, dtbt = τ/M. SM coefficients: β = 0.281105, ζ = 0.00976186 − 0.13451357χ, γ = 0.083445 − 0.513584χ,
δ = ½ + 2ζ + γ + 2χ.

| array | shape | extent written | carried across steps | in restart |
|---|---|---|---|---|
| `se_eta` | (nNodL, 2): η^{−1/M}, η^{−2/M} | nNodL | yes | yes |
| `se_ubt` | (2, nElemL, 0:2): Ūⁿ, Ū^{−1/M}, Ū^{−2/M} | owned + exchanged eDim | yes | yes (halo ELEM) |
| `se_sbar_old` | (2, nElemL): š^{n−1} | owned | yes | yes |
| `se_ubtn` | (2, nElemL): copy of Ūⁿ for V(Ūᵐ − Ūⁿ) | owned + eDim | no | no |
| `se_etam`, `se_etanew` | (nNodL): η^m, η^{m+1} | nNodL (after `exchange_nod`) | no | no |
| `se_umean`, `se_force` | (2, nElemL) | owned (F: after the shared sync) | no | no |
| `se_H` | (nElemL): Hⁿ | owned + eDim | no | no |
| `se_c` | (MAX_NV, nElemO) | owned | no | no |
| `se_kappa` | (MAX_NV, nElemO) | owned (geometry, built once) | no | no |

Scalars in `t_dyn`: `use_ssh_se`, `se_M`, `se_chi`, `se_visc_ah`, `se_check`, `se_ready`, plus the four SM
coefficients.

### 2. One step
```
predictor (unchanged)                      -> uv = u^{n-1/2}, UV_rhs = u** - u^{n-1/2}, helem^n, eta_n = eta^n
se_bt_forcing:
  H^n = sum_k helem^n (nElemL) ; s^n = sum_k helem^n uv ; G = sum_k helem^n UV_rhs
  F = G/tau + g H^n grad(eta^n) + f e_z x (a2 s^n + a1 s_old)       ! = Rbar - r
  exchange_elem_shared(F) ; s_old = s^n ; ubtn = U^n ; [zstar] c_{e,v} from zbar_3d_n^n
se_subcycle (m = 0..M-1):
  U*   = (1.5+beta) U^m - (0.5+2beta) U^{m-1} + beta U^{m-2}
  umean += U*/M
  eta^{m+1} = eta^m + dtbt * (I(U*) - [nonlinfs] W areasvol) / areasvol ; exchange_nod
  etaA = delta eta^{m+1} + (1-delta-gamma-zeta) eta^m + gamma eta^{m-1} + zeta eta^{m-2}   (nNodL)
  HA   = H^n + sum_v elem_w c (etaA - eta^n)
  U^{m+1} = U^m + dtbt * ( -f e_z x U* - g HA grad(etaA) + F + V(U^m - ubtn) ) ; exchange_elem
  shift rings by copy
se_trim_finish:
  uv = u** - (sum_k helem^n u** - umean) / H^n ; exchange_elem_shared(uv) ; exchange_elem(uv)
  (nNodL) hbar_old = hbar ; hbar = eta^M ; eta_n = eta^M
  ssh_rhs_old = I(umean) - [nonlinfs] W areasvol ; [SE_CHECK] se_check_core
tail (unchanged): GM, vert_vel_ale (zstar: hbar - hbar_old), tracers, update_thickness_ale ; [SE_CHECK] se_check_tail
```
I is the inflow scatter (positive = convergence), the sign of `ssh_rhs_old`.

### 3. Communication per step
2M small exchanges (η node, Ū element; the latter only with viscosity), one shared sync of F, and the existing sync
and halo exchange of uv. No global reductions outside SE_CHECK and the one-off CFL guard.

### 4. Knobs (env only)

| env | default | meaning |
|---|---|---|
| `FESOM3_SSH_SE` | 0 | 1 = split-explicit |
| `FESOM3_SE_M` | 50 | substeps per baroclinic step |
| `FESOM3_SE_M_FORCE` | 0 | 1 = run below the CFL-guard minimum |
| `FESOM3_SE_CHI` | 0.1 | SM dissipation parameter |
| `FESOM3_SE_VISC_AH` | 0.01 | A_h = value·l_e (m/s); 0 = no BT viscosity |
| `FESOM3_SE_CHECK` | 0 | 1 = per-step invariant suite |

## Post-Completion
*Informational, no checkboxes.*

- **Sergey / Dmitry packet:** H_e(η) under the vertex bottom; the linear viscosity on Ū/H; Coriolis via the stored
  sum (D4); R = h·UV_rhs with the vector-invariant momentum advection; SM vs the Zenodo FB variant. Send after the G3
  numbers exist.
- **Production trial:** an SE segment in Dmitry's CORE2 setup at 512 ranks (`FESOM3_STEP_DIAG=1`), compared with
  results17/21.
- **JAX port:** the `FESOM3_JAXDUMP` stages for the SE call points, the rebase onto `jax-oracle`, linfs first.
- **Later:** NEMO SE (per-mesh M), the zstar 1-year twin, wide halo (the notes' next step), the γ1 viscosity if the
  screens ask for it, the SI-zstar restart gaps, and storing M/χ in the checkpoint with a warning on change.
