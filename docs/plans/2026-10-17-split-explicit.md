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
  - in a second hook after the tail (zstar only): layer sum max|Σhnode − (H0 + η)| and max|w(1)|; ➕ and the depth
    law max|Σhelem^{n+1} − H_e(η^{n+1})| (Task 3.1).
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
    **allreduce MIN** of the limit before deciding; print M_min on rank 0; die if M < M_min unless FORCE
    (➕ replaced by the operator-based guard of Task 0.4: the res-based bound admitted unstable M);
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

#### ➕ Task 0.4: operator-based CFL guard ✅

*A guard defect found by the Task 5.1 diagnosis: the res-based guard gave M_min = 35 on CORE2 at Δt 1800 s, and
M = 35 blows up at step 5 at χ = 0.1. The defaults (χ 0.1, M 50, `se_visc_ah` 0.01) are unchanged (Stage 5).*

**Files:**
- Modify: `src/oce/oce_ssh_se.F90` (the CFL GUARD block, the `se_cfl_*` routines, the guard part of `se_init`)
- Modify: `test/test_ssh_se.F90` (C3)

- [x] the guard follows the substeps' own linear operator instead of a resolution estimate:
  - `se_cfl_apply_a`: A η = −I(H(−g∇η))/areasvol, the gradient of `se_tendency` (gradient_sca) and the inflow
    `se_inflow`; H = Hⁿ = Σ_k helem over the element's layers (the loop of `se_elem_depth`, computed locally so that
    the guard does not depend on the zstar weights; under zstar the current thickness at `se_init`, before a
    restart read). A is self-adjoint in the areasvol inner product and ≥ 0, so its eigenvalues are ω² ≥ 0;
  - `se_cfl_omega_max`: ω_max by **100 power iterations** on the owned nodes, start vector a Weyl sequence in the
    global node id (Knuth's multiplicative hash mod 2³², integer arithmetic), ‖·‖_W normalisation with one
    allreduce sum and one `exchange_nod` per iteration; λ_k = ‖A v_k‖_W lies between the Rayleigh quotient of v_k
    and λ_max (a lower bound, like any power iteration). The rank-0 line reports the change of the last iteration;
  - `se_cfl_stable(a, χ, r)`: all roots of the characteristic polynomial of the AB3–AM4 substeps for one gravity
    mode, z⁴(z−1)² + a²(w₀z² + w₁z + w₂)(δz³ + d₀z² + γz + ζ), in |z| < r, decided by the Schur–Cohn recursion on
    p(rz) (no root finder; the weights of `se_sm_coeffs`, formed as in `se_subcycle`); `se_cfl_rho` = the spectral
    radius by bisection on r; `se_cfl_acrit(χ)` = the first a with a root on |z| = 1 (scan from a = 0.5 in steps of
    0.005, then 60 bisections): **1.8530680 / 1.5788384 / 1.3290986 at χ = 0 / 0.1 / 0.25** (numpy root finder of
    the Task 5.1 note: 1.853 / 1.579 / 1.329);
  - `se_init`: **M_min = ⌈τ/dtbt_max⌉, dtbt_max = s·a_crit(χ)/ω_max, s = `SE_CFL_SAFETY` = 0.9**; `se_cfl_m_min`
    and `se_cfl_decision` (OK / FORCED / DIE, `FESOM3_SE_M_FORCE`) unchanged; `se_cfl_dtbt_limit` removed. One
    rank-0 line, e.g. CORE2: `oce_ssh_se:   CFL guard: omega_max = 3.1518E-02 1/s (100 power iterations, last
    change  0.0E+00), a_crit(chi) = 1.5788, s = 0.90 -> dtbt <= 45.084 s -> M_min = 40 (0.057 s)`; the stop names
    ω_max·dtbt and s·a_crit
  - **s = 0.9**, why: s has to cover what the scalar test leaves out. (i) The power-iteration value is a lower bound;
    on CORE2 and pi the iteration reaches a fixed point (last change 0), and even without a spectral gap 100
    iterations from a generic start on N nodes miss ω_max by about ln(N)/400 (< 5 % up to N = 10⁸). (ii) Coriolis
    (f ≤ 1.5e-4 s⁻¹ against ω_max = 3.2e-2 s⁻¹ on CORE2), the BT viscosity (ν·dtbt/l² ≈ 4e-3) and, under zstar, H(η)
    (|η|/H ≈ 1e-3 in the trenches where the fastest mode lives) move the limit by well under 1 %. No damping has to
    be bought: at 0.9 a_crit the fastest mode decays (spectral radius 0.885 / 0.678 / 0.571 per substep at χ = 0 /
    0.1 / 0.25), and the scheme turns unstable abruptly above a_crit (2.19 / 1.88 / 1.70 at 1.1 a_crit). The upper
    end: s ≥ 0.854 keeps the default M = 50 admissible at χ = 0.25 on CORE2 (the setting gated in Task 2.2). s = 0.9
    is the value proposed in the Task 5.1 diagnosis
- [x] tests (`test_ssh_se` C3, rotated pi, np 1/2/8, Intel and GNU; C3 now runs after C4, the operator reads
  helem):
  - a) a_crit to 1e-3 of 1.853 / 1.579 / 1.329; `se_cfl_stable` flips between 0.999 and 1.001 a_crit; the radius at
    1.1 a_crit equals numpy's 2.186123 / 1.879877 / 1.700745 to 1e-5 (measured: equal to 6 digits); < 1 at 0.9 a_crit;
  - b) A self-adjoint: |(u, Aw)_W − (Au, w)_W| ≤ 6.5e-16 of ‖u‖_W‖Aw‖_W (np 1/2/8), (u, Au)_W > 0;
  - c) **pi ω_max = 5.9801241e-3 s⁻¹, last change 0; np 2 and np 8 equal to np 1 bitwise (relative difference 0),
    M_min 8 on all** (reference file next to the P4 one, `FESOM3_TEST_SE_P4REF.c3`); ω_max² ≥ the Rayleigh quotients
    of two hashed vectors;
  - d) M_min = ⌈τω_max/(s·a_crit)⌉ = **8 on pi at χ = 0.1** (the old guard: also 8), the decision at M = 1,
    M_min − 1, M_min, 40, 50;
  - e) **empirical, the real `se_subcycle`** (F = 0, U = 0, η = the power-iteration vector, viscosity on, one call
    per baroclinic step): at M = M_min (pi: 7 / 8 / 9 at χ = 0 / 0.1 / 0.25, ω_max·dtbt = 0.83 / 0.85 / 0.90 a_crit)
    ‖η‖_W falls to 6.3e-4 of the start over 300 steps; at ω_max·dtbt = 0.97 a_crit to 6.5-6.9e-4 over 3000
    substeps; at 1.03 a_crit it grows by > 10¹² (1.45 / 1.28 / 1.21 per substep) and at 1.1 a_crit by > 10¹² at
    **2.181 / 1.876 / 1.698 per substep against `se_cfl_rho` 2.186 / 1.880 / 1.701** (within 0.3 %; checked to 10 %).
    So the real loop's limit is a_crit to ±3 % on pi as on CORE2 (Task 5.1 note: to the substep)
- [x] CORE2 `prod_env`, Δt 1800 s (job 28218958, Intel build of this tree): **ω_max = 3.1518e-2 s⁻¹** (the
  diagnosis: 3.152e-2), last change 0; **M_min = 40 at χ = 0.1 and 48 at χ = 0.25** (np 8, 5 SE steps each with
  SE_CHECK, M = 50: etacomp ≤ 5.1e-15 m, trim ≤ 1.1e-13 m²/s, spreads 0); **`FESOM3_SE_M=35` now stops in
  `se_init`** (`M = 35 is below the CFL-guard minimum M_min = 40 (dtbt = 51.429 s > 45.084 s: omega_max*dtbt = 1.621
  > s*a_crit(chi) = 1.421)`), where before it blew up at step 5. **Guard wall time: 0.565 s at np 1, 0.057 s at
  np 8** (CORE2, 126 858 nodes); pi 0.002-0.04 s
- [x] GATE (job 28219005, 8.2 min): incremental builds; **ctest Intel 175/175; GNU 169/175, exactly the 6 known
  failures** (`test_wimpl_tra_np1/np2`, `meshdiag_mixed/quad/pi_w`, `test_shared_elem_np1`; the serial rerun fails
  the same 6); the driver stops `fesom_conserve_se_{badswitch,badm,cfl}_np1` pass unchanged (the cfl one now reads
  `M = 1 is below the CFL-guard minimum M_min = 8 (dtbt = 1800.000 s > 237.613 s: omega_max*dtbt = 10.764 >
  s*a_crit(chi) = 1.421)`, which the existing regex matches). **G0 GREEN** (linfs, zstar: 99 checkpoint variables
  and 1 649 154 dump records max|Δ| = 0, 20 step lines equal; conserve 249 lines, 26 configs; 361 s): SE off is
  untouched. Intel `fesom_lifecycle_native_mr` sha256 `bbbd552160a2bd5e…`. Jobs 28218952 (a compile error in the
  test, 34 s), 28218958 (builds, SE ctests verbose, CORE2), 28219005; 0.13 node-h. Scripts and logs
  `/work/ab0995/a270088/fesom3_se_ref/jobs/t04_*`, run dirs `/scratch/a/a270088/se_t04/`
- Task 5.1 screen (b) "M = M_min" now means the M_min of this guard (40 at χ = 0.1, 48 at χ = 0.25 on CORE2 at
  Δt 1800 s)

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

#### Task 1.3: Trim, finish and the `step_oce` branch ✅

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`
- Modify: `src/step/mod_step_oce.F90`
- Modify: `test/test_ssh_se.F90`, `test/CMakeLists.txt`

- [x] `se_trim_finish`:
  - u** = uv + UV_rhs;
  - uvₖ = u**ₖ − (Σⱼhelemⁿⱼu**ⱼ − ⟨⟨Ū⟩⟩)/Hⁿ_e on owned elements;
  - `exchange_elem_shared(uv)` (under `sync_shared_elem`, TMR_OCE_SHSYNC), then `exchange_elem(uv)`, exactly as in
    `update_vel`;
  - on **nNodL**: hbar_old = hbar; hbar = η^M; eta_n = η^M (the ice and `compute_vel_rhs` read halo vertices);
  - ssh_rhs_old = I(⟨⟨Ū⟩⟩) − W·areasvol (non-linfs);
  - Ū^M stays in the ring

  Signature `se_trim_finish(dyn, mesh, partit, dt, step, water_flux)`. u** and Σhelemⁿu** in one pass over the wet
  layers ulevels..nlevels−1 (update_vel's range), the correction c = (Σhelemⁿu** − ⟨⟨Ū⟩⟩)/se_H with se_H summed in
  the same layer order. ssh_rhs_old = `se_inflow(⟨⟨Ū⟩⟩)`; for non-linfs minus water_flux·areasvol on the owned
  open-surface nodes and `exchange_nod(ssh_rhs_old)`, as compute_hbar_ale :253-259 (unreachable until Task 3.1:
  se_elem_depth still stops non-linfs). update_eta_n's alpha blend does not apply (η^M is the elevation at n+1).
  Not touched: dhe (read only by the SI stiffness update), d_eta and ssh_rhs (SI only)
- [x] `step_oce`: `if (dynamics%use_ssh_se) then` call the three SE routines under TMR_OCE_SSH, `else` the existing
  block **verbatim**. Add `dump_node_2d` for hbar/eta_n at the same substep ids. The SI block is re-indented by four
  blanks only: `git diff -w` shows nothing but the wrapping lines, and the de-indented block is byte-identical to the
  original 15 lines (checked by script). SE dumps hbar at DUMP_SUBSTEP_HBAR and eta_n at DUMP_SUBSTEP_ETA_N; there is
  no ssh_rhs / d_eta dump in SE
- [x] ➕ remove the temporary second stop in `se_step_guard` ("step_oce has no split-explicit step yet", Task 0.3).
  In its place the guard stops when `FESOM3_SSH_SE=1` and step_oce got no `partit` (the SE routines require it)
- [x] test: Σₖhelemⁿ·uv = ⟨⟨Ū⟩⟩ to rounding after the trim; the correction is depth-uniform in velocity.
  `test_ssh_se` C10 (np 1 and 2) on a hashed predictor state (uv, UV_rhs) with hashed ⟨⟨Ū⟩⟩ and η^M: **max|Σhelemⁿuv −
  ⟨⟨Ū⟩⟩| = 1.1e-13 m²/s of Σhelem|u**| = 763 (1.5e-16)**; uvₖ − u**ₖ + c = **2.8e-17 m/s** for every layer (GNU
  4.2e-17) of max|u**| = 0.21, max|c| = 0.083 (tolerance 1e-13); the layers below nlevels, the Ū ring and ⟨⟨Ū⟩⟩
  untouched; hbar_old = old hbar, hbar = eta_n = η^M on owned + halo and ssh_rhs_old = I(⟨⟨Ū⟩⟩) on owned nodes
  (bitwise); np 2: uv copy spread 0 and halo-valid. ➕ Two driver ctests: `fesom_conserve_se_linfs_np1` (pi, linfs,
  20 SE steps through step_oce, the driver's linfs invariants) and `fesom_conserve_se_zstar_np1` (pins the
  se_elem_depth stop until Task 3.1)
- [x] ctest green; G0. Job 28217186 (shared, 7.7 min): **Intel 174/174** (172 + the 2 driver ctests). GNU: the 6
  known failures plus 5 zstar `fesom_conserve` tests (redi, KPP, TKE, mixcut_REDI, wsplit) that ran to "CONSERVATION
  OK" and then got "mpiexec has exited ... exiting improperly", all in the first batch of 8 started, all at ~11 s.
  ⚠️ Transient, not the code: the same binaries rerun (job 28217290) give **GNU 168/174, exactly the 6 known
  failures**, in two full `ctest -j 8` passes, and the 5 pass at `-j 1`. **G0 GREEN** (linfs, zstar: 99
  checkpoint variables and 1 649 154 dump records max|Δ| = 0, 20 step lines equal; conserve: 249 log lines, 26
  configs), new `fesom_lifecycle_native_mr` sha256 `9abe27997a2f07be…`. Scripts
  `/work/ab0995/a270088/fesom3_se_ref/jobs/t13_gate.sbatch`, `t13_gnu_rerun.sbatch`

#### Task 1.4: Invariant suite `FESOM3_SE_CHECK` ✅

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`, `src/step/mod_step_oce.F90` (tail hook)
- Modify: `src/infra/mod_halo.F90` (➕ `allreduce_max_r1`), `src/types/mod_dyn.F90` (➕ the MSL reference)
- Modify: `test/test_ssh_se.F90`, `test/CMakeLists.txt`

- [x] `se_check_core`, called inside `se_trim_finish`:
  - η-compatibility, trim, mean-sea-level change in metres since the start of the run;
  - copy spreads of Ū and ⟨⟨Ū⟩⟩ (meaningful from Stage 2 on);
  - max|η|, max|Ū|;
  - NaN/Inf → die with the step number

  Split into `se_check_values` (collective; returns `vals(SE_NCHK)`, so the test reads them), `se_check_line` (the
  format) and `se_check_core` (values, the rank-0 line, the stop). se_trim_finish calls it after the commit, with a
  copy of ηⁿ taken before it, while helem is still at n:
  - etacomp = max over owned nodes |η^{n+1} − ηⁿ − τ·ssh_rhs_old/areasvol| [m] (ssh_rhs_old = I(⟨⟨Ū⟩⟩) − W·areasvol);
  - trim = max over owned elements and both components |Σₖhelemⁿuvₖ − ⟨⟨Ū⟩⟩| [m²/s];
  - msl = Σareasvol·η^{n+1}/Σareasvol − the same mean at the start of the run [m]. ➕ The reference lives in
    `t_dyn` (`se_msl0`, `se_msl0_set`, two scalars beyond §1) and is set from ηⁿ by the first check of the process
    (a restarted run measures from its own start);
  - spreadU / spreadUm = `shared_copy_max_elem` of Ū^M (ring level 0) and ⟨⟨Ū⟩⟩ (0 at np 1);
  - maxeta = max|η^{n+1}| [m], maxU = max|Ū^M| [m²/s] (owned);
  - NaN/Inf: a count over eta_n and ssh_rhs_old (owned nodes), ⟨⟨Ū⟩⟩ and Ū^M (owned elements) and the wet layers of
    uv; > 0 prints the line and stops ("SE_CHECK: k non-finite values ... at step n"). ⚠️ The Intel build traps
    invalid operations (`-fpe0`), so there a NaN-producing operation stops the run before SE_CHECK sees it; the
    count is the safety net of builds without the trap (GNU) and of NaN that arrive without an invalid operation

  **The line** (rank 0, one per step, blank-separated key=value tokens, values `es11.4` without blanks):
  `SE_CHECK step=20 etacomp=1.6792E-15 trim=1.1369E-13 msl=-1.7969E-17 spreadU=0.0000E+00 spreadUm=0.0000E+00
  maxeta=1.8008E+00 maxU=4.2636E+02` (one line; pi, step 20 of `fesom_conserve_se_linfs_np1`). Parse with
  `grep '^SE_CHECK '` and split each token at `=` (run_conserve_pi.sh does, Task 1.5)
- [x] `se_check_tail`, called in `step_oce` after `update_thickness_ale` when SE and SE_CHECK are on: the zstar layer
  sum and max|w(1)| (filled in by Task 3.1). Wired after `timer_stop(TMR_OCE_TRACER)`; a no-op under linfs (hnode is
  frozen, w(1) is the free-surface velocity); for non-linfs it stops ("not implemented yet", Task 3.1), unreachable
  today because se_elem_depth stops first
- [x] reductions via `allreduce_max`/`allreduce_sum`; nothing is computed or printed when off. One `allreduce_max`
  of the six maxima (➕ a rank-1 `allreduce_max_r1` in the generic interface, additions only) and one
  `allreduce_sum` of four sums per step, plus the two spread probes. Off: se_trim_finish does not take the ηⁿ copy
  and never calls the check (C10: `se_msl0_set` stays false)
- [x] test: on a valid state the checks report rounding-level values; with a deliberately perturbed uv on one element
  the trim check reports the perturbation. `test_ssh_se` C11 (np 1 and 2), a full SE step (M = 40 substeps + the
  trim, which prints the line) on the hashed state: **etacomp 3.4e-16 m** of max(|ηⁿ|, |Δη|) = 0.41 m (GNU 3.1e-16),
  **trim 7.1e-14 m²/s** of Σhelem|uv| = 766 (GNU 6.4e-14), **msl 8.7e-19 m** (np 2 −2.8e-18; GNU 1.7e-18), copy
  spreads **0** at np 1 and np 2, maxeta and maxU equal to the direct maxima, no non-finite value (tolerances 1e-12 of
  the scales).
  The line has the key=value form with single blanks; uv + δ (δ = 1e-3 m/s) in the top layer of one element gives
  **trim = 5.000e-3 m²/s = helemδ** (relative 1e-6) with etacomp and the count unchanged; a NaN there gives count 1
  (the Intel `-fpe0` trap is switched off around that one call with `ieee_set_halting_mode`).
  `fesom_conserve_se_linfs_np1` now runs with `FESOM3_SE_CHECK=1` and needs the step-20 SE_CHECK line
- [x] ctest green. Job 28217323 (shared, 2.3 min): **Intel 174/174; GNU 168/174, the 6 known failures**, no new
  one; `test_ssh_se_np1/np2` and `fesom_conserve_se_*` pass on both. The pi SE run (`fesom_conserve_se_linfs_np1`,
  20 steps): step 1 etacomp 1.7e-16 m, trim 2.8e-14 m²/s; step 20 etacomp 1.7e-15 m, trim 1.1e-13 m²/s, msl
  −1.8e-17 m, maxeta 1.80 m, maxU 426 m²/s (GNU: the same to 1-2 digits). Script `t13_gate.sbatch` (TAG=t14)

#### Task 1.5: GATE Stage 1 — linfs np 1 ✅

**Files:**
- Modify: `tools/run_conserve_pi.sh` (the SE blocks), `src/drivers/fesom_lifecycle_native_mr.F90` (➕ max CFL_z line)

- [x] pi, `fesom_conserve`, `FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1 FESOM3_WHICH_ALE=linfs`, 1000 steps: SE_CHECK
  invariants at rounding (record the levels); `fesom_conserve`'s linfs checks green (T10 no-leakage, helem
  consistency, finiteness). Heat and salt drift reported only, as for SI linfs. Job 28217370 (np 1, M = 50, launched
  as the SE blocks of `run_conserve_pi.sh`), maxima over the run, 1000 SE_CHECK lines each:
  - default physics: **etacomp 5.0e-15 m, trim 5.7e-13 m²/s, |msl| 3.5e-16 m, spreadU = spreadUm = 0**; max|η|
    4.03 m, max|Ū| 913 m²/s;
  - vector-invariant momentum, rvo_upwind 1, sea ice: **4.8e-15 m, 4.5e-13 m²/s, 2.9e-16 m, spreads 0**; 3.98 m;
  - the linfs invariants hold for all 1000 steps (no T10 / helem / non-finite stop), divsum ≤ 2.3e-16 (SI 1.5e-15),
    copy spreads 0. Drift (reported): heat −3.676e-4, salt −1.591e-6 (SI −3.692e-4, −1.604e-6); with ice −3.708e-4,
    −1.566e-6 (SI −3.722e-4, −1.584e-6), ice masses ≤ 1.5e-15. Wall 90 s (SI 76 s), with ice 125 s (SI 100 s)
- [x] CORE2 `prod_env`, np 1, 48 steps (~6–10 min) with SE and with SI: no NaN, invariants at rounding, max|η| and
  max CFL_z comparable. Job 28217370 (`t15_gate.sbatch`; SE 422 s, SI 354 s):
  - **M_min = 35** (limit 52.199 s; the default M = 50 passes), as port_kokkos;
  - SE_CHECK over 48 steps: **etacomp ≤ 1.9e-14 m, trim ≤ 3.4e-13 m²/s, |msl| ≤ 4.4e-16 m, spreads 0**, no NaN;
    divsum ≤ 1.1e-15 (SI 1.3e-15);
  - max over the run: **max|η| SE 4.29 m (step 31) vs SI 1.66 m (step 13); max CFL_z SE 0.870 (step 24) vs SI
    0.612 (step 19)**; max|uv| 1.90 vs 1.79 m/s. SE − SI at step 48 (node dumps): rms Δη 0.16 m of rms η_SI
    0.51 m; rms Δw 3.8e-5 of 9.3e-5 m/s; rms ΔT 9.2e-3 °C, rms ΔS 3.6e-3
  - ⚠️ **max|η| is 2.6 times SI's: investigated, physics, no code change.** The extremes are smooth, coherent
    set-ups (not grid-scale noise) at the 20 m minimum depth of the resonant macrotidal gulfs, same sign as SI and
    ~4.6 times larger at step 48: head of the Bay of Bengal (91.9°E 22.4°N) +4.25 m vs +0.92, Gulf of Khambhat
    −4.08 vs −0.79, Gulf of California −3.91 vs −0.73. Decisive Δt experiment (job 28217549, one model day,
    end-of-day η): **SE is Δt-independent** (Bay of Bengal head 4.25 / 4.31 m at Δt 1800 / 900 s, Khambhat −4.08 /
    −4.10, California −3.91 / −3.95; max|η| over the day 4.29 / 4.41 m) **while SI converges to it as Δt shrinks**
    (Bay of Bengal head 0.92 → 1.74 → 3.20 m at Δt 1800 / 900 / 300 s, Khambhat −0.79 → −1.81 → −3.05, California
    −0.73 → −1.15 → −1.98; max|η| over the day 1.66 / 2.21 / 3.21 m; rms η 0.507 / 0.513 / 0.527 m vs SE 0.548).
    SI's θ = 1 backward Euler damps the resonant external gravity waves of the cold-start adjustment (damping per
    step grows with ωΔt); SE resolves them. port_kokkos M12 recorded the same ("hotter than SI in transients by
    physics", cold-start peak 3 m vs 2 m). The larger CFL_z follows from the resolved waves. SE at Δt 900 s:
    M_min 18, SE_CHECK etacomp 1.2e-14 m, trim 2.8e-13 m²/s, |msl| 6.0e-16 m. Stage 5 should watch these gulfs
    in the cold-start peak
  - ➕ max CFL_z: the STEP_DIAG step line has no CFL_z, so the lifecycle now prints a separate line
    `fesom_lifecycle_native_mr: CFL_z step N  max CFL_z=…` (dstat(10), the same MPI_MAX). The step line stays
    byte-identical (G0 diffs it; run_cutout_short.sh reads its fields)
  - timers (np 1, mean ms/step, informational; job 28217684, SE without SE_CHECK, the two runs alone on the
    allocation): **SE "dynamics ssh" 947.6, of which the barotropic substeps 853.3 (M = 50, 17 ms per
    substep); SI 496.9, of which the CG solve 268.7**; step_oce 6558 vs 6089, loop 7509 vs 7036 (SE 368 s, SI
    344 s). At np 1 the SE SSH phase costs 1.9 times SI's; Stage 5 measures the parallel case
- [x] record the measured levels here and set the gate thresholds (margin ×100):

  | invariant | pi, 1000 steps (max of 2 configs) | pi gate (`run_conserve_pi.sh` SE_TOL_*) | CORE2 np 1, 48 steps | CORE2 gate (Stage 2) |
  |---|---|---|---|---|
  | etacomp [m] | 5.0e-15 | 5e-13 | 1.9e-14 | 2e-12 |
  | trim [m²/s] | 5.7e-13 | 6e-11 | 3.4e-13 | 3.5e-11 |
  | \|msl\| [m] | 3.5e-16 | 4e-14 | 6.0e-16 (Δt 900 s) | 6e-14 |
  | spreadU, spreadUm | 0 | 0 exactly | 0 | 0 exactly |

  `tools/run_conserve_pi.sh` has the SE blocks (header and two configs, np 1: default, vinv + rvo_upwind 1 + ice;
  `while read` list, so Task 2.2 adds np 2/8 as lines): the driver's linfs invariants plus the run maxima of the
  SE_CHECK tokens against SE_TOL_ETACOMP / _TRIM / _MSL, spreads exactly 0, and one line per step. G0 runs the frozen
  copy, so its case set is unchanged
- [x] ➕ final gate on the Task 1.5 tree (job 28217684, shared, 24.8 min, 0.31 node-h): **ctest Intel 174/174, GNU
  168/174 (the 6 known)**; `run_conserve_pi.sh` **GATE OK**, 28 configs, the SE blocks at 20 steps: etacomp
  2.0e-15 / 1.6e-15 m, trim 1.1e-13 / 1.7e-13 m²/s, |msl| 1.0e-16 m, spreads 0; **G0 GREEN** (linfs, zstar, conserve
  byte-identical to the frozen main references; 416 s); **restart gate linfs np 1 GREEN** (mid-run and month
  boundary, checkpoint and live dump max|Δ| = 0; 198 s). Intel `fesom_lifecycle_native_mr` sha256
  `b4661db7b23d4c5f…`. Scripts `/work/ab0995/a270088/fesom3_se_ref/jobs/t15_gate.sbatch`, `t15_where.sbatch`,
  `t15_dtexp.sbatch`, `t15_final.sbatch`; run dirs `/scratch/a/a270088/se_t15/`. Stage 1 cost (Tasks 1.3-1.5):
  0.78 node-h

### Stage 2 — linfs, many ranks
*The risk port_kokkos never had: shared border elements under the vertex bottom.*

#### Task 2.1: One-owner forcing sync and copy-spread checks ✅

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`
- Modify: `test/test_ssh_se.F90`, `test/CMakeLists.txt`

- [x] `exchange_elem_shared(se_force)` after `se_bt_forcing` (under `sync_shared_elem`). In place since Task 1.1: it
  is the last statement of `se_bt_forcing` (the `is_multirank` / `partit%sync_shared_elem` block after the F loop,
  timer TMR_OCE_SHSYNC); P1 below tests it
- [x] tests (`test_ssh_se` np 2/8 on rotated pi): a separate block P1-P4 (`p_many_ranks`, between C10 and C11),
  ➕ `test_ssh_se_np8` registered. The state is a real predictor: hashed uv (0.02 m/s), η (0.1 m) and pgf, then
  every step the real `compute_vel_rhs` and `viscosity_filter` (opt_visc 7, the drivers' coefficients), whose edge
  scatter adds into a shared element in each rank's own edge order. P1 and P2 share 6 SE steps (M = 40) with the
  sync on. Levels (Intel; GNU in brackets where it differs), job 28218252:
  - **P1, F before the sync > 0:** every step a copy runs `se_bt_forcing` with the sync off. **np 8: spread
    8.7e-19 m²/s² in 5 of 6 steps** (GNU 1.7e-18), max|F| 1.4e-4, UV_rhs spread 3.5e-18 m/s; 10 of the 6 × 226
    receive slots differ before the sync. The real call leaves the spread **exactly 0**, and every slot that is not
    a receive slot of `com_elem2D_shared` keeps its unsynced value bitwise. ⚠️ At **np 2 the Intel build has no F
    spread in any of the 6 steps** (35 receive slots; the UV_rhs spread of 1.7e-18 m/s is absorbed in the layer sum
    G; GNU 8.7e-19 in 2 steps), so the "> 0" checks of P1 and P3 are required at np ≥ 8 only (printed at np 2)
  - **P2, after every step:** copy spreads of the three Ū ring levels, ⟨⟨Ū⟩⟩, uv and uv_rhsAB **exactly 0** (np 2
    and 8); **inflow-sum ratio** |Σ I(X)| / Σ|I(X)| over the owned nodes, reduced (the STEP_DIAG / fesom_conserve
    divsum), **≤ 1.2e-17 for X = ⟨⟨Ū⟩⟩ and ≤ 1.4e-17 for X = Ū^M** (tolerance 1e-14)
  - **P3, the same 6 steps with the sync off (F and uv):** np 8 spreads Ū^M 5.6e-15, ⟨⟨Ū⟩⟩ 4.4e-15 m²/s (GNU
    4.4e-15 both), so P2's zeros are the sync's doing; np 2 Intel 3.6e-15 / 1.3e-15 (GNU 0 / 0: absorbed)
  - **P4, np 1 vs np 2 vs np 8** (η and Ū^M after the 6 steps, gathered by global id; np 1 writes
    `FESOM3_TEST_SE_P4REF` = `<build>/test/test_ssh_se_p4_np1.bin`, ctest fixture `se_p4_ref`, np 2 and 8
    compare): **np 2 bitwise identical to np 1** (Intel and GNU); **np 8: max|Δη| 6.9e-17 m = 3.6e-16 of max|η|
    0.19 m, max|ΔŪ| 2.3e-14 m²/s = 7.7e-16 of max|Ū| 30 m²/s** (GNU 3.8e-16, 8.6e-16); tolerance 1e-12
  - ⚠️ **Found and fixed: the copies of ⟨⟨Ū⟩⟩ differed by 1 ULP although every input was bitwise equal.** The first
    run (job 28218068) failed P2 on Intel at np 2 and 8: ⟨⟨Ū⟩⟩ spread 1.8e-15 m²/s, the Ū ring spreads 0 (GNU
    passed). `se_subcycle` accumulated `umean += ustar/rM`. The Intel anchor flags include `-no-prec-div`, and the
    element loop is vectorised (vector length 2) with a scalar remainder loop (vectorisation report, job
    28218254). A division by a loop invariant then rounds differently in the vector body and in the remainder, so
    the result depended on the element's local index, which differs between the owners of a shared element. The
    fix is `umean += ustar·(1/M)` with the reciprocal computed once (`rMinv`); a product is correctly rounded on
    every code path, and the report shows the loop no longer vectorised. After the fix all P2 spreads are 0 (job
    28218127). The defect was also visible at scale with the old binary: CORE2 np 128, spreadUm > 0 in 298 of
    340 steps, up to 1.1e-13 m²/s (job 28218070); pi np 2 and 8 over 1000 steps, spreadUm 2.8e-14 (job
    28218069), which the new SE np 2/8 lines of `run_conserve_pi.sh` would have failed. So D2's "gather in fixed
    slot order" is necessary but not sufficient: **the arithmetic of a per-element loop must not depend on the
    element's local index.** No other division by a loop invariant feeds unsynced duplicated state: F's `G/τ` is
    synced afterwards, the trim divides by the per-element Hⁿ (and uv is synced), the checks are diagnostics
- [x] ctest green. Job 28218252 (shared, ctest 30 s Intel, 45 s GNU): **Intel 175/175** (174 + `test_ssh_se_np8`);
  **GNU 169/175, exactly the 6 known failures**; `test_ssh_se_np1/np2/np8` pass on both. Scripts
  `/work/ab0995/a270088/fesom3_se_ref/jobs/t21_ctest.sbatch` (TAG t21, t21b), `t21_optrpt.sbatch`,
  `t22_final.sbatch`

#### Task 2.2: GATE Stage 2 — linfs np > 1 ✅

- [x] add an SE block to `tools/run_conserve_pi.sh` (linfs + ice, np 2 and 8, `FESOM3_SSH_SE=1`) next to the
  SHARED block; the existing blocks stay unchanged. (Task 1.5 put the SE block in place with np 1 configs: add
  `<label> <np> <env>` lines to its `while read` list; re-measure the SE_TOL_* levels at np 2/8). Two lines,
  `momadv-vinv-upw1+ice 2` and `… 8` (vector-invariant momentum, rvo_upwind 1, sea ice), with
  `FESOM3_DIVSUM_TOL=$TOL`, so the driver's shared-element gate (divsum ≤ 1e-12, dUV = dAB = dsig = 0) runs as well.
  Re-measured over **1000 steps** (job 28218250, fixed binary, default χ = 0.1), np 2 / np 8: **etacomp 4.7e-15 /
  6.3e-15 m, trim 4.5e-13 / 3.4e-13 m²/s, |msl| 7.4e-16 / 3.0e-16 m, spreads 0, divsum 2.9e-16 / 1.7e-16**,
  dUV = dAB = dsig = 0; at most 2.1 times the np 1 levels, so **SE_TOL_* stay unchanged** (margins ≥ 54). The full
  script at 20 steps (job 28218252): **GATE OK, 30 configs**; np 2 etacomp 1.6e-15, trim 1.7e-13, |msl| 8.3e-17,
  divsum 1.2e-16; np 8 1.7e-15, 1.7e-13, 5.7e-17, 1.5e-16. With the binary before Task 2.1's fix both lines fail
  (spreadUm 2.8e-14 over 1000 steps, job 28218069)
- [x] CORE2 `prod_env`, np 8 (480 steps) and np 128 on one node (480 steps), `FESOM3_STEP_DIAG=1 FESOM3_SE_CHECK=1`:
  - divsum ≤ 1e-13;
  - dUV = dAB = dsig = 0;
  - SE copy spreads = 0;
  - invariants at the Stage 1 level (the CORE2 gate column of the Task 1.5 table)

  ⚠️ **At the default χ = 0.1 CORE2 blows up at step 337-340** (7 days; floating overflow), at np 8 (job
  28218069; job 28218164 with the fixed binary) and at np 128 (job 28218070) in the same step, so it is not
  partition-dependent and not a shared-element defect (spreads 0, dUV = dAB = dsig = 0, divsum ≤ 1.5e-15 up to step
  336). Where (node dumps of steps 325-340, jobs 28218164 / 28218239): a cluster of 20 m deep (minimum depth)
  Antarctic shelf nodes at 72-75°W, 71-73°S (Bellingshausen Sea coast, sea ice a = 1) carries a grid-scale η
  pattern, ±5 m between neighbouring nodes, with w ≈ 3-5e-3 m/s (CFL_z 1.0-1.8 in the top layer) from step 325 on;
  SI at step 336 has a smooth −1.6 m there and CFL_z ≤ 0.032. At step 337 η reaches 21 m and uv 55 m/s. A
  grid-scale gravity wave in 20 m of water (~30 km node spacing) has ωτ ≈ 2.6 but ω·dtbt ≈ 0.05: SI's
  backward Euler damps it by 1/(1 + (ωτ)²) per step, while the SM substeps hardly damp it at χ = 0.1. Screens at
  np 8, 480 steps (report only; jobs 28218164, 28218250): **χ = 0.25 runs 480 steps**; at χ = 0.1, **M = 100
  blows up earlier (step 192)**,
  **`se_visc_ah = 0` earlier (step 298)**, **`se_visc_ah = 0.1` runs 480 steps** (max|η| 4.72 m, max CFL_z 1.61), and
  **M = 35 = M_min blows up at step 5**: the CFL guard's 0.5·res/√(gH) is not a sufficient stability limit (M = 50
  is 0.69 of it). **The gate therefore runs with `FESOM3_SE_CHI=0.25`** (inside D1's knob range); the default χ =
  0.1 is unchanged here, and the choice of the default (χ, `se_visc_ah`, M) is Stage 5's (➕ note at Task 5.1).
  Diagnosed 2026-10-18 (➕ note at Task 5.1): both blow-ups are the scheme, not a code defect. M = 35 exceeds the
  AB3–AM4 limit (the guard underestimates ω_max by 3.29×; true M_min 36 at χ = 0.1, 43 at χ = 0.25); the step-337
  growth is a 3-step-period grid-scale shelf mode fed through the sea ice (ice tilt + ice→ocean stress, frozen per step).
  Results with χ = 0.25 and the fixed binary (`bin_t22_5a43ca3` = this tree's Intel build, sha256
  `5a43ca392d64eebe…`):
  - **np 8** (job 28218250, shared, 725 s): **divsum ≤ 1.19e-15; dUV = dAB = dsig = 0; spreadU = spreadUm = 0**;
    SE_CHECK max **etacomp 6.6e-14 m, trim 3.2e-12 m²/s, |msl| 1.6e-15 m** (gate 2e-12, 3.5e-11, 6e-14); max|η|
    4.70 m (step 67; 2.06 m at step 480), max|uv| 2.97 m/s, max CFL_z 1.60 (step 300);
  - **np 128** (job 28218251, one compute node, 85 s): **divsum ≤ 3.1e-16; dUV = dAB = dsig = 0; spreads 0**;
    **etacomp 6.3e-14 m, trim 2.3e-12 m²/s, |msl| 1.3e-16 m**; max|η| 4.69 m (step 67), max|uv| 2.98 m/s, max CFL_z
    1.60 (step 301). SI on the same node: max|η| 2.04 m, max|uv| 2.97 m/s, max CFL_z 1.59 (step 300), divsum ≤
    1.1e-15
  - the invariant levels over 480 steps are 3-10 times the 48-step np 1 levels of Task 1.5 and stay inside the
    CORE2 gate column
  - timers at np 128 (job 28218251, the three runs one after the other on one node; mean ms/step over 480 steps;
    informational): **SE (no SE_CHECK): TMR_OCE_SSH 12.93, of which TMR_OCE_SE_BT 10.52 and the shared sync
    0.67; LOOP 146.2. SI: TMR_OCE_SSH 9.13, of which the CG solve 5.43 and the shared sync 0.29; LOOP 144.2.** SE
    with SE_CHECK: 14.56 / 10.81, LOOP 150.3
- [x] determinism: the same binary at the same np, run twice, is bitwise identical (`dump_diff.py --rel-floor 0`);
  np 1 vs np 8 difference recorded. CORE2 np 8, 480 steps, χ = 0.25, runs A and B (job 28218250): the 1440 step /
  CFL_z / SE_CHECK lines identical, **`dump_diff.py --rel-floor=0` on the step-480 node dumps: 1 395 438 records
  max|Δ| = 0; `zarr_diff.py --rel-floor 0` on the step-480 checkpoint: 99 variables max|Δ| = 0**. (At χ = 0.1 the
  two np 8 runs were also line-identical up to their common blow-up, job 28218069.) np 1 vs np 8 after 48 steps
  (node dumps, χ = 0.25): **max|Δη| 4.6e-2 m (rms 9.7e-4 of rms η 0.55 m), max|Δw| 4.6e-4 m/s, max|ΔT| 5.1 °C (rms
  0.027), max|ΔS| 1.9 (rms 0.011)**; SI the same: max|Δη| 4.9e-2 m (rms 8.7e-4), max|ΔT| 5.2 °C (rms 0.027), max|ΔS|
  1.9. So SE adds no partition dependence beyond the model's own growth of roundoff over 48 steps (pi P4 after 6
  SE steps: 3.6e-16 relative)
- [x] G0. Job 28218252 (with the Task 2.1 tree): **G0 GREEN** (linfs, zstar: 99 checkpoint variables and 1 649 154
  dump records max|Δ| = 0, 20 step lines equal; conserve: 249 log lines, 26 configs), 449 s; Intel
  `fesom_lifecycle_native_mr` sha256 `5a43ca392d64eebe…`. Scripts `/work/ab0995/a270088/fesom3_se_ref/jobs/
  t22_np8.sbatch`, `t22_np128.sbatch`, `t22_diag.sbatch`, `t22_where.sbatch`, `t22_gate2.sbatch`,
  `t22_np128b.sbatch`, `t22_final.sbatch`; run dirs `/scratch/a/a270088/se_t22/`. Stage 2 cost: 1.0 node-h (the
  two np 128 jobs 0.13)

### Stage 3 — zstar
*Make "switch zstar on" a tested fact.*

#### Task 3.1: zstar depth law and freshwater ✅

**Files:**
- Modify: `src/oce/oce_ssh_se.F90`
- Modify: `test/test_ssh_se.F90`
- ➕ Modify: `src/types/mod_dyn.F90` (`se_fwmsl`), `src/step/mod_step_oce.F90` (W to `se_subcycle`),
  `test/CMakeLists.txt` (the zstar SE conserve tests)

- [x] `se_elem_depth` under zstar, from **zbar_3d_n differences** exactly as `vert_vel_ale` (:438-444) does:
  - Htilde_v = zbar_3d_n(1,v) − zbar_3d_n(nlev_v−1,v);
  - c_{e,v} = (zbar_3d_n(ule(e),v) − zbar_3d_n(K_{e,v}+1,v))/Htilde_v, with K_{e,v} = min(nle(e)−1, nlev_v−2);
  - recomputed once per step;
  - the substep uses H^A = Hⁿ + Σ_v elem_w·c·(η^A − ηⁿ)

  Checked against the code, the formula is right as written: `vert_vel_ale` stretches layers 1..nlev_v−2 of a
  vertex by (zbar_3d_n(nz) − zbar_3d_n(nz+1))·Δη/Htilde_v (the deepest layer is the fixed anchor),
  `update_thickness_ale` sets helem = Σ_v elem_w·hnode over the element's full range ule..nle−1, and the zbar
  differences telescope to the c above. Since nlevels(e) = min_v nlevels_nod2D(v), c == 1 bitwise (numerator and
  Htilde are the same expression) at every vertex with nlev_v ≤ nle + 1, not only the shallowest one; 0 < c < 1
  at deeper vertices. c = 0 on a triangle's slot 4 (elem_w = 0) and at a vertex without an open surface. Owned
  elements, a gather in slot order from halo-valid zbar_3d_n (copies of a shared element agree bitwise). The
  `se_elem_depth` stop now fires only for an ALE mode other than linfs and zstar
- [x] W in the η equation when `which_ALE /= 'linfs'` (mask `ulevels_nod2D == 1`, as `compute_hbar_ale` :253-259).
  `se_subcycle(…, water_flux)` (optional; step_oce passes it) subtracts W·areasvol from I(U*) in every substep;
  `se_trim_finish` already put the same term into ssh_rhs_old, so the η-compatibility check covers it. linfs
  ignores W, bit for bit the old loop
- [x] `se_check_tail`: layer sum max|Σhnode − (H0 + η)|, max|w(1)|; `se_check_core`: mean-sea-level change against
  the integrated freshwater. Done:
  - `se_check_tail_values` (exposed for the tests) and a second line `SE_CHECK_TAIL step=N laysum=… w1=…
    hlaw=…` (`'^SE_CHECK '` does not match it); NaN → stop. laysum: H0_v = zbar(1) − zbar(nlev_v), owned nodes.
    **w(1) = 0 to rounding, also with freshwater**: `vert_vel_ale` sets w(1) = I_h/area − (hbar − hbar_old)/τ − W,
    the SE elevation change is τ(I(⟨⟨Ū⟩⟩) − W·areasvol)/areasvol, area == areasvol and the trim gives
    Σhelemⁿuv = ⟨⟨Ū⟩⟩. ➕ hlaw = max|Σₖhelem^{n+1} − H_e(η^{n+1})| over owned elements, with H_e(η^{n+1}) = se_H +
    Σ elem_w·c·(hbar − hbar_old): the depth law of the substeps against what the tail built, in every run;
  - msl= is now net of the freshwater: msl = (mean η^{n+1} − mean η_start) − fw, with fw = −Σ_steps τ·ΣW·areasvol/
    Σareasvol (`t_dyn%se_fwmsl`, accumulated by the one check per step that gets water_flux; reset with se_msl0).
    Non-linfs lines end with a token `fw=` (linfs lines unchanged; fw ≡ 0 there)
- [x] tests (`test_ssh_se`, block Z1–Z6 after C11, run last; np 1 and 2, Intel and GNU identical to the digits
  shown unless noted):
  - Z1: nlevels(e) == min over the vertices of nlevels_nod2D; c == 1 exactly at vertices with nlev_v ≤ nle + 1,
    0 < c < 1 at deeper ones, 0 beyond elem2D_nnodes; cold and stretched states. pi: **3532 elements (np 1) have
    a c < 1, min c = 0.0189**;
  - Z2: H_e(ηⁿ) == se_H == Σhelemⁿ bitwise;
  - **Z3 (decisive):** a hashed Δη (|Δη| ≤ 0.5 m, max|η| 0.90 m) through the real `vert_vel_ale` +
    `update_thickness_ale`, twice (from the cold state, then from the stretched one with c recomputed):
    **max|Σhelem^{n+1} − H_e(η^{n+1})| = 3.6e-12 / 4.5e-12 m of max H = 5650 m** (GNU 3.6e-12 / 5.5e-12; tol
    1e-14·max H); the c = 1 law misses by 0.26 / 0.24 m and the linfs law by 0.49 m; layer sum 2.7e-12 / 3.6e-12 m;
  - Z4: uniform W = 1e-6 m/s, Ū = 0, M = 40: **max|η^M − η⁰ + τW| = 1.7e-18 m of τW = 1.8e-3 m**, ΔV = −τΣW·areasvol
    (−6.115e11 m³) to 1e-12; through `se_trim_finish`: ssh_rhs_old = I(⟨⟨Ū⟩⟩) − W·areasvol, fw = −1.800e-3 m, msl net
    of it 5.9e-18 m, etacomp 2.2e-18 m, the line carries fw=; under linfs the same W leaves η exactly 0;
  - Z5: one full zstar SE step with a hashed W (1e-6 m/s) from a hashed state whose thicknesses carry its η:
    `se_subcycle(W)` → `se_trim_finish(W)` → `vert_vel_ale(W)` → `update_thickness_ale`. **etacomp 2.3e-16 m (GNU
    3.5e-16) of 0.41 m, trim 8.5e-14 m²/s, msl net of fw (−9.5e-6 m) −2.4e-18 m, laysum 2.7e-12 m, hlaw 4.5e-12 m,
    w(1) 7.1e-19 m/s (GNU 7.8e-19) of max|Δη/τ| + |W| = 2.3e-4 m/s**; np 2: copy spreads of c, helem^{n+1}, U^M and
    ⟨⟨Ū⟩⟩ 0, and those of ⟨⟨Ū⟩⟩ and U^M 0 after the substeps for every M = 1..40;
  - Z6: under linfs c ≡ 0 (bitwise)

  ⚠️ **Found and fixed by Z5 (np 2): ⟨⟨Ū⟩⟩ copies of shared elements differed in the last bit.** A per-substep
  probe (temporary instrumentation, not committed) showed U*, all three U ring levels and H^A bitwise equal at every
  substep while ⟨⟨Ū⟩⟩ differed (1e-16 … 1.8e-15 m²/s), **also under linfs** (the C9/C11 calls), sometimes rounding
  back to equal by m = M. Cause: `se_umean += U*/rM`: with Intel's `-no-prec-div` the division by the loop-invariant
  M is evaluated differently in the vectorised body and in the scalar remainder of the element loop, and a shared
  element sits at different loop positions on its owners. Fix: multiply by rMinv = 1/rM (one line; Stage 1 linfs
  values change at rounding level only, e.g. `fesom_conserve_se_linfs_np1` step-20 etacomp 1.68e-15 → 1.47e-15 m,
  Intel). After it, no copy differs at any
  substep of any test call at np 2 and np 8. Lesson: a division by a loop-invariant scalar in a per-element loop
  is not copy-safe under `-no-prec-div`; Stage 2 must not reintroduce it
- [x] ctest green. Job 28218185 (shared, 32 cores): **Intel 175/175; GNU 169/175, exactly the 6 known failures**;
  the new tests pass on both. The pinned stop `fesom_conserve_se_zstar_np1` is now a real test,
  `fesom_conserve_se_zstar_np{1,2}` (pi, zstar, 20 SE steps, SE_CHECK, `FESOM3_CONSERVE_TOL=1e-12`): heat 1.7e-15
  / 3.2e-15, salt −2.6e-14 / −1.5e-14, volume 1.4e-14 / −2.1e-14 (np 1 / np 2), step 20 laysum 3.6e-12 m, w1
  7.0e-19 m/s, hlaw 5.5e-12 m, spreads 0. Script `/work/ab0995/a270088/fesom3_se_t02_jobs/t31_build_ctest.sbatch` (logs `t31b_*`); debug jobs
  28218130, 28218139, 28218151, 28218161, 28218172/3

#### Task 3.2: GATE Stage 3 — zstar np 1 and np > 1 ✅

**Files:**
- Modify: `tools/run_conserve_pi.sh` (➕ the zstar SE block, its own `while read` list after the linfs SE block)

- [x] `run_conserve_pi.sh` zstar cases with SE (np 1/2/8): hard heat/salt conservation as in the existing zstar
  cases; layer sum and w(1) at rounding. The new block runs default np 1/2/8, vinv rvo_upwind 1 + ice np 1/8 and
  GM + Redi + TKE + w split np 1 with `FESOM3_CONSERVE_TOL=$TOL` (heat, salt, volume) and gates the run maxima of
  the SE_CHECK tokens (etacomp, trim, msl net of the freshwater, spreads exactly 0) and of the SE_CHECK_TAIL tokens
  (laysum, w1, hlaw), one line of each per step. Levels over 1000 steps (job 28218186, M = 50, χ = 0.1, the six
  configs; the 20-step gate cannot reach them): **etacomp 4.9e-15 m, trim 5.7e-13 m²/s, |msl| 6.9e-16 m, laysum
  1.8e-11 m, w1 3.1e-18 m/s, hlaw 7.3e-12 m, spreads 0**; drift heat ≤ 6.4e-15, salt ≤ 1.6e-14, volume ≤ 4.3e-14
  (SI zstar default np 1: 1.1e-14, 9.9e-15, 1.4e-14), ice masses ≤ 2.8e-15. Tolerances (×100): SE_Z_TOL_ETACOMP
  5e-13, _TRIM 6e-11, _MSL 7e-14, _LAYSUM 2e-9, _W1 4e-16, _HLAW 8e-10. Final 20-step gate (job 28218430):
  **GATE OK, 34 configs**; zstar SE: etacomp ≤ 2.5e-15 m, trim ≤ 1.7e-13 m²/s, |msl| ≤ 1.3e-16 m, laysum ≤ 4.5e-12
  m, w1 ≤ 1.4e-18 m/s, hlaw 6.4e-12 m, spreads 0, heat/salt drift ≤ 3.2e-15 / 2.6e-14
- [x] CORE2 zstar np 8 (480 steps) and np 1 (48 steps), SE_CHECK: all invariants and copy spreads; record the levels.
  `prod_env 1e-5` + `FESOM3_WHICH_ALE=zstar FESOM3_SSH_SE=1 FESOM3_SE_CHECK=1 FESOM3_STEP_DIAG=1`, M = 50 (M_min 35):
  - np 1, 48 steps, **χ = 0.1** (job 28218186, 400 s; SI 375 s): etacomp ≤ 1.8e-14 m, trim ≤ 3.4e-13 m²/s, |msl|
    ≤ 4.5e-16 m, |fw| ≤ 5.6e-17 m, laysum ≤ 5.5e-12 m, w1 ≤ 1.1e-17 m/s, hlaw ≤ 7.3e-12 m, no NaN; max|η| 4.33 m
    (step 31; SI 1.66 m, the linfs SE/SI picture of Task 1.5), max CFL_z 0.63 (SI 0.61), divsum ≤ 1.3e-15;
  - np 8, 480 steps, **χ = 0.25** (job 28218429, 849 s): **etacomp ≤ 7.2e-14 m, trim ≤ 2.7e-12 m²/s, |msl| ≤ 2.1e-15
    m, |fw| ≤ 4.9e-18 m, laysum ≤ 1.3e-11 m, w1 ≤ 4.3e-17 m/s, hlaw ≤ 8.2e-12 m, spreadU = spreadUm = 0, dUV = dAB
    = dsig = 0**, divsum ≤ 1.1e-15; max|η| 4.67 m (step 67), 2.05 m at step 480 (SI zstar 2.03 m), max CFL_z 1.59
    (SI 1.58), max|uv| 2.96 m/s (SI 2.96)
  - ⚠️ **At the default χ = 0.1 the np 8 run blows up at step 331** (job 28218186; floating overflow). Not zstar:
    linfs SE np 8 blows up at step 337-340 and zstar with M = 100 at step 187 (job 28218261), the Stage 2 finding
    (np 8 and np 128, linfs). The invariants stay at rounding to the last step (etacomp 2.4e-14 m, laysum 1.0e-11
    m, hlaw 7.3e-12 m, w1 1.4e-17 m/s, spreads 0), so the scheme stays consistent and the instability is
    dynamical: a cluster of 30 m vertex columns (aux3d depth 20 m) on the Antarctic shelf near 75°W, 72°S under
    full ice cover (m_ice ≈ 2.6 m) sits near η ≈ −2 m with grid-scale jumps of several metres between neighbours
    and from step to step (steps 310-330), until η reaches ±17 m at step 331 (node dumps of steps 310-331, job
    28218261, `fesom3_se_t02_jobs/py/se_eta_series.py`). As Stage 2 decided, the defaults stay (Nikolay's call,
    Stage 5) and the 480-step gate runs at χ = 0.25 (SI zstar np 8, job 28218186, is smooth: max|η| ≤ 2.03 m)
  - CORE2 zstar SE_CHECK thresholds (×100 of the max over both runs; spreads exactly 0):

    | invariant | CORE2 zstar np 1 (48, χ 0.1) | np 8 (480, χ 0.25) | CORE2 zstar gate |
    |---|---|---|---|
    | etacomp [m] | 1.8e-14 | 7.2e-14 | 8e-12 |
    | trim [m²/s] | 3.4e-13 | 2.7e-12 | 3e-10 |
    | \|msl\| net of fw [m] | 4.5e-16 | 2.1e-15 | 3e-13 |
    | laysum [m] | 5.5e-12 | 1.3e-11 | 2e-9 |
    | w1 [m/s] | 1.1e-17 | 4.3e-17 | 5e-15 |
    | hlaw [m] | 7.3e-12 | 8.2e-12 | 9e-10 |
    | spreadU, spreadUm | 0 | 0 | 0 exactly |

    |fw| stays at rounding on CORE2 because the ice-ocean coupling balances the global net freshwater into
    water_flux (`mod_ice_oce_coupling.F90:228`): W enters η per node (etacomp and w(1) cover it) but not the global
    mean; a nonzero global fw is covered by test_ssh_se Z4/Z5
- [x] G0. Job 28218430: **G0 GREEN** (linfs and zstar CORE2: 99 checkpoint variables and 1 649 154 dump records
  max|Δ| = 0, 20 step lines equal; conserve 249 lines, 26 configs), 290 s; Intel `fesom_lifecycle_native_mr` sha256
  `79b48650d3a8bcfe…`. Scripts `/work/ab0995/a270088/fesom3_se_t02_jobs/t32_*` (gate, diag, where, core2chi,
  final); run dirs `/scratch/a/a270088/se_t32/`, G0 `/scratch/a/a270088/g0_se/t32_28218430`. Stage 3 cost (Tasks
  3.1-3.2, all 13 jobs, sacct AllocCPUS × elapsed): ≈ 0.6 node-h

### Stage 4 — Restart

#### Task 4.1: zstar derived thickness after a restart ✅

**Files:**
- Modify: `src/drivers/fesom_lifecycle_native_mr.F90`

- [x] after `restart_read` (:923), when `which_ALE == 'zstar'`: set hnode_new = hnode on nNodL, then call
  `update_thickness_ale(mesh, partit)`. This rebuilds zbar_3d_n and Z_3d_n bottom-up from the fixed bottom anchor,
  and helem (owned plus `exchange_elem`), with the same loops as the straight run, so the result is bitwise. linfs is
  untouched (G0). Done as written (9 lines + one `use oce_ale` line). No side effect: the routine only rewrites hnode
  with its own values and rebuilds zbar_3d_n / Z_3d_n / helem; it reads hnode_new, the cold bottom anchor
  zbar_3d_n(nlevels_nod2D-1) and elem_w, all valid after restart_read. hbar_old is reset from hbar in
  compute_hbar_ale before any use; nothing else in the step is derived incrementally from the thicknesses
- [x] verify: `run_restart_gate_core2.sh` with `FESOM3_WHICH_ALE=zstar` and **SI** shows whether the remaining
  SI-zstar gaps (dhe, stiffness) still break SI; record the result, do not fix it (D6). The split-explicit zstar
  restart is gated in Task 4.3. Results (CORE2, mid and month-boundary splits, np 1 and 2; plus the diagnostic
  N = 3, K = 2, whose final step is the first resumed step, so `dump_diff.py` names the first diverging substep):
  - **before** (se-t41 HEAD d23361c, job 28216193, 3.7 min): FAIL, 26 of the checkpoint variables differ at N4 K2
    (np 1 mid: eta_n 8.0e-4, hnode 4.0e-4, temp 2.9e-2, salt 5.8e-3, u 3.7e-4, v 5.4e-4, w 1.5e-6, sigma11 3.9e2,
    hice 1.7e-4, urhs_AB 3.0e1; np 2 the same to 2-3 digits). The first resumed step departs at substep 1
    PRESSURE_BV (density 2.2e-3): pressure_bv reads the cold Z_3d_n
  - **after** (job 28216287, 4.0 min): still FAIL (26 variables, eta_n 1.4e-3, temp 1.0e-4, salt 3.8e-5, u 1.1e-4
    at N4 K2), but the first resumed step is now bitwise through substep 8 (pressure, mixing, ssh_rhs) and first
    departs at substep 9 SSH_SOLVE (d_eta 5.5e-4, np 1 and 2): the SI stiffness. `update_stiff_mat_ale` adds
    −dhe of the previous step to `ssh_stiff%values` every step and these increments accumulate from the cold start;
    the checkpoint has neither dhe (cold 0 on resume) nor the accumulated values. The Jacobi preconditioner is
    built once at each process's first solve (`oce_ssh_solve.F90:80-83`, `lfirst` save) from the then-current
    matrix, the cold one in both legs today (dhe = 0 at the first step); restoring dhe or the values would change that
  - **probe** (job 28216587, 4.7 min; source copy `fesom3_se_t02_jobs/t41_probe_src` = this fix plus the
    `update_stiff_mat_ale` call commented out, never committed): **GREEN** at np 1 and np 2, N4 K2 and N3 K2, mid
    and boundary, checkpoint and live dump max|Δ| = 0. So with the fix the only remaining zstar restart gap is
    the SI stiffness (dhe + accumulated values), out of scope under D6; nothing else needs rebuilding for SE
  - no regression (job 28216588, 6.4 min): ctest Intel 167/167, GNU 161/167 (the same 6 known failures);
    `run_g0_se.sh` G0 GREEN (linfs, zstar, conserve); linfs restart gate GREEN np 1 (217 s) and np 2 (151 s).
    Jobs, scripts and logs: `/work/ab0995/a270088/fesom3_se_t02_jobs/t41_*`; run dirs
    `/scratch/a/a270088/fesom3_se_rgate/t41_*`, G0 `/scratch/a/a270088/g0_se/t41_28216588`. 0.23 node-h in total

#### Task 4.2: SE fields in the checkpoint ✅

**Files:**
- Modify: `src/io/mod_io_restart.F90`
- ➕ Modify: `src/drivers/fesom_restartroundtrip.F90`, `tools/run_restartroundtrip.sh` (the SE round trip and
  pre-check), `tools/run_restart_gate_core2.sh`, `tools/run_restart_gate_multirank.sh` (echo of the case env and
  of each leg's se_init line)

- [x] in `restart_register_state` (:280-354), `if (dyn%use_ssh_se)`:
  - `se_eta1`, `se_eta2` (node, default halo);
  - `se_ubt{0,1,2}_{u,v}` (elem, `RESTART_HALO_ELEM`);
  - `se_sbar_old_{u,v}` (elem, `RESTART_HALO_NONE`);
  - the rank-0 count of registered fields grows by 10

  Done as written, after sigma (2-D pointers into `se_eta(:,k)`, `se_ubt(c,:,k)`, `se_sbar_old(c,:)`, units m and
  m2/s); a flag `R%se` in `t_restart` marks the SE set for the read pre-check. The halo modes, from the code:
  - η levels are kept on owned + halo nodes, each level `exchange_nod`'ed in its substep → NODE;
  - with `se_visc_ah > 0` V reads Ū at eDim neighbours; level 0 is `exchange_elem`'ed after every substep and the
    shift copies over nElemL (owned + eDim), so all three levels hold owner values there → ELEM;
  - š^{n−1} is written and read on owned elements only (`se_bt_forcing`); its halo keeps se_init's 0 in a
    straight run and in a resumed one → NONE.

  ⚠️ With `se_visc_ah = 0` the eDim entries of the Ū ring stay at their cold-start values in a straight run, while
  a resumed run (ELEM) gets owner values there. Nothing reads them in that mode (the exchange and V are skipped,
  `se_ubtn`'s copy is used only by V, `se_inflow` reads owned elements); the visc_ah = 0 gate of Task 4.3 is the
  test: a read of a stale entry would make the resumed run differ, and it is max|Δ| = 0. The registered count on
  the CORE2 lifecycle (default physics, no TKE) goes **27 → 37** (log line `restart: registered 37 prognostic
  fields`); at most 42 with AB3 + TKE + SE, so RESTART_MAXF = 64 suffices. SE off: the set is unchanged (G0)
- [x] pre-check on read: SE on and the checkpoint has no `se_eta1/.zgroup` → `die('checkpoint has no split-explicit
  state: written with FESOM3_SSH_SE off; SI<->SE restarts are not supported')`. In `restart_read` right after the
  folder is resolved, before any field is read, through the module's `zarr_check` (error stop on every rank; the
  folder is appended to the message). Not covered: an SE checkpoint read with SE off reads silently (the se_*
  stores are ignored); the plan asks only for the SE-on direction
- [x] verify that both legs of the restart gates inherit `FESOM3_SSH_SE`, `FESOM3_SE_*` and `FESOM3_WHICH_ALE`
  (`run_seg` uses `env "${e[@]}"`). `run_seg` (core2 gate) and both legs of the multirank gate set only the restart,
  run-length and dump variables on top of the inherited environment. Both scripts now print the case env
  (`FESOM3_WHICH_ALE`, `FESOM3_SSH_SE`, `FESOM3_SE_*`, `FESOM3_SHARED_SYNC`) and grep each leg's se_init line
  (`oce_ssh_se:   M = 50, dtbt = 36.000 s, chi = 0.1000, visc_ah = …`); in job 28218753 every leg of every SE gate
  shows it with the requested visc_ah, and SI runs print nothing new
- [x] ➕ test: `fesom_restartroundtrip` (pi) with `FESOM3_RESTARTRT_SE=1` allocates the SE rings with se_init's
  shapes (nNodL, nElemL), fills distinct owned values per ring level and component, and round-trips **38 fields
  max|Δ| = 0 over the full local extent** at np 1 and np 2 (the corrupt step now uses each array's own size, as the
  SE element rings are nElemL long, not nElemF). `FESOM3_RESTARTRT_SE=2` writes without SE and reads with SE
  registered: **aborts with the message at np 1 and np 2** (exit 1). Both in `tools/run_restartroundtrip.sh`
  (AB2/AB3 28/32 fields unchanged); GATE GREEN, 33 s, job 28218753. ctest unchanged (176): **Intel 176/176, GNU
  170/176 (the 6 known failures)**, job 28218753; Intel `fesom_lifecycle_native_mr` sha256 `c27836e46fd1331e…`

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
  - (b) M = M_min from the guard (➕ the operator-based guard of Task 0.4: 40 at χ = 0.1, 48 at χ = 0.25 on CORE2
    at Δt 1800 s);
  - (c) `se_visc_ah = 0`;
  - (d) zstar headline
- [ ] per screen, record: max|η| over time, cold-start peak, max CFL_z, max|uv|, divsum, SE_CHECK, NaN. Decide
  whether the linear viscosity suffices (⚠️ if (a) misbehaves: try the γ1 term in the same gather operator, as a
  new task)
- ➕ ⚠️ from Task 2.2 (480 steps, np 8): (a) χ = 0.1, M = 50 blows up at step 337 (grid-scale η on the 20 m
  Antarctic shelf near 74°W 72°S); (b) M = M_min = 35 blows up at step 5 at χ = 0.1 (the guard is not a stability
  limit); (c) `se_visc_ah = 0` blows up at step 298; M = 100 at step 192. χ = 0.25 and `se_visc_ah = 0.1` run 480
  steps. So the screens start from a decision on the default χ / `se_visc_ah` (and the guard's safety factor)
- ➕ ⚠️ **Diagnosis of both Task 2.2 blow-ups (2026-10-18): the scheme, not a code defect; nothing changed in
  src/.** Diagnostic build = HEAD 376f187 plus env-only hacks, never committed
  (`/work/ab0995/a270088/fesom3_se_diag/src_dbg1`): `DBG_EIG` (in se_init: power iteration and an F = 0 stability
  test of the real `se_subcycle`), `DBG_SHELF` (in se_trim_finish: one line per step for the shallow Antarctic shelf,
  lat < 60°S and H ≤ 40 m: max|η|, grid-scale noise |η − mean of the surrounding element means|, |U^M − ⟨⟨Ū⟩⟩|/H,
  η at 4 probe nodes of the blow-up cluster), `DBG_CD` (bottom drag), `DBG_NO_IOSTRESS` (ice→ocean stress zeroed),
  `DBG_ICE_NOTILT` (the ice sees η = 0). Jobs `/work/ab0995/a270088/fesom3_se_ref/jobs/dbg_eig.sbatch` (28218500),
  `dbg_shelf.sbatch` (28218513), `dbg_margin.sbatch` (28218732); run dirs `/scratch/a/a270088/se_dbg/`; 0.35 node-h.
  CORE2 `prod_env`, np 8, Δt 1800 s, cold start throughout.
  - **(A) M = 35 at step 5: the CFL guard underestimates the fastest discrete gravity wave; the substeps reach
    exactly their theoretical limit.** Power iteration of the substeps' linear operator Aη = −I(H·(−g∇η))/areasvol
    (CORE2, linfs H): **ω_max = 3.152e-2 s⁻¹** (8 digits after 50 iterations). Its eigenvector peaks in the
    Kuril–Kamchatka trench (gid 81175, 160.8°E 50.9°N, H 5712 m, res 28.1 km, ω_max·res/c = 3.76 there), not on a
    shelf; the guard's critical node is in the Aleutian trench (gid 32328, 166.4°W 52.1°N, H 6000 m, c/res =
    9.579e-3 s⁻¹). ω_max/(c/res)_max = 3.29, so at the guard limit ω_max·dtbt = 1.645. The AB3–AM4 limit of the
    oce_ssh_se weights (largest root of z⁴(z−1)² + a²(w₀z² + w₁z + w₂)(δz³ + d₀z² + γz + ζ), a = ω·dtbt) is
    **a_crit = 1.853 / 1.579 / 1.329 at χ = 0 / 0.1 / 0.25**, so the stable minimum on CORE2 at Δt 1800 s is
    **M = 31 / 36 / 43**, not the guard's 35. The real substep loop (F = 0, U = 0, η = 10⁻³ × the eigenvector)
    agrees with the theory to the substep: χ = 0.1 unstable at M = 30 / 33 / 35 (growth 2.56 / 1.73 / 1.24 per
    substep; theory 2.68 / 1.79 / 1.26), stable from M = 38; χ = 0.25 unstable up to M = 40, stable at 45; χ = 0
    unstable at 30, stable at 33; the same with `se_visc_ah = 0`. 1.24³⁵ ≈ 1.6e3 per baroclinic step is the growth of
    the Task 2.2 M = 35 run (×3300 per step at steps 2-3). M = 50 and M = 100 at χ = 0.1 decay over 20 000 F = 0
    substeps (no slow substep instability). The ratio 3.29 is mesh-dependent: an ideal equilateral P1 mesh with res =
    2√(areasvol/π) gives ω_max·res/c = √6·1.05 = 2.57. **At χ = 0.25 the guard's M_min = 35 admits the unstable
    M = 35-42.**
  - **(B) χ = 0.1, M = 50, step 337: an instability of a grid-scale gravity mode on the 20 m shelf, fed through
    the sea ice once per baroclinic step.** The diagnostic build reproduces the blow-up (steps 337-340). The shelf
    noise starts growing at step ~150, at **+0.9 %/step** (steps 150-330; 0.31 m at step 150, 1.37 m at step 330),
    and the probe nodes oscillate with a **period of 3.0 baroclinic steps (ωτ ≈ 2.1)**: a grid-scale gravity wave in
    20 m of water. The SM substeps damp it only by about χ(ωτ)²/M ≈ 0.9 %/step (χ = 0.1, M = 50; ∝ 1/M), while SI's
    backward Euler damps it to 1/√(1 + (ωτ)²) ≈ 0.43 in amplitude per step, so SI never sees it. On/off runs at χ =
    0.1, M = 50, 480 steps:
    - **no ice→ocean stress** (R1): runs, shelf noise **decays −0.45 %/step** (0.10 m at step 330), no 3-step period;
      max|η| 4.74 m (step 67, a gulf), max CFL_z 1.60, the same as the χ = 0.25 gate run;
    - **ice sees η = 0** (R3, no ssh-tilt force on the ice): runs, noise **decays −0.45 %/step** (0.09 m at step
      330), max|η| 4.74 m, CFL_z 1.60;
    - **no bottom drag** (R2): runs, growth reduced to +0.1 … +0.6 %/step, then decays (0.31 m at step 330);
      max|η| 8.80 m (step 156, without drag elsewhere);
    - margins (job 28218732, 400 steps): **χ = 0.25** noise decays −0.3 … −0.55 %/step (0.05 m at step 330);
      **`se_visc_ah = 0.1`** decays −0.05 … −0.65 %/step (0.08 m); **M = 100** the same cluster grows from step ~60
      at +0.9 … +1.6 %/step and blows up at step 190-192.

    So the feedback runs through the ice: before each ocean step the runloop calls ocean2ice → ice_timestep →
    oce_fluxes_mom, so the ice reads hbar = ηⁿ (the previous step's η^M), the grid-scale tilt −m g∇ηⁿ drives the
    compact ice (a = 1), and the ice hands the force to the ocean through the ice→ocean stress, which enters the
    barotropic forcing F frozen over the whole step [tⁿ, tⁿ⁺¹].
    Cutting either link removes the growth; the explicit bottom drag (also frozen in F, from the previous step's
    velocity) adds to it. It is not the linfs breakdown (CFL_z > 1 appears only after the noise has grown, and R1/R3
    run with CFL_z 1.6 elsewhere), and not a defect of SE on shallow or boundary elements (R1/R3 run the same SE code
    on the same elements and decay). The same coupling exists under SI (the ice reads hbar = ηⁿ⁺¹ there too); only
    the damping of the mode differs. A one-mode model of the SE substeps plus a pressure-like force of relative size
    ε = 0.03 taken from the elevation at the start of the step and frozen over it (`jobs/dbg_toy_ice.py`; the
    lagged-drag variant `dbg_toy_lag.py`, the limit `dbg_sm_limit.py`, the series `dbg_shelf_series.py`, the
    diagnostic patch `dbg_src_dbg1.diff`, all in `/work/ab0995/a270088/fesom3_se_ref/jobs/`) gives the observed pattern: χ = 0.1, M = 50: +0.9 %/step at ωτ = 2.2 (observed 2.1); `se_visc_ah`
    = 0: +1.4 %; M = 100: +1.5 %; χ = 0.25: +0.07 %; `se_visc_ah` = 0.1: −0.02 %. port_kokkos M12 recorded the same
    Antarctic-shelf death at χ = 0.05 with zstar and the flow-dependent Zenodo viscosity.
  - **Remedies for Nikolay to choose (none applied; (A) since done as a guard fix, ➕ Task 0.4):**
    - (A) guard: compute ω_max in se_init by power iteration (≤ 50 iterations of the 2-D operator on CORE2) and set
      M_min = ⌈τ·ω_max/(s·a_crit(χ))⌉ with a_crit from the polynomial above and a safety s < 1 (s = 0.9 gives
      M_min = 40 / 48 at χ = 0.1 / 0.25); or keep the res-based bound with a smaller factor: on CORE2 the true limit
      is 0.48 (χ = 0.1) / 0.40 (χ = 0.25) / 0.56 (χ = 0) instead of 0.5, but the ratio is mesh-dependent. Screen
      (b) "M = M_min from the guard" needs the corrected M_min. ✅ Done as ➕ Task 0.4 (100 power iterations,
      Schur–Cohn a_crit, s = 0.9: M_min 40 / 48 on CORE2; M = 35 now stops in se_init).
    - (B1) χ = 0.25: measured stable (480 steps np 8 and np 128, noise decays); lowers the substep limit (M ≥ 43 at
      Δt 1800 s; M = 50 has a 15 % margin) and damps the resolved fast waves more (max|η| 4.70 vs 4.74 m at step
      67 in R1).
    - (B2) `se_visc_ah = 0.1`: measured stable; V(Ūᵐ − Ūⁿ) acts only on the change within the step, so it damps
      the fast modes and leaves the slow flow alone; explicit-viscosity limit far away (ν·dtbt ≈ 24·0.1·36 s/l ≈
      4e-3 at l = 20 km).
    - (B3) give the ice a time-filtered elevation instead of η^M (targets the feedback path; R3 shows that the path
      is the whole growth); not tested.
    - (B4) the SM2005 averaging filter (secondary weights) for ⟨⟨Ū⟩⟩ and the elevation handed to the slow consumers
      (D1 chose none); not tested.
    - (B5) the flow-dependent γ₀/γ₁ viscosity of port_kokkos; not tested here.
    - Not remedies: larger M (M = 100 fails earlier), χ < 0.1, `se_visc_ah = 0`.
    - Stage 3 (zstar): the same coupling and the same guard issue apply (port_kokkos died the same way with zstar);
      H_e(η) changes ω_max negligibly in the deep trenches where it peaks.

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
and halo exchange of uv. No global reductions outside SE_CHECK and the one-off CFL guard (➕ Task 0.4: in `se_init`,
100 power iterations with one allreduce sum and one `exchange_nod` each).

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
