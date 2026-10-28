# RK3 for FESOM3: handoff for the execution session (2026-10-28, second version)

**Task:** execute `docs/plans/2026-10-28-rk3.md` (23 tasks, Stages 0–8) task by task. The plan is authoritative: its
decision table D1–D14, the Technical Details and the checkboxes. This note says where things stand, how to work, and
what is easy to miss. It replaces the first version of this file: a first execution session was stopped and its work
undone, and Dmitry then changed D7 (§1, §3).

## 1. State

- Checkout `/home/a/a270029/fesom3.0`, local branch **`rk3`** at `d5b0d85` ("docs: add RK3 time stepping
  implementation plan"). It holds the plan and the brainstorm handoff `docs/plans/2026-10-28-rk3-handoff.md` (now
  background only). The branch is not pushed.
- `rk3` starts at main = origin/main `3bc68f5` (origin = github.com/FESOM/fesom3). **No code is written yet.**
- **Uncommitted; commit both with Task 1, with explicit paths:**
  - `docs/plans/2026-10-28-rk3.md`: the plan with Dmitry's new D7 (§3) and everything that follows from it (tests R2,
    R5, R6, R11, Z3; Tasks 2, 4, 6, 7, 9, 10, 12, 15, 19, 21; Technical Details §1, §2, §5, §7, §9; the questions for
    Sergey);
  - this file (untracked).
- Untracked, leave alone: `rk3.tex` (Sergey's algorithm text; it stays untracked unless Dmitry says otherwise),
  `docs/HANDOFF-next-session.md` (the runs report, unrelated), `results_2.0/`, `work/`.
- The production binary `build_intel_dp` was built from main. Don't rebuild or touch it.
- **The first execution session (2026-10-28), and why nothing of it remains.**
  - It did most of Task 1: the `3bc68f5` worktree and build, the reference binaries and the G0 references. It also
    began Task 2 (`src/oce/oce_rk3.F90`, the `t_dyn` fields).
  - Then Dmitry stopped it ("will update the plan!") and asked to undo all changes. The commit of the plan and the
    handoff, the code, the git worktree and `/work/ab0995/a270029/fesom3_rk3_ref` were all removed.
  - He then changed D7.
  - **Start at Task 1 from scratch.** §4 says how that session did Task 1; it worked.

## 2. How to work (Dmitry's standing rules and preferences)

- **Implement the whole plan in this session, task by task, without questions until it is done.** Dmitry,
  2026-10-28: "please implement the plan! dont question me untill you are done!". Don't ask about the execution mode
  either.
  - One commit per task with explicit paths.
  - Tick `[x]` immediately, add ✅ to a gated Task header, and mark ➕ new tasks and ⚠️ blockers in the plan.
  - Record measured numbers in the task line that gates them.
  - Stop only for a genuine blocker.
  - If Dmitry says "stop", stop at once and wait.
- **Commits:** no Claude/Co-Authored-By/session attribution, ever. Git is `/sw/spack-levante/git-2.43.3-bm2hrp/bin/git`.
  **Push only when Dmitry asks**; a PR to FESOM/fesom3 only when he asks.
- **Where the gates run** (Dmitry, 2026-10-28, answering the question the plan puts at Task 1):
  - G0 and the restart gates run on the login node.
  - The 480-step np 8 CORE2 screens (Tasks 12, 15, 20) run in an allocation: `srun -p shared -A ab0995`, e.g.
    `--ntasks=8 --mem=60G` as the header of `run_g0_se.sh` suggests.
  - The option he picked ended with "and I ask before each one". His later "dont question me until you are done"
    covers that: run these screens without a separate question.
  - Write the answer into the plan's Development Approach at Task 1 and tick that box. Don't ask it again.
- **Runs, otherwise:**
  - Dmitry runs the long simulations and configures the production runs himself. Task 21's compute-node timing and
    the 3000-step screens at Δt = 1800 s and 3600 s are his runs: under "no questions until done", put that request
    into the final report, together with the env blocks;
  - never edit `work/job_levante*`;
  - don't touch `build_intel_dp`, `work/`, `results*` or `/work/ab0995/a270029/meshes/*`.
- **Builds** in separate directories, `build_intel_dp_rk3` and `build_gnu_dp_rk3`, with cmake directly (the command
  is in the plan's Development Approach). `configure.sh` would write into `build_intel_dp`.
- The settled decisions are not reopened.

## 3. Decisions to keep in mind (details in the plan)

- Both barotropic versions: **CG first**, then SE (`FESOM3_SSH_SE=1` selects SE). linfs first, then zstar
  (Stage 6). Double precision only.
- **Stage-2 PGF = the textbook p(T\*)** (Dmitry corrected this explicitly): compute the stage-2 pressure *before* the
  stage-2 tracer update overwrites T\* in `valuesAB`. qq_new2 uses p(T\*\*); don't copy that.
- **w split (D7), changed by Dmitry on 2026-10-28 ("read carefully rk3.tex"): only in stage 3.**
  - `rk3.tex` has no w split at all: every stage diagnoses its own full w and advects with it.
  - Stages 1–2 use the full w of the stage for momentum advection and for every tracer, FCT or not
    (`vert_vel_ale(split=.false.)`: w_e = w, w_i = 0). A uniform tracer stays uniform in every stage, exactly.
  - Dmitry stated the rule for the tracers: "we use full w for stages 1 and 2! We only do apply w_split at the last
    stage!". The plan applies it to momentum as well, as `rk3.tex`'s Rⁿ uses wⁿ. He was told and did not object.
  - Stage 3 uses the split of the stage-3 w, made by `vert_vel_ale` as in AB2:
    - momentum: w_e explicit, w_i in the TDMA;
    - non-FCT tracers: w_e explicit, w_i in the diffusion TDMA;
    - FCT tracers: the split in the LO only; the HO and the antidiffusive flux use the full w.
  - With the split in the LO, one LO step is the default (`FESOM3_RK3_LO_NSUB=1`); stage 3 is then `do_oce_adv_tra`
    itself on T\*\*. N = 2, 3 stay as an option for a horizontal LO Courant number near 1.
  - There is no frozen split fraction and no `rk3_wfrac`.
  - **Known limit** (plan §5): momentum and non-FCT tracers are stable up to a per-cell vertical Courant number of
    about 9 at maxcfl 1. For comparison, the old rule (w_e in all stages) is stable at any value, and plain RK3 up to
    2.51. FCT tracers stay bounded by the limiter. The screens record the max; the final report says how close CORE2
    comes.
- **The stage velocity** is passed to the kernels as an optional `uv_in`; `dyn%uv` holds uⁿ until the final update.
  The stage tracer lives in `valuesAB`.
- **The new tracer driver** `do_oce_adv_tra_rk3` (stages 1–2, and the LO substeps for N > 1) sits next to
  `do_oce_adv_tra`, which stays untouched.
- **Default unchanged:** with `FESOM3_RK3` unset, everything must stay byte-identical to main (G0). The AB2 path gets
  only optional arguments whose absent case is today's code.

## 4. Findings that are easy to miss (all checked in the code)

### Gates and references

- **Gate scripts are unsafe with their defaults:**
  - `run_restart_gate_core2.sh`, `run_restart_gate_multirank.sh` and `run_restartroundtrip.sh` default to another
    user's (a270088) worktree binary and scratch dir, and they `rm -rf` the run dir;
  - `run_conserve_pi.sh` and `run_restart_gate_zstar.sh` default to `build_intel_dp`;
  - `run_g0_se.sh`'s default `RUN` (`/scratch/a/a270088/…`) is not writable for a270029.

  Always use the invocation block in the plan's Development Approach: `F3`, `BUILD`, `BIN`, `PY`, `REF`, and
  `RUN=/scratch/a/a270029/rk3/<gate>_<tag>`. The python `/work/ab0995/a270088/mambaforge/bin/python3` imports
  xarray + zarr (checked); use it read-only.
- **G0 references must be fresh.** The existing ones (`/work/ab0995/a270088/fesom3_se_ref`) are frozen at main
  `6cd6449`, and main has moved by 75 commits. Task 1 builds `3bc68f5` from a clean worktree and writes new ones to
  `/work/ab0995/a270029/fesom3_rk3_ref` (`MAKE_REF=1`).
- **Task 1 as the first session did it.** It worked; redo it the same way, with `REF=/work/ab0995/a270029/fesom3_rk3_ref`
  and `S=$REF/src_main_3bc68f5`:
  - `git worktree add --detach $S 3bc68f5`. `/work` was 97 % full, with 82 TB free.
  - Build inside the worktree, in the background:
    `cd $S && source env.sh intel && cmake -S . -B build_intel_dp -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" && cmake --build build_intel_dp -j 16`.
    `$S/build_intel_dp` is the worktree's own directory, not the production one.
  - Copy `fesom_lifecycle_native_mr` and `fesom_conserve` to `$REF/bin_main_3bc68f5`, then
    `sha256sum fesom_* > SHA256SUMS` there.
  - Write the references with the **worktree's** script and `F3=$S`, so that `env.sh`, `prod_env.sh` and the frozen
    copy of `run_conserve_pi.sh` all come from main:
    `cd $S && MAKE_REF=1 F3=$S NEW_BIN=$REF/bin_main_3bc68f5 REF=$REF PY=$PY bash $S/tools/run_g0_se.sh`.
    With `MAKE_REF=1` the script ignores `RUN` and writes `$REF/g0`.
  - On the login node this took 4 min 29 s:
    - linfs and zstar each wrote `fesom.1958.001.36000` with 20 step lines;
    - conserve ran 43 configs and printed `run_conserve_pi: GATE OK`.
  - Not reached: the compare-mode control with the same binary, the ctest baselines of `3bc68f5` (Intel and GNU),
    and the record in the plan.
  - The worktree is registered in the repo's `.git/worktrees`. `git worktree remove --force` and
    `git worktree prune` remove it.
- **GNU ctest is not fully green on main:** TESTING.md:494 lists 6 deterministic failures. "Green" for GNU means the
  same failure set as the Task 1 baseline.
- **G0 also compares** the run logs (`run_conserve_pi.sh` parses fixed lines), the timer report and
  `checkpoint.json` (except `fesom_git`). Hence:
  - RK3 prints nothing when off;
  - the new timers are TQUIET and TSUBSET;
  - `"time_scheme"` is written only by RK3.

### Code facts for Task 2 (read in the first session)

- `oce_ssh_se`'s knob helpers (`get_knob`, `parse_01`, `parse_int`, `parse_real`, `se_die`) are private, so
  `oce_rk3` needs its own copies. `rk3_die` should do what `se_die` does: MPI_Abort on partit's communicator, then
  `error stop`.
- `se_env_requested` is public: `rk3_init` can use it to say CG or SE in its header line.
- The first-call guard of `step_oce` is the `se_checked` block (`mod_step_oce.F90:113-116`). `rk3_step_guard` goes
  there, and the RK3 hook goes right after it.
- `mod_timer`:
  - `NTIMER = 20`, a pre-order tree; `TNAME`, `TPARENT`, `TSUBSET` and `TQUIET` are sized by `NTIMER`.
  - Id 20 (tracer) is the last child of `TMR_STEP_OCE`. Ids 21–23 as further children, with TSUBSET and TQUIET,
    keep the tree order.
- The drivers allocate `dyn%uv`, `uv_rhs` and the PGF as `(…, nElemF)`, and `w`, `w_e`, `w_i` as `(nl, nNodL)`
  (`fesom_conserve.F90:322-336`). `alpha` and `theta` come from `mod_param_phys`.
- `rk3_init` goes right before `se_init`: `fesom_lifecycle_native_mr.F90:973`, `fesom_conserve.F90:558`.
- AB2's MUSCL gradient `tr_xy` is the gradient of `values` (Tⁿ), not of `valuesAB` (`oce_tracer_mod.F90:101-114`,
  LESSONS L11). RK3's `rk3_tracer_prep` builds it from the stage tracer in `valuesAB` (plan Task 6). This is a
  deliberate difference.
- `compute_vel_rhs` adds momentum advection only for `momadv_opt` 1 or 2. A test that sets `momadv_opt = 0` and
  uses uniform T/S (zero PGF) and η = 0 gets Coriolis alone: the setting for R4.
- A test that needs the whole model state (dyn, tracers and the PHC IC on pi) can copy `fesom_conserve`'s init
  (roughly lines 160–560); the tree duplicates drivers on purpose. `test/test_util.F90` has `setup_test_mesh`,
  `check`, `finish`, the reductions and `hval`.
- The system `python3` has no numpy. For scratch analysis use `/work/ab0995/a270088/mambaforge/bin/python3 -I`.

### The kernels

- **The kernels read `dynamics%uv` directly:** `compute_vel_rhs`, `momentum_adv_vinv`/`relative_vorticity`,
  `momentum_adv_scalar`, `vert_vel_ale`, `compute_vel_nodes`, `viscosity_filter` and `impl_vert_visc_ale` (the base
  velocity). Only the first four get `uv_in`. The TDMA, the viscosity and the vel-nodes must keep seeing uⁿ.
- **`compute_vel_rhs` starts from ab1·`uv_rhsAB(prev)`** (`oce_dyn_velrhs.F90:100-103`), so RK3 needs the
  `no_history` option (Euler weights without reading the history).
- **`init_tracers_AB` is more than AB:** it also zeroes `del_ttf*` and builds `tr_xy` per tracer
  (`oce_tracer_mod.F90:63-70, 109-114`), and `flux2dtracer` only adds. RK3 needs `rk3_tracer_prep` per tracer *in
  every stage*, stage 3 included (plan Tasks 6 and 10).
- **The flux-kernel convention:** `o_init_zero=.false.` returns new − old (`oce_adv_tra_hor.F90:115-120`). The LO
  substep sums (N > 1 only) need their own scratch arrays (plan Task 7).
- **The stage update** must have the identity form of `diff_tracers_ale` (`oce_ale_tracer.F90:390-393`), otherwise
  T\*\* ≠ Tⁿ by an ulp at zero transport and the bitwise test of Task 10 fails. That test runs with one LO step
  (`FESOM3_RK3_LO_NSUB=1`, the default).
- **SE:**
  - `se_H`/`se_c` are zero until `se_elem_depth`, which only `se_bt_forcing` calls (`oce_ssh_se.F90:602, 806`), so
    `se_bt_forcing_rk3` must call it first;
  - lfirst sets only a2 = 1 (`:810-812`), and the a1·`se_sbar_old` term stays (`:829-830`), so the cold AB start
    after an RK3 checkpoint must zero `se_sbar_old`;
  - `se_trim_finish` and `se_check_values` assume the AB2 trim of `uv`; RK3 needs `se_rk3_finish` and a
    transport-velocity argument.
- **The dumps live only in `step_oce`.** `step_oce_rk3` must emit the same substep ids, or the restart gate's dump
  comparison fails (`run_restart_gate_core2.sh:144-149`).
- **CG = the RK3 corrector only for α = θ = 1** (`oce_ale.F90:149`, `oce_ssh_rhs.F90:374`); `rk3_init` stops
  otherwise.
- **`rk3_H` must be rebuilt every step**, because a restart rebuilds the geometry after `rk3_init`
  (`native_mr:973` vs `:1004-1006`).
- **Stage 3 must replicate:**
  - `sw_alpha_beta` for KPP|GM|Redi (`step_oce:169`);
  - the Redi `Ki` switch (`:307-315`);
  - `update_stiff_mat_ale` when not linfs (`:254`);
  - `se_check_tail` (`:328`).
- **zstar (Stage 6):**
  - no `water_flux` in stages 1–2 (W only in stage 3);
  - the order inside a stage: R_s on the h_{s−1} geometry → tracers → `rk3_stage_geometry(h_s)` → the trim with h_s
    weights;
  - `hbar` is restored to ηⁿ before the barotropic block.
- **The analytic patches are 1-rank only.** Tests at np 2 run on the pi_w and mixed mesh files (`add_mesh_test`).
  FESOM3 has no linear EOS.

## 5. What to read first

1. `docs/plans/2026-10-28-rk3.md`: the whole plan, especially D7, Development Approach, Testing Strategy and
   Technical Details §2 (the call sequence) and §5 (tracers, the stage-3 split, the stability limit).
2. `rk3.tex` (read it carefully, as Dmitry asked) and `~/qq_new2/oce_stepRK3_SE.F90`, `~/qq_new2/oce_stepRK3.F90`:
   the reference algorithm.
3. The FESOM3 files the plan's Context names, starting with `src/step/mod_step_oce.F90`,
   `src/oce/oce_dyn_velrhs.F90`, `src/oce/oce_ale.F90`, `src/oce/oce_adv_tra_driver.F90`,
   `src/oce/oce_ale_tracer.F90` and `src/oce/oce_ssh_se.F90`.
4. The test templates `test/test_wimpl_tra.F90` and `test/test_ssh_se.F90`, and the conserve ctest block
   `test/CMakeLists.txt:278-336`.
5. LESSONS L54, L57, L58, L60, L61, L70, L71B, L73–L78.

## 6. When the plan is done

- Final G0 against the Task 1 references.
- The docs of Task 23: README run switches, TESTING, IMPLEMENTATION, HANDOFF, LESSONS from L79.
- Move the plan to `docs/plans/completed/`.
- Report to Dmitry:
  - the measured invariant levels;
  - the CORE2 screen results, with the max vertical Courant number against D7's limit of about 9 and the max
    horizontal LO Courant number (the choice of N);
  - the open questions for Sergey (plan Post-Completion);
  - the env blocks for his RK3 runs (`FESOM3_RK3=1`, `FESOM3_STEP_PER_DAY`, `FESOM3_SE_*`), and the runs of Task 21
    that are his (the compute-node timing, the 3000-step screens at Δt = 1800 s and 3600 s).
- Then wait for his go before pushing.
