# momadv_opt==1: vector-invariant momentum advection

**STATUS: COMPLETE.** 32/32 ctest (incl. `test_vinv` np 1+2); conservation gate green on
15 configurations including `momadv-vinv` at np=1 and np=8.

## Overview

FESOM3 currently offers one momentum-advection operator, the flux/scalar form
(`momadv_opt==2`, `momentum_adv_scalar` in `src/oce/oce_dyn_velrhs.F90`). This plan adds
`momadv_opt==1`, the **vector-invariant** form

```
(curl u + f) x u  +  grad(u^2/2)  +  w du/dz
```

so the two operators can be compared in the same model, on the same mesh, with everything
else held fixed.

**Why this is a port and not a transcription.** The reference implementation is
`/home/a/a270029/qq/oce_vinv_mom_adv.F90` (routines `relative_vorticity`,
`v_inv_mom_adv`), which is **pre-ALE research code**. Our FESOM2 v2.7.3 oracle has the
`momadv_opt==1` branch but it is a stub that aborts:

```fortran
! /home/a/a270088/port2/fesom2/src/oce_ale_vel_rhs.F90:268
if (dynamics%momadv_opt==1) then
   if (mype==0) write(*,*) 'in moment not adapted mom_adv advection typ for ALE, check your namelist'
   call par_ex(partit%MPI_COMM_FESOM, partit%mype, 1)
```

So FESOM2 never adapted the vector-invariant form to ALE, and the user confirms it was
never tested there. **There is no oracle and no byte-gate for the new path.** That shapes
the whole plan: correctness has to come from analytic tests and a physical invariant
rather than from a reference dump.

## Context (from discovery)

- **Target:** `src/oce/oce_dyn_velrhs.F90` (326 lines) — `compute_vel_rhs` assembles
  AB2 + SSH gradient + PGF + Coriolis, then dispatches advection at line 129.
- **Reference:** `/home/a/a270029/qq/oce_vinv_mom_adv.F90` (535 lines). Contains
  `relative_vorticity`, `v_inv_mom_adv`, `v_inv_mom_adv_SE` (split-explicit variant, not
  needed — `use_ssh_se_subcycl=.false.` here).
- **Oracle:** `momadv_opt==1` aborts; no `relative_vorticity` anywhere in v2.7.3.
- **Available in our tree (all verified):** `mesh%helem` (ALE element layer thickness),
  `mesh%area` (1-D on this branch), `mesh%edge2D_in` (boundary-edge split),
  `mesh%edge_cross_dxdy` `(4,edge2D)` **in metres** — same convention `qq` assumes —
  `mesh%gradient_sca`, `dynamics%w_e` `(nl,nod2D)` nodal, `dynamics%uv_rhsAB`.
- **Absent:** `w_cv` control-volume weights, `vorticity`, `area(nz,n)` (collapsed to 1-D
  by the bottom-at-vertices work on this branch).
- Nine drivers allocate `dyn%work%uvnode_rhs`; the new `vorticity` field goes beside it in
  each.

## Development Approach

- **testing approach**: **TDD (tests first)**. The expected answers are known exactly in
  advance (solid-body rotation gives `zeta = 2*Omega`), and with no oracle the tests *are*
  the specification. Write `test/test_vinv.F90` before the operators.
- complete each task fully before moving to the next
- make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
- **CRITICAL: all tests must pass before starting the next task** — no exceptions
- **CRITICAL: update this plan file when scope changes during implementation**
- run tests after each change
- maintain backward compatibility: `momadv_opt==2` must stay **bit-identical**

## Testing Strategy

- **unit tests**: `test/test_vinv.F90`, registered via `add_fesom_test(test_vinv <np>)` in
  `test/CMakeLists.txt`, run at np 1 and 2. Analytic velocity fields on the real pi mesh
  with known vorticity.
- **invariant test**: ⚠️ **CORRECTED during Task 2.** The design originally proposed
  `u . [(f+zeta) x u] == 0` pointwise. That is **identically zero for any value of zeta**,
  because `(f+zeta)*zhat x u` is perpendicular to `u` by construction:
  `U*(V*zeta) + V*(-U*zeta) = 0`. It therefore tests only that the two signs are opposite
  and the same `zeta` is reused — it is insensitive to the vorticity VALUES and so does
  **not** validate `area(nz,n)->area(n)` or `w_cv->1/3` as claimed.
  Replaced by **global kinetic-energy conservation**, the property the vector-invariant
  form is actually chosen for:
  `sum_elem u . (advective tendency) * volume  ~  0`
  which does depend on `zeta`, on `grad(KE)` and on the vertical flux being mutually
  consistent. Added to `src/drivers/fesom_conserve.F90` beside heat/salt/volume. The cheap
  sign check is kept as a separate assertion since it costs nothing.
- **gate**: a `FESOM3_MOMADV_OPT=1` configuration added to `tools/run_conserve_pi.sh`.
- **no e2e tests** in this project (Fortran model; the conservation gate is the
  integration-level net).
- Commands: `source env/levante.dkrz.de/shell.intel` then
  `cd build_intel_dp && make -j16 && ctest`, and `bash tools/run_conserve_pi.sh`.

## Progress Tracking

- mark completed items with `[x]` immediately when done
- add newly discovered tasks with ➕ prefix
- document issues/blockers with ⚠️ prefix
- update plan if implementation deviates from original scope
- keep plan in sync with actual work done

## Solution Overview

`momentum_adv_vinv` is a **pure addition** into `UV_rhsAB`, exactly like
`momentum_adv_scalar`. `compute_vel_rhs` step 2 already writes `f x u`; the new routine
adds `zeta x u - grad(KE) + w du/dz`, which completes `(f + zeta) x u`. Steps 1, 2 and 4
of `compute_vel_rhs` are untouched, so no restructuring is needed and the `momadv_opt==2`
path cannot move.

Dispatch at `src/oce/oce_dyn_velrhs.F90:129` becomes:

```fortran
if (dynamics%momadv_opt == 1) then
    call momentum_adv_vinv(dynamics, mesh, partit)
else if (dynamics%momadv_opt == 2) then
    call momentum_adv_scalar(dynamics, mesh, partit)
end if
```

### Key design decisions

**Decision 1 — fidelity: faithful transcription of `qq`**, deviating only where ALE or our
conventions force it, with every deviation documented in the code. Chosen over a
"principled rewrite" so the provenance stays traceable and the deviations are an explicit,
reviewable list rather than scattered judgement calls.

**Decision 2 — drop the `Av` term.** `qq` keeps viscosity inside the advective flux even
in its implicit-viscosity branch:

```fortran
uvert(1,nz) = -umean*w + Av(nz,elem)*(U_n(nz-1,elem)-U_n(nz,elem))/(Z(nz-1)-Z(nz))
!!! Attention: Av above is necessary even if time stepping is implicit.
```

FESOM3 solves vertical viscosity implicitly in `impl_vert_visc_ale`, so carrying that term
would apply `Av du/dz` **twice** and over-damp momentum. Dropping it also means
`momadv_opt==1` and `momadv_opt==2` differ **only** in the advection operator, which is
the point of having both.

**Not transcribed:** `qq`'s `else` branch (explicit vertical viscosity — wind stress plus
`friction=0.005 !!! Soufflet` hard-coded over the real bottom drag). We take its
`i_vert_visc=.true.` branch, whose own comment reads "Do only advection".

## Technical Details

### Forced deviations from `qq`

| `qq` | FESOM3 | why forced |
|---|---|---|
| `dz = zbar(1:nl-1)-zbar(2:nl)` (global 1-D) | `mesh%helem(nz,elem)` | ALE thickness is per-element, per-step |
| `area(nz,n)` | `mesh%area(n)` | collapsed to 1-D by bottom-at-vertices |
| `w_cv(:,elem)` | `1.0_WP/3.0_WP` | absent from our mesh; triangles |
| `elnodes(4)`, `gradient_sca(1:4)/(5:8)` | `elnodes(3)`, `gradient_sca(1:3)/(4:6)` | MAX_NV=4 slicing (the L15 trap) |
| `exchange_nod3D(x)` | `exchange_nod(x, partit)`, `is_multirank`-guarded | halo API |
| loops from `1` | `ulevels(elem)..nlevels(elem)-1`, `ulevels_nod2D(n)..nlevels_nod2D(n)-1` | cavity bounds |
| `Wvel` | `dynamics%w_e` | same vertical velocity as `momadv_opt==2`, so the two are comparable |
| whole-RHS assembly (AB2, PGF, SSH, Coriolis, viscosity dispatch) | only the three advection blocks | the rest already exists in `compute_vel_rhs` |

`dz` appears twice and **both** uses become `helem`: the interpolation weight and the flux
divisor. This is the one place the ALE thickness enters twice, so it is where any
conservation error is most likely to appear.

### Block A — relative vorticity (edge loop, circulation integral)

```
c1 = dX1*U(el1) + dY1*V(el1) - dX2*U(el2) - dY2*V(el2)
vorticity(nz,n1) += c1 ;  vorticity(nz,n2) -= c1
```

`edge_cross_dxdy(1:2,·)` for `el1`, `(3:4,·)` for `el2`. **Keep `qq`'s three level
ranges** — both elements wet, then `el1`-only, then `el2`-only — these one-sided ranges are
what make the operator correct at a bathymetry step. Then `vorticity /= mesh%area(n)` and
one `exchange_nod`.

### Block B — kinetic energy and its gradient

```
KE_node(nz,elnodes) += 0.5*(U^2 + V^2) * (1/3) * elem_area(elem)
KE_node /= mesh%area(n)
KE_node = 0 at lateral-wall nodes            (edges beyond mesh%edge2D_in)
exchange_nod
per element:  Fx = sum(gradient_sca(1:3)*(-KE_node(elnodes))) ; Fy from (4:6) ; each * elem_area
```

### Block C — vertical flux

⚠️ **Bottom-face fix (found by plan review), a deviation FROM `qq` rather than a forced
one.** `qq` sets `uvert(:,nl1+1) = 0`, but `da` at `nz = nl1` reads `w_e(nlevels(e))`,
which is **not** zero under bottom-at-vertices: `vert_vel_ale` zeroes `Wvel` only at
`nlevels_nod2D(n)` (`src/oce/oce_ale.F90:373-392`) while `nlevels(e) = minval` over the
element's nodes. So on every element beside a deeper vertex column the telescoping breaks
and the bottom layer picks up a spurious `-w_bot*U/helem` — order 1e-8 m/s^2 on pi, ~0.2%
of `f*U`, concentrated along sloping topography where it would read as physics. Treat the
bottom face symmetrically with the surface:

```
uvert(:,ul)    = -w_top * UV(:,ul,elem)
uvert(:,nl1+1) = -w_bot * UV(:,nl1,elem)      ! NOT 0 -- see above
nz = ul+1 .. nl1:
   w     = (1/3) * sum(w_e(nz,elnodes))
   umean = (U(nz-1)*helem(nz) + U(nz)*helem(nz-1)) / (helem(nz-1) + helem(nz))
   uvert(1,nz) = -umean * w            ! no Av term (Decision 2)
```

then the divergence, into the same slot as Coriolis:

```
da = (1/3) * sum( w_e(nz,elnodes) - w_e(nz+1,elnodes) )
UV_rhsAB(1,1,nz,elem) += ( uvert(1,nz) - uvert(1,nz+1) + da*UV(1,nz,elem) ) * elem_area / helem(nz)
```

The `+ da*U` term converts `d(wu)/dz` into `w du/dz` — the energy-conserving form `qq`'s
comment refers to.

## What Goes Where

- **Implementation Steps** (`[ ]`): all code, tests, gate configuration and docs in this
  repo.
- **Post-Completion** (no checkboxes): choosing production settings and running the
  opt==1 vs opt==2 comparison, which is the user's call.

## Implementation Steps

### Task 1: Add the `vorticity` field and the `momadv_opt==1` dispatch stub

**Files:**
- Modify: `src/types/mod_dyn.F90`
- Modify: `src/oce/oce_dyn_velrhs.F90`
- Modify: `src/drivers/fesom_conserve.F90`, `fesom_lifecycle.F90`, `fesom_lifecycle_mr.F90`, `fesom_lifecycle_native.F90`, `fesom_lifecycle_native_mr.F90`, `fesom_pressuredump.F90`, `fesom_stepdump.F90`, `fesom_stepdump_mr.F90`, `fesom_stepfull_mr.F90`

- [x] add `vorticity` `(nl-1, nod2D)` to `t_dyn_work` in `mod_dyn.F90`, documented as the vector-invariant relative vorticity
- [x] allocate + zero-init it beside `uvnode_rhs` in all nine drivers (`nNodL` or `mesh%nod2D` matching each driver's existing convention)
- [x] add the `momadv_opt == 1` branch at `oce_dyn_velrhs.F90:129` calling a stub `momentum_adv_vinv` that does nothing yet
- [x] write a test asserting `momadv_opt==2` output is unchanged (run `ctest`; the existing suite IS this test)
- [x] run tests — `ctest` 30/30 and `bash tools/run_conserve_pi.sh` 13/13 GATE OK

➕ `vorticity` deliberately NOT added to `write/read_t_dyn_work`: unlike `uvnode_rhs`/`u_c`/`v_c`
   it is recomputed from `UV` every step, and adding it would change the restart layout.

### Task 2: Write `test/test_vinv.F90` with the analytic expectations (TDD — fails until Task 3)

**Files:**
- Create: `test/test_vinv.F90`
- Modify: `test/CMakeLists.txt`

- [x] create `test/test_vinv.F90` loading the pi mesh, following the `test_bottom.F90` harness style (`check_true`, `nfail`, `error stop 1`)
- [x] add the solid-body rotation case: set `UV(1,:,e) = -Omega*y_e`, `UV(2,:,e) = Omega*x_e`, call `relative_vorticity`, assert `zeta == 2*Omega` within discretisation tolerance on interior nodes
- [x] add the uniform-flow case: `UV = const`, assert `zeta == 0` to round-off AND the KE gradient `== 0` to round-off
- [x] add the linear-shear case: `UV(1,:,e) = alpha*y_e`, assert `zeta == -alpha`
- [x] register `add_fesom_test(test_vinv 1)` and `add_fesom_test(test_vinv 2)` in `test/CMakeLists.txt`
- [x] run tests — `test_vinv` is expected to FAIL here (stub); record the failure messages as the specification. Mark as `[x] (fails until Task 3)`

### Task 3: Implement Block A — `relative_vorticity`

**Files:**
- Create: `src/oce/oce_dyn_vinv.F90`

- [x] create the module with the header documenting the `qq` provenance, the stub-oracle situation, and the forced-deviation table
- [x] implement the edge-loop circulation integral with `edge_cross_dxdy(1:2)`/`(3:4)`, keeping `qq`'s three level ranges verbatim
- [x] normalise by `mesh%area(n)` (deviation from `area(nz,n)`, documented inline) and `exchange_nod` under `is_multirank`
- [x] verify the assertions in `test_vinv` PASS at np 1 and 2
- [x] run tests

➕ Zeroed `vorticity` over the FULL local range (owned+halo, all levels), not just wet
   owned rows: the edge scatter writes halo rows and `-init=zero` does not cover
   allocatables, so accumulating onto fresh heap could trap under `-fpe0`.

➕ **V2 validated the `area(nz,n)` -> `area(n)` deviation.** `zeta == -alpha` to relative
   5e-13 over 81729 node-levels holds only if `area(n) == sum of elem_area/3` over the
   node's wet adjacent elements. This is the review's T-c composition identity, obtained
   for free from the linear-shear case.

➕ **`qq`'s circulation sign is correct for our `edge_tri`** -- no flip needed. I briefly
   flipped it on a wrong diagnosis and reverted.

⚠️ Two test-design errors found and fixed while getting V2 to pass, both mine:
   (a) the analytic field must be linear in `coord_nod2D` (the ROTATED computational frame
       the geometry lives in), not `geo_coord_nod2D`. With the geographic field the test
       failed worst at the rotated pole (mean `zeta/(-alpha)` = 0.836).
   (b) the interior mask must loop `1..nEdgeO` (local owned edges), not `1..mesh%edge2D`
       (the GLOBAL count) -- the latter read past the local arrays and produced an
       all-false mask, so at np=2 the test silently asserted nothing.

### Task 4: Implement Block B — the `zeta x u` term AND the kinetic-energy gradient

⚠️ **The `zeta x u` term was MISSING from the original plan** (found by plan review). Blocks
A/B/C as first written computed the vorticity, the KE gradient and the vertical flux but
never multiplied `zeta x u` — i.e. they implemented `f x u - grad(KE) - w du/dz`, not the
operator in the Overview. In `qq` the `zeta` term and `grad(KE)` share one elemental loop
(`oce_vinv_mom_adv.F90:174-177`), so they belong in one task, but the `zeta` half must be
explicit:

```
zbar_e = (1/3) * sum( vorticity(nz, elnodes) )          ! element average of zeta
UV_rhsAB(1,1,nz,elem) += ( UV(2,nz,elem)*zbar_e + Fx ) * elem_area(elem)
UV_rhsAB(1,2,nz,elem) += (-UV(1,nz,elem)*zbar_e + Fy ) * elem_area(elem)
```

➕ **Missing deviation row:** `qq` averages `coriolis_node(elnodes)` and `vorticity` together
with the same `w_cv` weights, so `f` and `zeta` get identical treatment. FESOM3 step (2)
supplies the element-centre `mesh%coriolis(elem)` instead, so `f` and `zeta` are weighted
differently: `coriolis(elem) + sum(zeta)/3` rather than `sum(coriolis_node + zeta)/3`.
`mesh%coriolis_node` does exist, so this is a choice, not an absence — it keeps
`momadv_opt==2` bit-identical and uses FESOM2's standard Coriolis.

### Task 4b: Implement the kinetic-energy gradient

**Files:**
- Modify: `src/oce/oce_dyn_vinv.F90`

- [x] add the `KE_node` element scatter with `w_cv -> 1/3`, normalised by `mesh%area(n)`
- [x] zero `KE_node` at lateral-wall nodes using `mesh%edge2D_in`, then `exchange_nod`
- [x] add the per-element `grad(KE)` with `gradient_sca(1:3)`/`(4:6)` and `* elem_area`, adding into `UV_rhsAB`
- [x] extend `test_vinv` to assert `grad(KE) == 0` for uniform flow and that `KE_node == 0.5*|u|^2` for uniform flow
- [x] write a test that `KE_node == 0` at lateral-wall nodes
- [x] run tests — must pass before task 5

### Task 5: Implement Block C — the vertical flux

**Files:**
- Modify: `src/oce/oce_dyn_vinv.F90`

- [x] add the `uvert` build with `dz -> helem` in the interpolation weight, surface value `-w_top*UV`, and `uvert(:,nl1+1) = 0`
- [x] document inline that the `Av` term is deliberately absent (Decision 2) with the reason
- [x] add the flux divergence including the `+ da*UV` energy-conserving term, `* elem_area / helem(nz)`
- [x] write the Block C net: **uniform `u` with NONZERO `w` => vertical contribution is zero to round-off, including at `nz = nlevels(e)-1` on sloping elements**. (The originally planned `w == 0` test is blind to the bottom-face bug, since every term is proportional to `w`; keep it only as a cheap smoke check.)
- [x] write a test for the T10 property: `UV_rhsAB == 0` for `nz >= nlevels(elem)` (no momentum leaking into dry cells)
- [x] run tests — must pass before task 6

### Task 6: Add the `u . [(f+zeta) x u] == 0` invariant and the gate configuration

**Files:**
- Modify: `src/drivers/fesom_conserve.F90`
- Modify: `tools/run_conserve_pi.sh`

- [x] add a `FESOM3_MOMADV_OPT` env switch to `fesom_conserve.F90` following the existing `FESOM3_REDI` / `FESOM3_MIX_TKE` pattern
- [x] add the per-step **global KE budget** invariant `sum_elem u.(adv tendency)*volume`, relative to total KE, beside the existing heat/salt/volume checks
- [x] add the cheap `max |u . [(f+zeta) x u]|` sign assertion too (trivially zero, but free and catches a transposed sign)
- [x] add `momadv1 FESOM3_MOMADV_OPT=1` and `momadv1+TKE FESOM3_MOMADV_OPT=1 FESOM3_MIX_TKE=1` to the gate's config heredoc
- [x] write both checks so they report their magnitude even when passing
- [x] run `bash tools/run_conserve_pi.sh` — all configurations including the two new ones must pass

### Task 7: Verify acceptance criteria
- [x] verify all requirements from Overview are implemented
- [x] verify `momadv_opt==2` is bit-identical: instrument once to compare `UV_rhsAB` before/after the change, confirm 0 mismatches, then remove the probe
- [x] verify edge cases: single-layer column, bathymetry step (the one-sided vorticity ranges), `w == 0`
- [x] run full test suite: `cd build_intel_dp && ctest`
- [x] run `bash tools/run_conserve_pi.sh` — all configurations green

### Task 8: [Final] Update documentation
- [x] document `momadv_opt` values and the `FESOM3_MOMADV_OPT` switch in `docs/HANDOFF.md`
- [x] note in `docs/LESSONS.md` that the vector-invariant form has no FESOM2 oracle (the v2.7.3 branch aborts), so its net is analytic + the no-work invariant
- [x] update `CLAUDE.md` if new patterns were discovered
- [x] move this plan to `docs/plans/completed/`

## Post-Completion

*Items requiring manual intervention or external systems — no checkboxes, informational only*

**Manual verification:**
- A pi or core2 run with `momadv_opt=1` vs `momadv_opt=2`, comparing kinetic energy, SSH
  variance and eddy activity. The tests prove the operator is implemented *consistently*;
  only a run shows whether it is *better*. This is the user's call.
- Watch the `u . [(f+zeta) x u]` magnitude over a long run — a slow growth would indicate
  the ALE thickness entering the vertical divergence twice is not fully consistent.

**Open question deferred:**
- `v_inv_mom_adv_SE` (the split-explicit variant in `qq`) is not ported, since
  `use_ssh_se_subcycl=.false.` here. It would be needed if split-explicit subcycling is
  ever enabled together with `momadv_opt==1`.

**Production switch:**
- No driver sets `momadv_opt = 1`; all nine pin `2`. Enabling it in
  `fesom_lifecycle_native_mr.F90` (and exposing `FESOM3_MOMADV_OPT` there, as done for the
  spline switches) is a separate, deliberate step.
