# MUSCL horizontal advection: on-the-fly up/downwind gradients (drop `edge_up_dn_grad`)

## Overview

Replace the stored per-edge gradient array `twork%edge_up_dn_grad(4, nl-1, nEdgeO)` by
on-the-fly evaluation inside the MUSCL kernels, following the design of the reference RK3
code (`/home/a/a270029/qq/oce_stepRK3.F90`, `t_hor_adv_muscl_RK3` :638-723): keep one
elemental gradient `tr_xy` (full halo) and look up the up/downwind triangle's gradient per
edge when the flux is formed.

Unlike the reference, the result must stay **bit-identical** to the current scheme: our
`fill_up_dn_grad` uses the Miura node-averaged gradient on non-shared levels and on edges
without both up/downwind triangles (the reference uses a zero increment there). That average
depends only on (node, level), so it is computed once per node into `gnod(2, nl-1, nNodL)`
instead of ~6 times per node (once per edge).

Benefits: one fewer 4-component edge array (memory + one full write/read sweep per tracer
per step), node averaging once per node, one fewer elemental-gradient computation and two
fewer allocate/deallocate per tracer per step (diffusion reuses the same `tr_xy`).

## Context (from discovery)

- Producer today: `init_tracers_AB` (`src/oce/oce_tracer_mod.F90:101-111`): local `tr_xy`,
  `tracer_gradient_elements(values)` (owned), `exchange_elem_full`, `fill_up_dn_grad`.
- `fill_up_dn_grad` (`src/oce/oce_muscl_adv.F90:252-367`): shared levels
  `[nzmin, nzmax)`, `nzmin = maxval(ulevels_nod2D_max(ednodes))`,
  `nzmax = minval(nlevels_nod2D_min(ednodes))` -> `tr_xy` of `edge_up_dn_tri(1|2)`;
  node-k-only levels `[ulevels_nod2D(nk), nzmin)` and `[nzmax, nlevels_nod2D(nk))` ->
  Miura average around node k (components 1,3 for node 1; 2,4 for node 2); edges with a
  missing up/dn triangle -> Miura average over `[ulevels_nod2D(nk), nlevels_nod2D(nk))`;
  entries never written stay 0 (array zeroed at allocation in
  `find_up_downwind_triangles`).
- Consumers: `adv_tra_hor_muscl` / `adv_tra_hor_mfct` (`src/oce/oce_adv_tra_hor.F90:113-302`),
  via `do_oce_adv_tra` (`src/oce/oce_adv_tra_driver.F90:91,164,166`). Our FCT does NOT reuse
  the array (local `AUX`, `oce_adv_tra_fct.F90:23-29`).
- `solve_tracers_ale` (`src/oce/oce_ale_tracer.F90:130,155`) recomputes the same
  `tracer_gradient_elements(values)` for horizontal diffusion / Redi (advection writes only
  `del_ttf`, so `values` is unchanged).
- `t_tracer_work` declares/serializes `edge_up_dn_grad` (`src/types/mod_tracer.F90:41,120,140`).
  Legacy dump drivers `fesom_advhordump(_mr)` write it.
- `edge_dxdy` already carries `r_earth*mean(elem_cos)` (R7 fold, `test_bottom` T7).
- `edge_up_dn_grad` is `real(MP)`, `MP = max(WP,4)`: identical to WP in DP/SP builds.

## Development Approach

- **testing approach**: TDD; every step must be bit-identical to the current scheme
- the gate `tools/run_conserve_pi.sh` must stay GATE OK with all drift rows bit-identical
- do not copy from the reference: its zero increment at boundaries, its inline metric
  (`dxdy*r_earth*cos`, already folded in our `edge_dxdy`), its tendency-instead-of-flux
  output and `area(nz,n)` divisor, its lack of cavities/`nboundary_lay`

## Technical Details

Accessor (per edge, per side k = 1 upwind/node 1, k = 2 downwind/node 2):

```
if (both up/dn triangles exist .and. nzmin <= nz < nzmax)  g = tr_xy(:, nz, edge_up_dn_tri(k, edge))
else if (ulevels_nod2D(nk) <= nz < nlevels_nod2D(nk))       g = gnod(:, nz, nk)
else                                                         g = 0
```
(with both triangles present the node-only ranges are exactly the complement of the shared
range inside the node's wet range, so the same rule covers both fill branches).

`gnod(:,nz,n) = sum_{elem around n, wet at nz} tr_xy(:,nz,elem)*elem_area(elem) / sum elem_area`
with the fill's loop order and wet test `.not. (nlevels(elem)-1 < nz .or. nz < ulevels(elem))`,
for `nz in [ulevels_nod2D(n), nlevels_nod2D(n))`, over `n = 1..nNodL`.

## Implementation Steps

### Task 1: `muscl_node_grad` + equivalence test of the node averages

**Files:**
- Modify: `src/oce/oce_muscl_adv.F90`
- Create: `test/test_muscl_onthefly.F90`
- Modify: `test/CMakeLists.txt`

- [x] write `test_muscl_onthefly` part G: pi mesh np 1/2, a smooth+noisy tracer, elemental
  gradient + full exchange, run `fill_up_dn_grad` (oracle) and `muscl_node_grad`; for every
  owned edge/level where the fill wrote a node average, assert bitwise equality with `gnod`
  of the corresponding node; also a run with synthesised cavity columns
- [x] implement `muscl_node_grad(gnod, tr_xy, mesh, partit)`
- [x] run the test np 1/2 + full ctest

### Task 2: kernels with on-the-fly gradients

**Files:** `src/oce/oce_adv_tra_hor.F90`, `test/test_muscl_onthefly.F90`

- [x] part F: old kernels (fill + `edge_up_dn_grad`) vs new kernels (`tr_xy`, `gnod`,
  `edge_up_dn_tri`): `adv_tra_hor_muscl` and `_mfct` fluxes bitwise equal over all owned
  edges/levels, both velocity signs; positive control: `gnod = 0` (reference behaviour) differs
- [x] new kernel variants; per edge resolve the level ranges once outside the `nz` loop
- [x] tests np 1/2 + full ctest

### Task 3: switch the production path, reuse `tr_xy` for diffusion

**Files:** `src/types/mod_tracer.F90`, `src/oce/oce_tracer_mod.F90`,
`src/oce/oce_adv_tra_driver.F90`, `src/oce/oce_ale_tracer.F90`

- [x] persistent `twork%tr_xy`, `twork%gnod`; `init_tracers_AB` computes them (no fill)
- [x] driver passes them to the new kernels; `solve_tracers_ale` reuses `twork%tr_xy`
- [x] full ctest + gate: all drift rows bit-identical to the previous gate log

### Task 4: remove the stored array

**Files:** `src/types/mod_tracer.F90`, `src/oce/oce_muscl_adv.F90`, `src/oce/oce_adv_tra_hor.F90`,
`src/drivers/fesom_advhordump*.F90`, `test/test_types.F90`, `test/test_muscl_onthefly.F90`

- [x] drop `edge_up_dn_grad` (declaration, allocation, serialization), the old kernel variants;
  `fill_up_dn_grad` survives only inside the test as the oracle
- [x] dump drivers: drop the field or rebuild it via the oracle
- [x] full ctest + gate (bit-identical)

### Task 5: performance + docs

- [x] timing (deviation: no core2 job submitted -- runs are the user's; instead the test
  binary's part T on the core2 mesh on the login node, one MFCT call incl. gradient
  preparation, old vs new path): first version ~20 % slower (np 4 299 vs 246 ms); after
  (a) one branch per level instead of a lookup routine / column copy and (b) `gnod` only on
  the levels an owned edge reads (`muscl_node_ranges`): np 4 198 vs 218 ms (3 repeats),
  np 1 823 vs 884 ms -- ~9 % faster, plus one saved `tracer_gradient_elements` per tracer;
  memory -443 MiB total (~1 MiB/rank at 512). All still bit-identical (test parts G/F, gate).
- [x] HANDOFF section "MUSCL horizontal tracer advection: on-the-fly up/downwind gradients",
  LESSONS L59; plan moved to `docs/plans/completed/` (at the end)

## Post-Completion

- Production run decisions remain the user's. Gradient of `ttfAB` instead of `values`, or the
  reference's zero increment at boundaries, are possible later scheme changes (not refactors).

## Follow-up (2026-10-06): the reference RK3 rule

The user asked for the reference implementation proper ("nearly same implementation" as
`qq/oce_stepRK3.F90`): the up/downwind triangle's `tr_xy` where that triangle is wet, a zero
increment otherwise. `gnod`, `muscl_node_grad`, `muscl_node_ranges` removed; the kernels take
`(tr_xy, edge_up_dn_tri)`. **Results change** at coasts and bathymetry/cavity steps (core2:
0.8 M of 10.9 M edge-levels, max |Δflux| 0.4 % of max |flux|); identical to FESOM2 on the
shared levels. `test_muscl_onthefly` rewritten: part R (production == reference rule,
bitwise), part S (== FESOM2 on shared levels bitwise, differs elsewhere), part T (timing).
core2 timing: np 4 42 vs 74 ms, np 1 180 vs 316 ms (~43 % faster than FESOM2's path).
Gate: conservation OK; drift rows no longer bit-identical to the FESOM2-rule log (expected).

