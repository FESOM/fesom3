# FESOM3 bottom at vertices (scalar-point bottom)

> **Revision 2** — after plan review. Five claims in revision 1 were wrong and are corrected
> below; see "Corrections from review" for what changed and why. The scope of Task 4 shrank
> substantially: the `nlevels_nod2D_min` / `ulevels_nod2D_max` arrays are **kept**, not deleted.

## Overview

Move the vertical bottom representation from an **element-defined** bottom to a
**vertex-defined** bottom tied to the scalar control volumes, per the design note
*"Bottom implementation for FESOM3"* (10 September 2026) and the agent brief
`FESOM3_BOTTOM_CODEX_SYSTEM.md`.

In FESOM2 the bottom depth lives at elements while the elevation lives at vertices, so a
scalar cell has its bottom at several levels. That complicates surface/bottom exchange
processes (ice sheets, sediment resuspension). After this change the vertex column is
authoritative, an element is wet only where **all three** of its vertices are wet, and
velocities touching topography are excluded by construction.

**This is a hard switch-over.** FESOM3 becomes vertex-bottom; there is no runtime flag.
`elvls.out` is no longer read. The **oracle-dependent** byte-gates are retired. The
**FESOM_F-internal** gates are not — see Testing Strategy.

### Key benefit

A scalar cell is now a straight prism: one bottom level, full horizontal area at every
wet layer. Bottom drag lands at a single well-defined level per element, and the stiffness
vertical integration has one unambiguous interval.

### The load-bearing discovery

The repository **already has the note's data model**, under FESOM2 names. Nothing needs
renaming and no kernel loop bounds change:

| design note | existing code | today | after |
|---|---|---|---|
| `tlayer(v)` | `mesh%ulevels_nod2D` | read (all 1, no cavity) | unchanged, authoritative |
| `blayer(v)` | `mesh%nlevels_nod2D` | read from `nlvls.out` | unchanged, **authoritative** |
| `tlayer_elem(e)` | `mesh%ulevels` | hard-set to 1 | `maxval` over element nodes |
| `blayer_elem(e)` | `mesh%nlevels` | read from `elvls.out` | **`minval` over element nodes** |
| `hnode(nz,v)` | `mesh%hnode` | exists (ALE) | unchanged |
| `helem(nz,e)` | `mesh%helem` | exists, `sum(hnode)/3` | unchanged (bound fixed, Task 4) |
| `edge_len(e)` | — | — | **new** |

Every velocity kernel already loops `nz = ulevels(e) .. nlevels(e)-1`, and bottom drag is
already applied at `nzmax-1` where `nzmax = nlevels(elem)`
(`src/oce/oce_dyn_ivertvisc.F90:177`). So requirements **R3, R5, R11 and R12 need no
kernel edits** — they follow from inverting the producer. This was independently verified
in review across `oce_dyn_ivertvisc`, `oce_ale`, `oce_ssh_rhs`, `oce_adv_tra_*`,
`oce_dyn_visc`, `oce_dyn_velrhs`, `oce_mixing_kpp/tke`, `oce_pgf`, `oce_mo_conv`,
`src/ice/`, `src/io/` and `src/drivers/`.

**The risk is not loop bounds — it is inverted-direction assumptions.** Four sites assume
`nlevels_nod2D(n)` is the **max** over adjacent elements, which is exactly what flips.
They are handled in Tasks 3 and 4 and are the real content of this plan.

## Corrections from review

Revision 1 asserted five things that are false. They are corrected throughout; recorded
here so the errors are not reintroduced.

1. **`fesom_analytic` is not a regression net.** `model_step` is
   `model%nsteps_done = model%nsteps_done + 1` (`src/step/mod_model.F90:91-97`). The driver
   runs no physics and touches no `area`, `nlevels` or kernel. Revision 1 called its
   bit-identity "the single most valuable free check in the plan" and used it as the pass
   gate for five tasks. It is worth nothing. **Fix:** the conservation gate moves to Task 2
   so a real numerical baseline exists before anything changes.
2. **`nlevels_nod2D_min` and `ulevels_nod2D_max` are not "pure aliases".** On pi,
   `nlevels_nod2D_min(n) - nlevels_nod2D(n)` has mean **-3.99**, min **-30**, and is
   negative at **88.4%** of nodes today (mean -3.37 after the inversion). Substituting is a
   multi-level numerical change at every site, not a rename. **Fix:** the arrays are
   **kept**; only the two zstar sites move, each justified independently.
3. **`oce_muscl_adv.F90:274-275` is a wetness guard, not the R4 edge interval.** It bounds
   an *unguarded* read at `:303-306`:
   `edge_up_dn_grad(1:2,nz,edge) = tr_xy(1, nz, edge_up_dn_tri(:,edge))` — with no wetness
   test on `edge_up_dn_tri`, which is chosen purely geometrically. `tr_xy` is allocated
   fresh per call (`oce_tracer_mod.F90:108`) and `tracer_gradient_elements` is `intent(out)`
   writing only `ulevels(elem)..nlevels(elem)-1` (`oce_tracer_grad.F90:62-70`), so
   below-bottom entries are **uninitialized heap**. Revision 1's substitution would have
   extended that read past the up/downwind triangles' bottoms on ~85% of pi's interior
   edges. **Fix:** leave the site alone.
4. **`area(nlevels_nod2D(n),n) == 0` is not load-bearing.** `cal_shortwave_rad` zeroes
   `sw_3d` over the whole column (`oce_shortwave_pene.F90:51-55`) and forces
   `sw_3d(nzmax,n) = 0` unconditionally (`:77-81`). The two other bottom-interface consumers
   hard-zero their own flux (`oce_adv_tra_ver.F90:68-69`, `:120-121`). **Fix:** keep the
   zero — it is correct and free — but drop the "load-bearing" framing, and do not write it
   into LESSONS.md as a trap that does not exist.
5. **`areasvol` cannot reach zero, so that was the wrong reason to order the tasks.** The
   plan's own zero-stagnant-cell invariant guarantees some adjacent `e*` has
   `nlevels(e*) == nlevels_nod2D(n)`, so `area(nlevels_nod2D(n)-1, n) >= elem_area(e*)/3 > 0`
   (and `areasvol_inv` is guarded at `mod_mesh_areas.F90:379-383` anyway). **Fix:** the
   ordering stands but the rationale is corrected, and the invariant is promoted from an
   audit statistic to a runtime assertion — because it is what keeps three **unguarded**
   divides alive: `oce_ale.F90:88` (`tx/tvol`), `oce_ale.F90:377` (`Wvel/area`),
   `oce_pressure_bv.F90:310` (`1/(3*vol)`).

Two things revision 1 missed entirely, both now tasks:

6. **`helem != sum(hnode)/3` at the element bottom layer under zstar** — issue detail in
   Task 4.
7. **A fifth consumer computed inline**, invisible to a name grep: `oce_fer_gm.F90:82-87`
   builds `nlevels_nod2D_min`/`ulevels_nod2D_max` by hand via `nod_in_elem2D` (deliberately
   not `minval`, per the comment at `:80`).

## Context (from discovery)

- **Project**: FESOM_F — a clean-architecture Fortran reimplementation of FESOM2
  (ocean + sea ice), normally validated by bit-identity against a FESOM2 oracle.
  Build: CMake + Intel/oneAPI on Levante. See `README.md`, `IMPLEMENTATION.md`, `TESTING.md`.
- **Files involved**: `src/types/mod_mesh.F90`, `src/mesh/mod_mesh_read.F90`,
  `src/mesh/mod_mesh_areas.F90`, `src/mesh/mod_mesh_analytic.F90`, `src/oce/oce_ale.F90`,
  `src/oce/oce_adv_tra_hor.F90`, `src/oce/oce_ale_tracer.F90`, `src/infra/mod_geom_dump.F90`,
  `test/`, `tools/`.
- **Patterns observed**: drivers are deliberately duplicated (32 of them, each with its own
  copy-pasted init block) — the repo prefers duplication over shared init helpers. Task 2
  follows that convention rather than refactoring.
- **Dependencies**: `nlvls.out` (kept), `elvls.out` (dropped), `aux3d.out` (unchanged).
  No mesh file format change.

### Settled design decisions (do not revisit)

1. **Hard switch-over.** No `mesh_bottom` runtime knob.
2. **`blayer(v) = nlvls.out` as-is.** Verified: `nlvls.out` is *exactly* `max` over
   adjacent `elvls.out` on both meshes (0 mismatches / 3140 pi nodes, 0 / 126858 core2).
3. **Keep existing names.** No `edge_tri` → `edge_elem` sweep. Only `edge_len` is new.
4. **Keep level indexing.** `nlevels(e)` stays a *level count*; layers run
   `ulevels(e) .. nlevels(e)-1`.
5. **Scope**: R1–R6, R11, R12 + R7 geometry. **Deferred**: R8/R9/R10, R13.

### Why `max` and not `min` — measured

The element reduction is fixed at `min` by the physics (velocity must vanish at a land
corner), so the two options compose differently: `blayer = max` gives `min ∘ max`, a
morphological **closing** (dilation and erosion cancel); `blayer = min` gives `min ∘ min`,
a **double erosion** with nothing to undo it.

```
core2 (126858 nodes, 244659 elements, nl=48)

A) blayer = nlvls  (= max over adjacent elvls)     <-- CHOSEN
   elem bottom vs elvls : mean +0.067   range [0, +17]   4.3% changed (10465)
   ocean volume         : +0.38 %   (elem_area-weighted)
   stagnant bottom cells: 0 nodes
   node cols vs nlvls   : 0 shallower, 0 deeper

B) blayer = min over adjacent elvls                <-- rejected
   elem bottom vs elvls : mean -1.973   range [-26, 0]   66.6% changed
   ocean volume         : -9.27 %
   stagnant bottom cells: 3966 nodes (3.13 %)
   node cols vs nlvls   : 80570 shallower (64 %), 0 deeper
```

Option A also carries a **provable invariant** that B lacks — every node's deepest scalar
cell has at least one adjacent element wet down to it:

> Let `e* = argmax elvls over adj(v)`, so `blayer(v) = elvls(e*)`. For every node `u` of
> `e*`, `blayer(u) = max over u's elements ≥ elvls(e*)`. Hence
> `blayer_elem(e*) = min over nodes of e* ≥ elvls(e*) = blayer(v)`. ∎

Equivalently: `maxval over e in adj(n) of nlevels(e) == nlevels_nod2D(n)`. This is **not
a quality metric** — three unguarded divides depend on it (see correction 5), so it is
asserted at runtime in Task 5 and checked in `test_bottom`.

The A deepening is concentrated at island edges and shelf breaks (Seychelles bank,
Sumatra shelf, Corsica, the Aleutians, Sulawesi) — **not** at overflow sills. 92% of
changed elements move by only +1 or +2 levels; the deep ocean (>3000 m) barely moves
(2.6% of elements, mean +1.08 levels).

## Development Approach

- **testing approach**: Regular (code first, then tests) — this is a semantic inversion of
  existing arrays, so tests are written against observed post-change behaviour and the
  numeric predictions above.
- complete each task fully before moving to the next
- **every task includes new/updated tests** for the code it changes
- **all tests pass before starting the next task**
- run `ctest` **and** `tools/run_conserve_pi.sh` after each change from Task 2 onward
- **update this plan file when scope changes during implementation**

### ⚠️ Runnability windows

| after task | 1-rank pi | multi-rank | meaning |
|---|---|---|---|
| 1 | ✅ unchanged | ✅ unchanged | documentation only |
| 2 | ✅ unchanged | ✅ unchanged | gate added, no model change — **this is the baseline** |
| 3–4 | ✅ runs | ✅ runs | new scalar-cell semantics, **old bottom** — runnable, not meaningful |
| 5 | ✅ **coherent** | ❌ halo `nlevels` stale | 1-rank is the new model |
| 6 | ✅ coherent | ✅ **coherent** | full model |
| 7 | ✅ coherent | ✅ coherent | geometry units migrated |

Tasks 3–4 must precede Task 5. **Not** because `areasvol` would hit zero — it cannot
(correction 5) — but because between them the model would combine a vertex bottom with an
element-gathered area, so `area(nz,n)` at depth would be only the surviving deep elements'
share rather than the new scalar cell's full prism area. That is silently wrong rather than
loudly broken, which is worse.

Tasks 6 and 7 must **not** be split further: Task 7 changes `edge_dxdy` to metres *and* its
two consumers in one step, because between them MUSCL is off by `r_earth*elem_cos` (~6.4e6).

## Testing Strategy

- **numerical baseline**: `tools/run_conserve_pi.sh` (Task 2), built and recorded **before**
  any behaviour change, then re-run after every subsequent task.
- **unit tests**: `test/test_bottom.F90` (new, Task 5) plus additions to `test/test_mesh.F90`.
- **kept gates** — these are FESOM_F-vs-FESOM_F and survive the switch-over:
  - `tools/run_restartroundtrip.sh` — in-process write→corrupt→read over owned+halo+eXDim,
    covers `mesh%hnode`/`hbar`, np 1 and 2.
  - `tools/run_output_gate.sh` — partition independence (dist_2 == dist_8 == 1-rank); its
    header states explicitly "No FESOM2 oracle needed".
  - restart straight-vs-resume gates.
  These are the real net for Task 6's halo `nlevels` derivation.
- **retired gates** — oracle-dependent only: `run_step_gate.sh`, `run_geom_gate*.sh`,
  `run_lifecycle_*_gate_*`, `run_advhor_gate*`, `run_pressure_gate*`, `run_ice*_gate*`.
- **`fesom_analytic` proves nothing** (correction 1). Do not use it as a gate.

## Progress Tracking

- mark completed items `[x]` immediately when done
- add newly discovered tasks with ➕ prefix
- document issues/blockers with ⚠️ prefix

## Solution Overview

1. **Invert the producer.** Derive `nlevels(e) = minval(nlevels_nod2D(elem2D_nodes))` and
   `ulevels(e) = maxval(ulevels_nod2D(...))`; stop reading `elvls.out`.
2. **Make the scalar-cell area depth-independent**, and fix the two node-averaging sites
   whose denominator silently changes meaning as a result.
3. **Widen the zstar stretch to the full column** and repair the `helem` bottom-layer bound
   that this exposes.
4. **Fold the metric factor into `edge_dxdy`** (R7) and add `edge_len` in metres.

## Technical Details

### The vertical indexing contract

To be written verbatim into the vertical block of `src/types/mod_mesh.F90`:

```
FESOM3 bottom at vertices. The VERTEX column is AUTHORITATIVE; element vertical
bounds are DERIVED from it and are never read from a mesh file.

  ulevels_nod2D(v) .. nlevels_nod2D(v)    LEVEL indices, inclusive
  layers of vertex v:   nz = ulevels_nod2D(v) .. nlevels_nod2D(v)-1
  scalar cell (nz,v) exists  <=>  nz is in that range
  hnode(nz,v) is its thickness;  area(nz,v) its horizontal area (depth-independent)

  DERIVED (mod_mesh_read: setup_vertical / read_mesh_local):
    ulevels(e) = maxval(ulevels_nod2D(elem2D_nodes(1:nnodes,e)))   ! tlayer_elem
    nlevels(e) = minval(nlevels_nod2D(elem2D_nodes(1:nnodes,e)))   ! blayer_elem+1
  layers of element e:  nz = ulevels(e) .. nlevels(e)-1
    == exactly the FULLY WET prisms. Velocity DOF outside this range are never
    assembled or updated, so velocities touching topography are zero by construction.

  RETAINED, and NOT aliases of the node column:
    nlevels_nod2D_min(n) = min over e in adj(n) of nlevels(e)   ! 2-ring min
    ulevels_nod2D_max(n) = max over e in adj(n) of ulevels(e)
  These bound work that reaches ADJACENT ELEMENTS from a node, not the node's own
  cell. Do not substitute nlevels_nod2D for them -- see oce_muscl_adv.F90:303.

  REQUIRED INVARIANT (asserted at setup; three unguarded divides depend on it):
    maxval over e in adj(n) of nlevels(e) == nlevels_nod2D(n)
  i.e. every node's deepest scalar cell has at least one wet adjacent element.
  Holds because nlvls.out == max over adjacent elvls.out. A mesh violating it
  gives NaN at oce_ale.F90:88, oce_ale.F90:377 and oce_pressure_bv.F90:310.

  Other invariants (test_bottom):
    1 <= ulevels_nod2D(v) < nlevels_nod2D(v) <= nl
    ulevels(e) < nlevels(e)
    zbar_e_bot(e) == zbar(nlevels(e))
    helem(nz,e) == sum(hnode(nz,elnodes))/3  for nz = ulevels(e)..nlevels(e)-1
```

Naming note: `tlayer/blayer` ARE `ulevels_nod2D/nlevels_nod2D`; `tlayer_elem/blayer_elem`
ARE `ulevels/nlevels`. The note's `ulayer_edge` (brief ambiguity #2) is a *bottom* bound
despite the `u`; it is never stored, so the inconsistency does not propagate.

### Multi-rank `nlevels` derivation — global array, not an exchange

At `npes > 1`, `mesh%elem2D_nodes` is populated for **owned elements only**
(`src/mesh/mod_mesh_read.F90:280-287`), but `nlevels` must be valid across the full element
halo (`nElemF`) — `oce_adv_tra_hor.F90` and `vert_vel_ale` both read `nlevels(el(2))` where
`el(2)` can be a halo element.

**Chosen: global-array derivation, no MPI.** Read `nlvls.out` into a global temp *first*,
then inside the existing `elem2d.out` scatter loop — which already has the global node ids
`gn1,gn2,gn3` in hand — set, for every `lid > 0` (owned **and** halo):

```fortran
mesh%nlevels(lid) = min(nlvls_g(gn1), nlvls_g(gn2), nlvls_g(gn3))
```

The `nNodG` integer temp is the same order as the `imap_nod` full-size inverse map already
allocated there. Deterministic, identical on every rank, and it *removes* an MPI dependency
rather than adding one. Rejected: `exchange_elem_full_2D_i`, which exists but adds a
collective to a path that does not need one.

**Allocation order matters** and revision 1 omitted it: `mesh%nlevels`/`mesh%nlevels_nod2D`
are allocated at `:318-319`, *after* the `elem2d.out` loop at `:276-288`, and `mesh%ulevels`
even later in `setup_vertical_local` at `:450`. All three allocations must move ahead of the
loop, and the node-side scatter at `:325-329` must still populate `nlevels_nod2D` from the
new global temp.

### Depth-independent scalar-cell area

`compute_node_areas` (`src/mesh/mod_mesh_areas.F90:347-359`) currently builds `area(nz,n)`
by gathering `elem_area/3` from adjacent elements wet at layer `nz`. Under a vertex bottom
the cell is a straight prism, so:

```fortran
do n = 1, nNodO
   A = 0
   do j = 1, nod_in_elem2D_num(n)
      A = A + elem_area(nod_in_elem2D(j,n)) / 3.0_MP     ! UNSCALED, as today
   end do
   do nz = ulevels_nod2D(n), nlevels_nod2D(n)-1
      area(nz,n) = A
   end do
end do
```

> **SUPERSEDED during implementation.** The array is **1-D**, exactly as the note writes it
> (`area(1:myDim+eDim)`). The reasoning below for keeping it 2-D was wrong in a way that
> mattered: `area(nz,n)` still *reads* as level-dependent at every call site, and that is
> precisely how a wrong denominator hid — nine node-averaging sites silently split into two
> groups that no longer agreed, and nothing failed. 1-D makes the level index unwritable.
> The bottom-interface zero it protected turned out not to be load-bearing. See the final
> commit and IMPLEMENTATION.md §13.

~~**The array stays 2-D**~~, even though the note writes `area(1:myDim+eDim)`. Keeping the
`nz = nlevels_nod2D(n)` entry at zero costs nothing, matches what the current accumulation
produces, and keeps the "closed bottom" idiom for any consumer added later. It is *not*
load-bearing today (correction 4).

Every explicit `area(nz)/areasvol(nz)` ratio in `oce_ale_tracer.F90:556-602, 630-652`
becomes exactly `1.0` in the interior and `0` at the bottom. **Leave those untouched** —
simplifying them is unrelated churn and loses the FESOM2 line correspondence.

#### Two implicit ratios that change meaning — DECIDED

Both accumulate `elem_area`-weighted sums over **wet** adjacent elements and then divide by
`areasvol`, which is exactly the wet area today and becomes the **full prism** area after
this change. Confirmed by the user on 2026-09-10:

| site | what it is | decision |
|---|---|---|
| `oce_ale_tracer.F90:440-448` — `Tx/3.0_WP/areasvol(nz,n)` | a node-**average** of element tracer gradients | ⚠️ **THIS DECISION WAS WRONG AND WAS REVERTED.** Renormalising by the wet area inflates the Redi flux by `full/wet` — up to **7.04×** at 10.4 % of pi node-levels — because the flux is `tr_xynodes * area`, which with the `areasvol` denominator collapses correctly to `sum over wet elements of tr_xy*(elem_area/3)`. A dry adjacent element is not missing data: its flux is genuinely ZERO and it must carry its full area weight. The original expression is correct. |
| `oce_dyn_velrhs.F90:213-217` + `:299-300` | a **flux divergence** over the control volume (`sum(UV*elem_area)*W`, then `*areasvol_inv`) | **leave alone** — dividing by the full CV area is correct under the new scheme |

`tr_xynodes` feeds Redi, which is on in the target production config, so this is not
hypothetical — it is the item to watch in the first core2 run.

`elem_area` itself is unchanged (R6).

### The zstar stretch, and the `helem` bound it exposes

Only two sites move off `nlevels_nod2D_min`, both in `oce_ale.F90`, both justified by
depth-independent area — the code's own comment at `:394` already says the range is meant
to be "where `area(nz)=area(1)`", which is now the whole column:

- `:396` (zstar `Wvel` stretch): `nlevels_nod2D_min(n)-1` → `nlevels_nod2D(n)-1`
- `:534` (zstar `hnode` commit): `nlevels_nod2D_min(n)-2` → `nlevels_nod2D(n)-2`

**This exposes a latent inconsistency.** The element `helem` rebuild in the same routine
(`:544-552`) loops `nz = nzmin, nzmax-1` with `nzmax = nlevels(elem)-1`, i.e. it stops at
`nlevels(elem)-2` and never rewrites the element's deepest layer. Today that is safe because
`nlevels_nod2D_min(n) <= nlevels(e)` for every `e ∋ n`, so no node ever stretches layer
`nlevels(e)-1`. After the inversion the inequality flips to `nlevels(e) <= nlevels_nod2D(n)`,
so layer `nlevels(e)-1` **is** stretched at the element's deeper nodes while `helem` stays
frozen there.

`helem` is the thickness used by `compute_hbar_ale` (`:193`/`:205`), `vert_vel_ale`
(`:330`/`:348`) and `adv_tra_hor`, so the SSH/continuity budget and the tracer volume would
stop agreeing under zstar. Fix: extend the element loop to `nz = nzmin, nzmax`, and assert
`helem(nz,e) == sum(hnode(nz,elnodes))/3` over the full element range.

### Sites that keep `nlevels_nod2D_min` / `ulevels_nod2D_max` — and why

| site | why it stays |
|---|---|
| `oce_muscl_adv.F90:274-275` | wetness guard on the unguarded `tr_xy` read at `:303-306` (correction 3). Needs a bound over *adjacent elements*, which is what this array is. |
| `oce_fer_gm.F90:231-232` | GM tridiagonal bounds |
| `oce_fer_gm.F90:82-87` | the same quantity computed **inline** via `nod_in_elem2D` — invisible to a name grep. Must stay consistent with `:231-232`. |

After the inversion these become a **2-ring min** (`min over adj e of min over e's nodes`),
i.e. shallower than today. Consequences, both acceptable and both to be recorded:

- MUSCL's "shared levels" range shrinks, pushing more of the deep column into the
  not-shared branches — which are correctly wetness-guarded (`:296`, `:313`), so this is a
  mild accuracy change, not a bug. A tighter guard using
  `min(nlevels(edge_up_dn_tri(:,edge)))` is the natural follow-up.
- GM's column shrinks, but `:82-87` and `:231-232` shrink **together**, so `fer_K`/`fer_c`
  and `fer_solve_Gamma` stay on the same column. Moving both to the node column is a
  follow-up to consider once the bottom change is validated — not part of this patch.

### R7 — physical `edge_dxdy`, new `edge_len`

FESOM2 stored `edge_dxdy` in radians and applied `r_earth * mean(elem_cos)` at each point
of use. `src/oce/oce_adv_tra_hor.F90:151-158` computes exactly that mean cosine inline,
resolving the brief's ambiguity #5 from existing code rather than by guesswork.

Producer, `compute_edge_geometry` (`src/mesh/mod_mesh_areas.F90:297`) — the two loops merge,
because the mean cosine needs `el1/el2` which only the second loop has:

```fortran
el1 = edge_tri(1,n);  el2 = edge_tri(2,n)
cosm = elem_cos(el1)
if (el2 > 0) cosm = 0.5_WP*(cosm + elem_cos(el2))
edge_dxdy(1,n) = a1 * cosm * r_earth                             ! [m]
edge_dxdy(2,n) = a2 * r_earth                                    ! [m]
edge_len(n)    = sqrt(edge_dxdy(1,n)**2 + edge_dxdy(2,n)**2)     ! [m]
```

`elem_cos` is halo-valid here — `exchange_elem_cos` runs earlier in `compute_geometry`
(`:88`). No `cartesian` special case: that mode sets `elem_cos = 1` while still scaling
everything else by `r_earth`.

⚠️ **Floating-point note.** `0.5*(cosm + elem_cos(el2)) * r_earth` is not the same
expression as today's `a = 0.5*(a + r_earth*elem_cos(el2))` with `a = r_earth*elem_cos(el1)`.
The re-association is invisible on any flat/Cartesian test (`elem_cos = 1`), so no test in
this plan can catch it. Decision 1 makes this a one-way door — record it in the final report.

Consumers drop `*a` and `*r_earth` at `oce_adv_tra_hor.F90:203-208` (muscl) and `:297-302`
(mfct). The local `a` is then dead in all three routines and is deleted, including in
`adv_tra_hor_upw1` (`:64`/`:71`) where it is **already dead today**.

`edge_len` goes into the type, the serialization pair, and `mod_geom_dump`. Not into
`mod_io_meshdiag`: that store has no edge dimension, and adding one for an unconsumed array
is not worth it. `edge_len` has no kernel consumer — the note introduces the array without a
use. Say so in the report.

## What Goes Where

- **Implementation Steps** (`[ ]`): source, test and tooling changes in this repo.
- **Post-Completion** (no checkboxes): production runs, and the deferred follow-ups.

## Implementation Steps

### Task 1: Document the vertical indexing contract

**Files:**
- Modify: `src/types/mod_mesh.F90`

- [x] add the contract block (Technical Details) above the `! ---- vertical structure ----`
      declarations, including the `tlayer/blayer` mapping and the `ulayer_edge` note
- [x] mark `nlevels`/`ulevels` as DERIVED, never read from file
- [x] document `nlevels_nod2D_min`/`ulevels_nod2D_max` as *2-ring* bounds over adjacent
      elements, explicitly **not** aliases of the node column, with the
      `oce_muscl_adv.F90:303` cross-reference
- [x] document the required invariant and the three unguarded divides that depend on it
- [x] build: `./configure.sh --compiler intel --precision dp --build`
- [x] run tests: `cd build_intel_dp && ctest --output-on-failure`

### Task 2: Conservation + no-leakage gate (baseline before any change)

**Files:**
- Create: `src/drivers/fesom_conserve.F90`
- Create: `tools/run_conserve_pi.sh`
- Modify: `test/CMakeLists.txt`

- [x] create `fesom_conserve` from `src/drivers/fesom_lifecycle_mr.F90` (unforced, and
      handles np>=1 unlike the 1-rank `fesom_lifecycle`), dropping the `mod_dump` hooks;
      duplication is the established driver convention in this repo
- [x] per step, print total heat and salt content as
      `sum over owned n, nz of tr(nz,n)*hnode(nz,n)*areasvol(nz,n)`, `allreduce_sum` at `npes>1`
- [x] add the **T10** assertion each step: `UV(:,nz,e) == 0` for all `nz >= nlevels(e)`,
      aborting with the offending element id
- [x] add the `helem(nz,e) == sum(hnode(nz,elnodes))/3` and `zbar_e_bot(e) == zbar(nlevels(e))`
      invariants over the full element range (moved here from Task 4 so the baseline covers them)
- [x] add a finiteness sweep: abort on any non-finite value in `UV`, `w`, `hnode`, `tr`
- [x] add `FESOM3_WHICH_ALE` support and `is_nonlinfs = merge(1,0, which_ALE/='linfs')`
- [x] enforce the tolerance **in the driver** via `FESOM3_CONSERVE_TOL` rather than parsing
      output in shell — makes the ctest registration trivial
- [x] `tools/run_conserve_pi.sh`: 20 steps on pi, zstar at np 1 and 2 plus linfs at np 1
- [x] register as a ctest at np 1 and np 2 (zstar only — see the finding below)
- [x] **run on unmodified code and record the baseline**

#### ⚠️ Finding: linfs is NOT conservative, so it cannot be gated on drift

Revision 2 assumed both ALE modes would conserve to round-off. Measured on pi, 20 steps
from the phc3.0 IC, before any bottom change:

```
zstar  np=1   heat -4.8721936444146653E-15   salt -1.6556314290872313E-14
zstar  np=2   heat  0.0000000000000000E+00   salt -7.8024008163121374E-15
linfs  np=1   heat -2.2486275162863836E-04   salt -1.1970160309106803E-06
```

`linfs` drift is four orders of magnitude too large to be round-off, and it is
**non-monotone** (heat falls to step 16 then recovers) — a free-surface adjustment
transient, not a leak. The cause is structural: the linear free surface freezes `hnode`,
so the surface vertical advective flux `-w*T*area` at `nzmin`
(`oce_adv_tra_ver.F90:66`) is a real source/sink with no thickness change to balance it.
That is the textbook reason `zlevel`/`zstar` exist, and it is not something this patch can
or should fix.

Consequences for the plan:

- **zstar is the conservation gate**, at `TOL=1e-12` (about two orders of headroom over the
  measured baseline). It must stay green through every remaining task — conservation has to
  hold both before and after the bottom change, so this is a genuine cross-task invariant
  and does not need a stored reference value.
- **linfs runs for the invariants only** (T10, `helem`, `zbar_e_bot`, finiteness). Its drift
  is reported, never gated.
- zstar is also the mode the user's production config uses, so the gate covers the path that
  matters.

Note: np=1 and np=2 totals differ by ~5e-6 relative (`4.62378995e18` vs `4.62381222e18`).
That is a partition difference in the IC interpolation, not a conservation defect — each
rank count conserves within itself.

#### Baseline result

All four invariants pass on unmodified code at both rank counts and both ALE modes.
`ctest` 23/23 (the two new gates add ~3 s).

### Task 3: Depth-independent scalar-cell area

**Files:**
- Modify: `src/mesh/mod_mesh_areas.F90`
- Modify: `src/oce/oce_ale_tracer.F90`
- Modify: `test/test_mesh.F90`

- [x] rewrite the `mesh%area` accumulation in `compute_node_areas` (`:347-359`) to the full
      median-dual area with no depth test, over `nz = ulevels_nod2D(n)..nlevels_nod2D(n)-1`
- [x] leave the `nz = nlevels_nod2D(n)` entry zero; comment that it is a deliberate closed
      bottom, **not** currently load-bearing
- [x] keep the deferred single `* r_earth**2` scaling and the `areasvol`/`area_inv`/
      `areasvol_inv` derivation exactly as they are
- [x] ⚠️ REVERTED — `tr_xynodes` keeps the `areasvol` (full-area) denominator. Renormalising
      by the wet area was a bug: it inflates the Redi isoneutral flux by `full/wet`, up to
      7.04× on pi. Pinned by test_bottom's flux-composition check.
- [x] leave `oce_dyn_velrhs.F90:213-217`/`:299-300` alone — a flux divergence over the full
      CV, correctly on the full area (this judgement held up)
- [x] leave every explicit `area(nz)/areasvol(nz)` ratio in `oce_ale_tracer.F90` untouched
- [x] add to `test_mesh.F90`: control areas `> 0`, `areasvol == area`, `areasvol_inv` finite
      and `> 0` (depth-independence became structural once the array went 1-D)
- [x] ➕ test that the Redi flux COMPOSES: `tr_xynodes*area == sum of per-element shares`.
      A value test, not a budget test — the Redi tendency is a telescoping flux divergence
      and so conserves for *any* denominator; a conservative operator launders a wrong
      coefficient.
- [x] run tests: `ctest --output-on-failure` **and** `run_conserve_pi.sh` for both ALE modes;
      compare against the Task 2 baseline

### Task 4: zstar full-column stretch + `helem` bottom-layer repair

**Files:**
- Modify: `src/oce/oce_ale.F90`

- [x] `:396`: `nlevels_nod2D_min(n)-1` → `nlevels_nod2D(n)-1`, with a comment that the
      stretch now spans the whole column because `area` is depth-independent
- [x] `:534`: `nlevels_nod2D_min(n)-2` → `nlevels_nod2D(n)-2`
- [x] `:549`: extend the `helem` rebuild to `nz = nzmin, nzmax` so the element's deepest
      layer is rewritten; comment why the old bound was safe and no longer is
- [x] **leave `oce_muscl_adv.F90` and `oce_fer_gm.F90` untouched** — see the retention table
- [x] add the `helem(nz,e) == sum(hnode(nz,elnodes))/3` invariant over
      `nz = ulevels(e)..nlevels(e)-1` to `fesom_conserve`'s per-step assertions
- [x] run tests: `ctest` **and** `run_conserve_pi.sh` with `FESOM3_WHICH_ALE=zstar` — this is
      the task that gate exists for

### Task 5: Derive `nlevels`/`ulevels` at 1 rank; stop reading `elvls.out`

**Files:**
- Modify: `src/mesh/mod_mesh_read.F90`
- Modify: `src/mesh/mod_mesh_analytic.F90`
- Create: `test/test_bottom.F90`
- Modify: `test/CMakeLists.txt`

- [x] delete the `elvls.out` read from `read_mesh` (`:95-99`); allocate `nlevels` in
      `setup_vertical` instead
- [x] derive `ulevels(e) = maxval(ulevels_nod2D(elnodes))` and
      `nlevels(e) = minval(nlevels_nod2D(elnodes))` over `elem2D_nnodes(e)` vertices, then
      `elem_depth(e) = zbar(nlevels(e))`
- [x] recompute `nlevels_nod2D_min`/`ulevels_nod2D_max` from the **derived** `nlevels`
      (they are now 2-ring bounds)
- [x] add a runtime check in `setup_vertical`: `maxval over adj(n) of nlevels(e) ==
      nlevels_nod2D(n)` for every node, `error stop` with the node id on failure
- [x] derive the same way in `mod_mesh_analytic.F90` instead of hard-setting `nlevels = nl`
- [x] create `test/test_bottom.F90` on the `test_mesh.F90` skeleton (`par_init`/`read_mesh`/
      `compute_geometry` + `check()`)
- [x] test **T2**: analytic mesh, override `nlevels_nod2D` on one triangle to `[8,6,5]`,
      re-derive, assert `nlevels(e) == 5`
- [x] test **T3**: two edge vertices with different bounds ⇒ edge interval is
      `minval(nlevels_nod2D(ednodes))`
- [x] test **T6**: on pi, `sum(hnode(:,v)) == zbar(ulevels_nod2D(v)) - zbar(nlevels_nod2D(v))`
      within tolerance; `hnode >= 0`
- [x] test **T9**: on pi, `zbar_e_bot(e) == zbar(nlevels(e))`
- [x] test the required invariant explicitly on pi (not the tautology
      `nlevels(e) <= minval(nlevels_nod2D(elnodes))`, which holds by construction)
- [x] register `add_fesom_test(test_bottom 1)`
- [x] run tests: `ctest` **and** `run_conserve_pi.sh` both ALE modes

### Task 6: Derive across the full element halo (multi-rank)

**Files:**
- Modify: `src/mesh/mod_mesh_read.F90`
- Modify: `test/test_bottom.F90`
- Modify: `test/CMakeLists.txt`

- [x] move the `mesh%nlevels`/`nlevels_nod2D` allocations (`:318-319`) and the `mesh%ulevels`
      allocation (`setup_vertical_local:450`) **ahead of** the `elem2d.out` loop at `:276-288`
- [x] move the `nlvls.out` read ahead of `elem2d.out` and load it into a global
      `nlvls_g(nNodG)` temp; keep the node-side scatter at `:325-329` populating
      `nlevels_nod2D` from that temp
- [x] delete the `elvls.out` read (`:320-323`); derive `nlevels(lid)` inside the `elem2d.out`
      loop for every `lid > 0` (owned **and** halo); deallocate the temp
- [x] derive `ulevels` the same way; set `elem_depth` from the derived `nlevels`
- [x] comment why this is a global-array derivation and not `exchange_elem_full_2D_i`
- [x] apply the same runtime invariant check on owned nodes in `setup_vertical_local`
- [x] multi-rank test: assert `nlevels(e) == min(nlvls_g of its 3 global nodes)` for **halo**
      elements (via `myList_elem2D`) — not merely `nlevels(e) > 0`
- [x] register `add_fesom_test(test_bottom 2)` and `(test_bottom 8)`
- [x] run tests: `ctest` all rank counts, `run_conserve_pi.sh` at np 1 and 2,
      `tools/run_restartroundtrip.sh`, `tools/run_output_gate.sh`

### Task 7: R7 — physical `edge_dxdy` and `edge_len` (producer + consumers together)

**Files:**
- Modify: `src/types/mod_mesh.F90`
- Modify: `src/mesh/mod_mesh_areas.F90`
- Modify: `src/oce/oce_adv_tra_hor.F90`
- Modify: `src/infra/mod_geom_dump.F90`
- Modify: `test/test_bottom.F90`

- [x] add `edge_len` to `t_mesh` with an explicit `! [m]` comment plus its
      `write_bin_array`/`read_bin_array` pair
- [x] merge the two loops in `compute_edge_geometry`; emit `edge_dxdy` in metres and
      `edge_len`; document the unit change next to both declarations
- [x] in the **same** task, drop `*a` and `*r_earth` at `oce_adv_tra_hor.F90:203-208` and
      `:297-302`, and delete the dead local `a` from all three routines (`:64`/`:71` included)
- [x] update the module header comment (`:17`) documenting the old inline convention
- [x] add `wr_r1(u, 'edge_len', ...)` to `mod_geom_dump`
- [x] test **T7**: `edge_len` matches an independent haversine on pi; `maxval(abs(edge_dxdy))`
      is metre-scale, not radian-scale; `edge_len > 0` everywhere
- [x] test: MUSCL reconstruction reproduces a linear tracer field exactly on the analytic
      mesh (catches a dropped or doubled metric factor)
- [x] run tests: `ctest` **and** `run_conserve_pi.sh` both ALE modes

### Task 8: Mesh-delta audit against the predicted numbers

**Files:**
- Create: `tools/bottom_delta.py`
- Create: `tools/run_bottom_audit.sh`

- [x] `tools/bottom_delta.py`: read a `mod_geom_dump` binary (`FGEOMDMP`; format documented
      atop `tools/geom_diff.py`) plus the mesh's `elvls.out`, and report the `nlevels` delta
      histogram, changed-element count, ocean-volume change and stagnant-cell count
- [x] make the volume weighting **explicit and `elem_area`-weighted** — an unweighted sum
      gives +1.37% on pi against the plan's +1.55%, and would trip the reconcile rule below
      on a units mismatch rather than a real disagreement
- [x] `tools/run_bottom_audit.sh`: run `fesom_geomdump` on pi and core2 and feed both to the
      script; use `/sw/spack-levante/python-3.9.9-fwvsvi/bin/python3` (the login-node
      `/usr/bin/python3` is 3.6.8 with no numpy)
- [x] verify **core2**: mean `+0.067`, range `[0,+17]`, **10465** changed, `+0.38 %` volume,
      **0** stagnant cells
- [x] verify **pi**: mean `+0.227`, range `[0,+20]`, **651** of 5839 changed, `+1.55 %`
      volume, **0** stagnant cells
- [x] verify `edge_dxdy` is metre-scale and `edge_len` is present in both dumps
- [x] record the actual numbers here; ⚠️ on disagreement, stop and reconcile

**Audit result — both meshes reproduce the prediction exactly, histogram included:**

```
pi     changed  651/5839    mean +0.227  range [0,+20]  +1.55 % volume  0 stagnant
core2  changed 10465/244659 mean +0.067  range [0,+17]  +0.38 % volume  0 stagnant
core2 histogram: +1:7773 +2:1411 +3:571 +4:286 +5:146 +6:97 +7:63 +8:41
                 +9:19 +10:19 +11:21 +12:2 +13:6 +14:4 +15:3 +16:1 +17:2
edge_len core2: 11.9 km .. 181.9 km  (R7 factor folded in)
```

Two performance notes on `bottom_delta.py`, both hit on the 250 MB core2 dump:
`read_dump` takes a field whitelist (the four `(nl, nod2D)` area arrays it never reads
cost 24M struct values), and the element node lists are sliced once per *row* rather than
once per element (`vals[i::d1]` builds a whole new list, so per-element is quadratic).
Together: >10 minutes to 0.66 s.

### Task 9: Verify acceptance criteria

- [x] vertex columns carry validated `ulevels_nod2D`/`nlevels_nod2D` and `hnode`
- [x] element intervals are intersections of their vertex wet columns; the required
      invariant is asserted at runtime on both mesh paths
- [x] no velocity contribution from partly-land prisms (T10 green in `fesom_conserve`)
- [x] `helem == sum(hnode)/3` over the full element range, under `zstar`
- [x] `elem_area` unchanged; scalar `area` depth-independent
- [x] `edge_dxdy` in metres, `edge_len` in metres
- [x] bottom drag at `nlevels(elem)-1` — confirm by inspection at
      `oce_dyn_ivertvisc.F90:177` that no edit was needed, and say why in the report
- [x] stiffness integration uses `zbar_e_bot - zbar(ulevels(e))` with the derived `nlevels`
      — confirm at `oce_ssh_rhs.F90:168`
- [x] `git diff` contains no unrelated changes (no renames, no formatting sweeps)
- [x] full suite: `ctest --output-on-failure`; `run_conserve_pi.sh` both ALE modes at np 1
      and 2; `run_restartroundtrip.sh`; `run_output_gate.sh`
- [x] GNU portability build: `./configure.sh --compiler gnu --precision dp --clean --build`
      with no new warnings

### Task 10: [Final] Update documentation

**Files:**
- Modify: `README.md`, `IMPLEMENTATION.md`, `TESTING.md`, `docs/LESSONS.md`

- [x] `README.md` §5: `elvls.out` is no longer read
- [x] `IMPLEMENTATION.md`: the bottom-at-vertices contract and the `tlayer/blayer` mapping
- [x] `TESTING.md`: split the gate catalogue into **oracle-dependent (retired)** and
      **self-consistency (kept)**; document `run_conserve_pi.sh` as the primary numerical net
- [x] `docs/LESSONS.md`: the `min ∘ max` vs `min ∘ min` asymmetry; the
      `oce_muscl_adv.F90:303` unguarded `tr_xy` read and why `nlevels_nod2D_min` must stay;
      the `helem` bottom-layer bound. **Do not** record the shortwave `area == 0` trap — it
      does not exist (correction 4).
- [x] write the brief's required final report (files changed, indexing contract,
      representation, R7 conversion incl. the FP re-association note, bottom drag and
      stiffness, tests run, unresolved ambiguity, deferred work)
- [x] note in the report that the change *repairs* several element-from-node averages that
      previously read undefined node levels (`oce_mixing_kpp.F90:217-224`,
      `oce_mixing_tke.F90:502-506`, `oce_ale.F90:550`)
- [x] move this plan to `docs/plans/completed/`

## ➕ Work done after the plan was written

Driven by review feedback during implementation, not foreseen here:

1. **`mesh%area`/`areasvol`(+`_inv`) collapsed to 1-D** and every consumer across the tree
   rewritten (~19 files). Saves 191 MB/rank on core2 and, more importantly, makes the wrong
   denominator unwritable.
2. **`compute_vel_nodes` moved to the full area.** A dry adjacent element contributes zero
   velocity carrying its full area weight. Intended consequence: a uniform velocity field no
   longer node-averages to the same value at every level — real topographic damping of the
   cell-mean velocity, which feeds the PP/KPP/TKE shear.
3. **Redi coverage.** `diff_ver_part_redi_expl` is the only caller of `tr_xynodes` and runs
   only under `if (Redi)`, which defaults `.false.` — every earlier gate had it off. The gate
   now sweeps Redi / KPP / TKE / GM+Redi+TKE / GM+KPP.
4. **Volume conservation** added as a per-step invariant, separate from tracer content:
   under `linfs` volume is exactly conserved while tracer drifts `-2.2e-04`, so neither
   check substitutes for the other.
5. **Four node-averaging sites deliberately left on the wet area** — `oce_muscl_adv`
   (`edge_up_dn_grad`), `oce_pressure_bv:516` (`sigma_xy`, feeding the GM/Redi neutral
   slope), `oce_pressure_bv:310` (`smooth_nod`). The zero-with-weight rule is stated for
   *velocities*; a dry element's gradient is undefined rather than zero, and a smoother must
   reproduce a constant field. Open question, `sigma_xy` first.

## Post-Completion

*Informational — no checkboxes.*

**Manual verification (user runs simulations himself):**
- a multi-year core2 run with `FESOM3_WHICH_ALE=zstar` and GM/Redi/TKE on. The two items
  most worth watching are the zstar stretch widening (Task 4) and the `tr_xynodes`
  denominator decision (Task 3), both of which touch Redi.
- an overflow-sensitive comparison if the deepened shelf-break elements turn out to matter;
  Task 8 lists exactly which elements moved.

**Deferred follow-ups:**
- **Tighter MUSCL guard** — replace `nlevels_nod2D_min` at `oce_muscl_adv.F90:274-275` with
  `min(nlevels(edge_up_dn_tri(:,edge)))`, the exact bound for the read at `:303-306`.
  Strictly better than the current conservative bound; deliberately not bundled here.
- **GM column** — consider moving `oce_fer_gm.F90:82-87` and `:231-232` together to the node
  column once the bottom change is validated.
- **R8/R9/R10** — `edge_elem(:,e) = [e,e]`, `edge_tri` → `edge_elem`, element self-neighbours.
  Notes: 13 `if (el(2) > 0)` sites; most are already numerically safe because
  `edge_cross_dxdy(3:4) = 0` at boundary edges makes the duplicate pass contribute exactly
  zero (traced through `adv_tra_hor` regions A–E and `init_stiff_mat_ale`'s `fy`).
  `oce_muscl_adv.F90:57` needs a boundary *flag*, and the repo has one: `ed > edge2D_in`
  (`oce_dyn_visc.F90:127`, `mod_ice_dyn.F90:415`, `mod_ice_setup.F90:67`). `elem_neighbors`
  is **allocated but never populated** — R10 needs a producer written from scratch.
- **R13** — horizontal diffusion (unmodified by decision), Redi (the note says adjustments
  are needed but does not specify them; do not guess), boundary scalar-cell area inflation.
- **`edge_len` has no consumer.** Populated and dumped, read by nothing.
