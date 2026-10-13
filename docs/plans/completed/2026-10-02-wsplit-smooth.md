# Smooth Courant-number-dependent explicit/implicit vertical advection (`use_wsplit`)

## Overview

Port FESOM2's `w_impl` option (`use_wsplit`: split `w = w_e + w_i`, advect `w_e` explicitly
and `w_i` implicitly) and replace its hard CFL switch by the smooth, C¹ limiting function of
Shchepetkin (2015, Ocean Modelling 91, 38-69, Sec. 3.1 / Fig. 9). The smooth function is the
ONLY split: there is no "FESOM2 mode" (decision 2026-10-02).

Problem solved: at high vertical Courant numbers (thin z* surface layers, cavities,
strong convection, fine vertical grids) explicit vertical advection is unstable. FESOM2 caps
the explicit Courant number at `wsplit_maxcfl` and sends the excess to an implicit upstream
solve, but `d(w_e)/d(CFL)` jumps from 1 to 0 at the threshold, so cells oscillating around it
flip between high-order explicit and partly first-order implicit treatment. The Shchepetkin
function makes the explicit share a C¹ function of the Courant number: identity below
`Cu_min`, a smooth bend, exact saturation at `Cu_max` above `Cu_cut = 2·Cu_max − Cu_min`.

Integration: the split lives where FESOM2 has it (`compute_Wvel_split` at the end of
`vert_vel_ale`); all consumers of `w_e`/`w_i` are already in place EXCEPT the FCT tracer path
(`adv_tra_vert_impl`, currently an `error stop`), which this plan ports. Nothing else in the
discretization changes.

Reviewed 2026-10-02 (plan-review agent): numerical core confirmed; test tolerances, the
smoothness test design, the `fesom_pressuredump` dependency and several coverage gaps
corrected below.

## Context (from discovery)

- `src/oce/oce_ale.F90`: `compute_CFLz` (443-483, FESOM2 :3126-3213) builds
  `CFL_z(nz,n) = |w(nz)|·dt·(1/h(nz−1) + 1/h(nz))` — the interface flux counted against BOTH
  adjacent cells (surface interface: one term). `compute_Wvel_split` (489-531, FESOM2
  :3217-3265): the hard split `dd=(CFL−C)/C`, `w_e = w/(1+dd)`, `w_i = w·dd/(1+dd)` for
  `CFL_z > wsplit_maxcfl`. Called at `vert_vel_ale` :438-439 after the `exchange_nod` of
  `Wvel`/`hnode_new`, over owned+halo nodes.
- Consumers of the split, state in FESOM3:
  - momentum TDMA `impl_vert_visc_ale`: `w_i` upwind flux form (+ advective-form correction
    for `momadv_opt==1`, commit e4d2cb6) — DONE
  - momentum advection (`momentum_adv_scalar`, `momentum_adv_vinv`): `w_e` — DONE
  - non-FCT tracers: `w_i` in the vertical-diffusion TDMA (`do_wimpl`,
    `src/oce/oce_ale_tracer.F90:538, 575-618`, f3 area rule) — DONE
  - FCT tracers: `adv_tra_vert_impl` on `fct_LO` + LO flux recomputed with the FULL `w`
    (FESOM2 `oce_adv_tra_driver.F90:282-292`, `oce_adv_tra_ver.F90:90-240`) — MISSING:
    `src/oce/oce_adv_tra_driver.F90:139-142` error-stops. The one real port.
- FESOM2 2.8.0 (fetched) is identical to 2.7.3 in all of the above (only PR #836, a cavity
  NaN clean-up, is nearby). NEMO `sshwzv.F90` `wAimp` implements the same Shchepetkin
  function (its constants 0.15/0.30 belong to NEMO's Courant definition and time stepping).
- Parameters: `t_dyn%use_wsplit`, `%wsplit_maxcfl` (`src/types/mod_dyn.F90:126-127`,
  serialized :199/:227; round trip `test/test_types.F90:125-156` checks neither today).
  Every driver pins `use_wsplit=.false.` EXCEPT `src/drivers/fesom_pressuredump.F90:513-514`
  (forces `.true.`, `maxcfl=1.0`, dumps `w_split_e/w_split_i` at :940-941 for the FESOM2
  byte-gate `tools/run_pressure_gate.sh`, and counts `cfl_z > maxcfl` at :563-575). The
  FESOM2 byte-gates are retired since bottom-at-vertices (`tools/run_conserve_pi.sh` header);
  the split fields there can no longer be byte-compared and will not be.
- Tests: `test/test_ivertvisc.F90` (momentum TDMA closed forms), `test/test_vinv.F90`
  (mesh+partition scaffold), `test/test_bottom.F90` (area/areasvol conventions).
  `add_fesom_test(<name> <np>)` in `test/CMakeLists.txt`; sources are GLOBbed.
- Mesh conventions: bottom at vertices, 1-D `area(n)`, `areasvol(n)` (`areasvol = area`,
  `mod_mesh_areas.F90:450`); `adv_tra_ver_upw1` flux `= −(w·T_up)·area(n)` at interfaces,
  0 at the bottom, unsigned `−w·T·area` at the surface; FCT LO update
  `fct_LO = (T·hnode + (F_h + F_v(nz)−F_v(nz+1))·dt/areasvol)/hnode_new`
  (`oce_adv_tra_driver.F90:128-136`).

## Development Approach

- **testing approach**: TDD (user preference, established for every no-oracle scheme here)
- complete each task fully before moving to the next
- make small, focused changes; preserve the existing discretization everywhere except the
  split function and the FCT port
- **CRITICAL: every task MUST include new/updated tests** for the code it changes; tests are
  separate checklist items and must pass before the next task
- **CRITICAL: update this plan file when scope changes during implementation**
- run `ctest` in `build_intel_dp` after each task; the conservation gate
  (`tools/run_conserve_pi.sh`) at Task 4 and at the end

## Testing Strategy

- unit tests: `test/test_wsplit.F90` (limiting function; split identity on pi; the
  transition-smoothness scan on the non-FCT path, np 1/2), `test/test_wimpl_tra.F90` (FCT
  implicit column solve + driver-level assembly, np 1/2), `test/test_types.F90` (parameter
  round trip)
- e2e: `fesom_conserve` configs with `FESOM3_WSPLIT=1` (hard conservation gate, np 1/2 in
  ctest, np 8 in the shell gate), with a measured `FESOM3_WSPLIT_MAXCFL` so the split fires
- every "no jumps"/"has teeth" claim carries a positive control that must FAIL
- no oracle exists for the smooth split (FESOM2 has the hard one only): the tests ARE the
  specification (L54)

## Progress Tracking

- mark completed items with `[x]` immediately when done
- add newly discovered tasks with ➕ prefix
- document issues/blockers with ⚠️ prefix

## Solution Overview

Three parts, in dependency order:

1. **Limiting function** (pure, elemental, its own module `oce_wsplit`):
   `f = wsplit_implicit_fraction(cfl, cmin, cmax)` = implicit share, `w_i = f·w`,
   `w_e = w − w_i`. Argument: FESOM2's `CFL_z` (unchanged definition, so `wsplit_maxcfl`
   keeps its exact meaning: explicit `CFL_z` never exceeds it). Plus
   `wsplit_check_params(cmin, cmax)` (logical) used by the drivers.
2. **Split** (`compute_Wvel_split`): one line per face instead of the `dd` branch; new
   `t_dyn%wsplit_mincfl` (type default `0.5`, i.e. `0.5·wsplit_maxcfl` at the default
   `maxcfl = 1.0`), serialized.
3. **FCT tracer port** (`adv_tra_vert_impl`): FESOM2's per-column upstream backward-Euler
   TDMA on the low-order solution with `w_i`, then the LO upwind flux recomputed with the
   full `w` so the antidiffusive flux is `HO(w) − LO(w)` as in FESOM2.

Decisions: Option A of the brainstorm (FESOM2 `CFL_z`, parameters `(wsplit_mincfl,
wsplit_maxcfl)`); no NEMO-style horizontal-Courant coupling; no FESOM2 hard-switch mode;
default `wsplit_mincfl = 0.5·wsplit_maxcfl`; `use_wsplit` stays `.false.` in production
drivers (enabling it in `job_levante` is the user's call).

## Technical Details

### Limiting function (Shchepetkin 2015, Sec. 3.1; NEMO `wAimp`)

With `Cu_cut = 2·Cu_max − Cu_min`, `D = Cu_max − Cu_min` and `F = 4·Cu_max·D`:

```
Cu <= Cu_min           : f = 0                                  (fully explicit)
Cu_min < Cu < Cu_cut   : f = (Cu−Cu_min)^2 / ( F + (Cu−Cu_min)^2 )
Cu >= Cu_cut           : f = (Cu − Cu_max) / Cu                 (explicit part capped)
```

Verified properties (review): at `Cu_cut` the middle branch gives `f = D/Cu_cut`, the upper
branch `(Cu_cut−Cu_max)/Cu_cut = D/Cu_cut`; slopes `2xF/(F+x²)² = Cu_max/Cu_cut²` and
`Cu_max/Cu²` agree; at `Cu_min` `f = f' = 0`. `Cu_e = Cu·(1−f)` is non-decreasing
(`d/dCu ∝ F − x² − 2x·Cu_min ≥ 0` on the middle branch). `f → 1` as `Cu → ∞`. The
degenerate `Cu_min = Cu_max` (`F = 0`) is FESOM2's hard switch in exact arithmetic — used
only as the positive control of the smoothness test, never as a mode.

Floating point: above `Cu_cut`, `1−f = Cu_max/Cu` is formed by cancellation, so
`Cu·(1−f)` carries a relative error `~eps·Cu/Cu_max`; tolerances below account for it.

Parameter validation: `wsplit_check_params(cmin, cmax) = (cmax > 0 .and. cmin >= 0 .and.
cmin <= cmax)`; the drivers `error stop` on `.false.`.

### `compute_Wvel_split`

```
if (.not. use_wsplit) then  Wvel_e = Wvel; Wvel_i = 0          ! literally today's off path
else
  do node = 1, nNodL;  do nz = ulevels_nod2D(node), nlevels_nod2D(node)
      f = wsplit_implicit_fraction(CFL_z(nz,node), mincfl, maxcfl)
      Wvel_i(nz,node) = f*Wvel(nz,node)
      Wvel_e(nz,node) = Wvel(nz,node) - Wvel_i(nz,node)
```
`w_e = w − w_i` makes `w_e + w_i` equal `w` to within 1 ulp (bitwise only when
`|w_i| ≥ |w|/2`, Sterbenz); the consumers need no more than that.

What the cap bounds: `CFL_z ≤ Cu_max` caps the FACE Courant number counted against both
cells; a cell with outflow through both faces can still have an explicit outflow fraction
up to `~2·Cu_max·h_min/h` on non-uniform layers. That is FESOM2's property too, and it is
why the boundedness test (C4) uses a uniform-`w` column (one inflow + one outflow face per
cell).

### `adv_tra_vert_impl(dt, w, ttf, mesh, partit)` (port of FESOM2 :90-240)

Per owned column, `zinv = dt`, `v = zinv·area(n)/areasvol(n)` (f3: 1-D; `= dt` exactly
since `areasvol = area`, kept in the FESOM2 form for readability):
```
surface  a=0;  b = hnode_new + w(nz)·v − min(0,w(nz+1))·v;  c = −max(0,w(nz+1))·v
interior a = min(0,w(nz))·v;  b = hnode_new + max(0,w(nz))·v − min(0,w(nz+1))·v;  c = −max(0,w(nz+1))·v
bottom   a = min(0,w(nz))·v;  b = hnode_new + max(0,w(nz))·v;  c = 0
rhs      tr = −a·T(nz−1) − (b − hnode_new)·T(nz) − c·T(nz+1);  solve;  T += tr
```
Column sums of the matrix equal `hnode_new` (conservative). Row sums are
`h' + (w_i(nz) − w_i(nz+1))·v = h − (w_e(nz) − w_e(nz+1))·v`, the thickness left after the
EXPLICIT step, so diagonal dominance (TDMA stable, pivots > 0) holds iff no cell is
drained by the explicit step — a PRECONDITION, not a kernel property (the cap bounds
faces, not cells: "What the cap bounds" above; C4 asserts it). Boundedness belongs to the
COMBINED step: `T^{n+1}` is a convex combination of `T^n`, the explicit donors' `T^n` and
the implicit donors' `T^{n+1}` (weights `(h − v·out_e)`, `v·in_e`, `v·in_i`), NOT of the
`T*` the implicit solve starts from (review 2026-10-05). The unsigned surface term matches the
explicit surface flux `−w·T·area` of `adv_tra_ver_upw1` (L57 form consistency). FESOM2
computes `zbar_n/Z_n` here but never uses them — dropped. Columns with fewer than 2 layers
are not supported by the row layout (same trap as `adv_tra_ver_upw1`, documented there):
a PRECONDITION stated in the kernel header and asserted once at setup by `read_mesh`
(`assert_min_layers`; review round 2 — the kernel validates nothing; pi minimum is 4
layers). The driver branch becomes FESOM2's:
`call adv_tra_vert_impl(dt, wi, fct_LO, mesh, partit)` then
`call adv_tra_ver_upw1(w, ttf, mesh, adv_flux_ver, o_init_zero=.true., partit=partit)`
with the FULL `w` (`pwvel => w` for FCT is already in place). Local `a,b,c,tr,cp,tp` are
per column (private if OpenMP is ever added).

### Why constants are preserved (the free-stream test)

`hnode_new` is advanced with the full `w`. For uniform `T`, the explicit LO step with `w_e`
gives `T* = T·(1 + dt·δw_i·area/(areasvol·hnode_new))`; the implicit step solves
`(hnode_new + dt·δw_i·area/areasvol)·T^{n+1} = hnode_new·T*`, i.e. `T^{n+1} = T` in exact
arithmetic, for ANY split with `w_e + w_i = w`. This pins the split identity, the
flux-form coefficients and the `hnode_new` usage.

### Why the smoothness test must NOT use the FCT low-order pair (review finding)

Explicit upwind with `w_e` followed by implicit upwind with `w_i` across a single face is
exactly split-independent: the donor cell gives `T₂·(h₂−dt·w_e)/(h₂−dt·w+dt·w_i) = T₂`,
and the receiver then holds `T₁h₁ + dt·w·T₂` whatever `f`. The pair differs from the
unsplit upwind step only at `O(dt²·L_i·L·T)`, which needs curvature in `T` or in `w`. The
kink of the hard switch is FIRST order only where explicit and implicit operators differ
at first order — the non-FCT path: QR4C (centred) on `w_e` plus the upwind TDMA on `w_i`.
There the tendency's derivative with respect to `w` jumps by `[UPW − QR4C](T)` at the
threshold, which is non-zero for a `T` with curvature (for linear `T` and uniform `w` the
two flux divergences coincide). Exactly (X0): `J = cq·h²·(1+Cu_c)/2` per unit `Cu` — the
shorthand `[UPW − QR4C](T) = cq·h²/2` is its `Cu_c → 0` limit; the factor `(1+Cu_c)` is
the explicit pre-advection `T* = T + w_c·P_q T` at the threshold, which the TDMA then acts
on (measured kink `1.0025·J`; the shorthand would read 2.005). Hence part X below runs on
QR4C + `diff_ver_part_impl_ale` with `Kv = 0`, quadratic `T`, uniform interior `w`, and
derives the jump in closed form.

### Diagnostics

`fesom_conserve` (as built; amended after review 2026-10-05): ONE statistics line printed
for EVERY config (max `CFL_z` is how a cap is chosen for a config, so it must be visible
with the split off), over owned faces at the last step, MPI-summed/maxed: max `CFL_z`,
faces with `w_i ≠ 0` (= `CFL_z > mincfl`, the onset), faces on the CAPPED branch
`CFL_z ≥ Cu_cut = 2·maxcfl − mincfl` (where `Cu_e` saturates exactly at `maxcfl`; at
`CFL_z = maxcfl` a face is still on the bend), faces with `f ≥ 0.5` (= `CFL_z ≥ 2·maxcfl`)
and `max f`. `FESOM3_WSPLIT_EXPECT_SPLIT=N` (VALUE-based): `error stop` unless at least `N`
faces split and at least one has `f ≥ 0.5` (the gate configs use `N = 1000`, 8x below
the measured 8632; the trip path is ctest `fesom_conserve_zstar_wsplit_vacuous_np1`).
The production driver prints the parameters in its banner. `fesom_pressuredump`: its
`cfl_z > maxcfl` count becomes a count of `w_i ≠ 0`; `w_split_e/w_split_i` are NO LONGER
dumped (nothing read them); `tools/pressure_diff.py` keeps them in `SKIP_FIELDS` so the
FESOM2-side records do not fail the "MISSING in F3" test (the byte-gate is retired; the
hard split no longer exists to compare against).

## What Goes Where

- **Implementation Steps**: code, tests, docs in this repo
- **Post-Completion**: enabling `use_wsplit` in `job_levante`, choosing `wsplit_mincfl`,
  long-run evaluation

## Implementation Steps

### Task 1: Limiting function `wsplit_implicit_fraction` + `wsplit_check_params` (TDD)

**Files:**
- Create: `src/oce/oce_wsplit.F90` (module `oce_wsplit`)
- Create: `test/test_wsplit.F90` (part W, no mesh)
- Modify: `test/CMakeLists.txt` (`add_fesom_test(test_wsplit 1)`, `... 2`)

- [x] write part W (parameters `(0.5,1.0)`, `(0.9,1.0)`, `(0.0,1.0)`, `(0.25,0.5)`):
  - W1 `f == 0` exactly for `Cu ∈ {0, Cu_min/2, Cu_min}`
  - W2 cap: `|Cu·(1−f) − Cu_max| ≤ 1e-14·Cu` for `Cu ∈ {Cu_cut, 2, 5, 50, 1e4}`
  - W3 limits/range: `f(1e6·Cu_max) > 1 − 2e-6`; `0 ≤ f ≤ 1` on a 10⁴-point grid in `[0, 20]`
  - W4 C⁰ joints: both branch formulas evaluated AT `Cu_min` and AT `Cu_cut` agree to
    1e-14 (the function itself is continuous iff its branches agree there)
  - W5 C¹ joints, `h = 1e-6`: at `Cu_min` both one-sided difference quotients have
    `|q| ≤ 2h/F` (closed-form slope 0); at `Cu_cut` both agree with `Cu_max/Cu_cut²` to
    1e-5 relative
  - W6 monotone: `f` non-decreasing on the grid; `Cu_e` non-decreasing up to the
    saturated-branch noise `4·eps·Cu` (⚠️ was "4 ulp of `Cu_max`", which drops the
    `Cu/Cu_max` factor of the error estimate above: measured max decrease 3.9e-15 at
    `Cu ≈ 17.5` for every parameter set = 17.5 ulp(1.0) / 35.5 ulp(0.5) = 1.23 eps·Cu)
  - W7 elemental: array call equals the element-wise loop bitwise
  - W8 `wsplit_check_params`: accepts `(0,1)`, `(0.5,1)`, `(1,1)`; rejects `(−0.1,1)`,
    `(1.1,1)`, `(0.5,0)`
- [x] implement the module (three branches; `wsplit_check_params`)
- [x] run `test_wsplit` — must pass before Task 2 (np 1/2 OK; full ctest 36/36)

### Task 2: `compute_Wvel_split`, `wsplit_mincfl`, pressuredump, and the smoothness scan

**Files:**
- Modify: `src/oce/oce_ale.F90` (`compute_Wvel_split` + its header comment :489-496)
- Modify: `src/types/mod_dyn.F90` (`wsplit_mincfl = 0.5_WP`, write/read lines :199/:227)
- Modify: `test/test_types.F90` (`test_dyn` :125-156: non-default `use_wsplit`,
  `wsplit_mincfl`, `wsplit_maxcfl` round-tripped)
- Modify: `src/drivers/fesom_pressuredump.F90` (:498-501, :513-514 comments; :563-575
  diagnostic counts `w_i /= 0`), `tools/pressure_diff.py` (drop `w_split_e/w_split_i`
  from the compared fields; comment why)
- Modify: `test/test_wsplit.F90` (parts S and X, pi mesh at np 1/2 like `test_vinv`)

- [x] write part S: prescribe `w` (nonzero at every level incl. surface, both signs) and a
  `cfl_z` field spanning `[0, 5]`; call `compute_Wvel_split` (sets `(0.5,1)`, `(0.9,1)`,
  `(0.25,0.5)`; TDD: against the hard split S1 measured 2 ulp and S2 was not bitwise):
  - S1 `|w_e + w_i − w| ≤ 1 ulp(w)` at every owned+halo face (measured max 1.00 ulp)
  - S2 `w_i(nz,n) == f(cfl_z(nz,n))·w(nz,n)` bitwise
  - S3 `use_wsplit=.false.` → `w_e == w`, `w_i == 0` bitwise (incl. sign of zero)
  - S4 faces with `cfl_z ≤ mincfl` have `w_i == 0`; faces with `cfl_z ≥ 2·maxcfl−mincfl`
    have `|w_e|·cfl_z/|w| == maxcfl` to 1e-14·cfl_z (measured ≤ 2.3e-16; every class
    populated: e.g. 11514 / 19796 / 73579 faces below/bend/capped for `(0.5,1)` at np 1)
- [x] write part X (transition smoothness, non-FCT path; needs a minimal tracer set-up
  with `tra_adv_lim /= 'FCT'`, `Kv = 0`, no surface fluxes, Redi off):
  - X0 derive in the test header the closed-form derivative jump of the hard switch at the
    threshold for a quadratic `T` on uniform layers with uniform interior `w`:
    `J = [UPW − QR4C](T)` at the probe cell (⚠️ derived EXACTLY: `J = cq·h²·(1+Cu_c)/2`
    per unit `Cu`; the shorthand `[UPW − QR4C](T) = cq·h²/2` is its `Cu_c → 0` limit — the
    factor `(1+Cu_c)` is the explicit pre-advection of `T` to `T*` at the threshold, which
    the TDMA then acts on. Measured with the hard switch: `D2(kink)/(J·ΔCu) = 1.0025`,
    the 0.25 % being the predicted `ΔCu/(1+Cu_c)` curvature of the implicit branch)
  - X1 scan `Cu` on ≥ 600 points in `[0, 3·maxcfl]` (601 points, `ΔCu = 0.005`, probe
    cell 5 of every owned column with ≥ 10 levels, both signs of `w`): measured
    `max|D2|/(ΔCu·max|D1|) = 0.96` (w>0) / `1.17` (w<0) against the bound 10
    (➕ review 2026-10-05: the bound is now 5 — a C⁰-but-not-C¹ linear ramp of `f` would
    have passed 10 for w<0 (expected ratio ~8); the probe is cell `nzmin + 4` of ONE
    representative column per `(ulevels, nlevels)` class (every other column shrunk to
    2 layers for the scan: equal-depth columns are identical by construction, and this
    took the test from 65-92 s to 10 s), with cavity columns `ulevels = 3` synthesised
    on every 10th global node; measured values unchanged, 0.962 / 1.170)
  - X2 positive control, the degenerate `(1,1)` function: ratio `100.3` / `64.1` (violates
    X1 by 10x / 6.4x; the SAME numbers came out of the pre-rewrite hard split, which
    failed X1 exactly this way); kink `D2 = 1.0025·J·ΔCu` (floor `0.5`, pinned to 5 %)
  - X3 fully explicit limit: bitwise at all 101 scan points with `Cu ≤ mincfl` (every
    owned cell); fully implicit limit: chain == pure `do_wimpl` TDMA bitwise, and the TDMA
    == the closed-form upwind row recursion to 3.8e-16 relative (both signs)
    (➕ review 2026-10-05: the "chain == pure TDMA bitwise" check was a tautology — with
    `w_e = 0` the QR4C flux is identically zero, so the chain IS the TDMA by
    construction — and was dropped; the chain is compared with the closed-form recursion
    directly, 3.6e-16)
- [x] rewrite `compute_Wvel_split` (table in Technical Details); update its header
  (`compute_CFLz`/`compute_Wvel_split` made public for the test)
- [x] add `wsplit_mincfl` to `t_dyn` + serialization; extend `test_types` `test_dyn`
  (non-default `use_wsplit=.true.`, `mincfl=0.3`, `maxcfl=0.8` round-tripped); drivers
  rely on the type default (only the env-hook drivers set it, Task 4)
- [x] `fesom_pressuredump`: comments, `w_i /= 0` count, `pressure_diff.py` field list
  (`SKIP_FIELDS = {w_split_e, w_split_i}`, printed as SKIP)
- [x] run `test_wsplit` np 1/2 (65 s / 38 s), `test_types`, full `ctest` 36/36 — must pass
  before Task 3

### Task 3: Port `adv_tra_vert_impl` (FCT path) + driver-level assembly test

**Files:**
- Modify: `src/oce/oce_adv_tra_ver.F90` (new public `adv_tra_vert_impl` + the 2-layer guard)
- Modify: `src/oce/oce_adv_tra_driver.F90` (:139-142 error stop → FESOM2 sequence; header
  :26-28)
- Create: `test/test_wimpl_tra.F90`
- Modify: `test/CMakeLists.txt`

- [x] write `test_wimpl_tra` (pi mesh, np 1/2; `hnode` prescribed, `area/areasvol` from
  `compute_geometry`; `hnode_new = hnode − dt·(w(nz)−w(nz+1))·area/areasvol` for a pure
  column; the LO update formula of the driver is replicated and documented as such; TDD:
  the test was written and registered first and failed to compile on the missing kernel;
  identical numbers at np 1 and 2):
  - C1 identity: `w_i == 0` → `ttf` unchanged bitwise (with `hnode_new ≠ hnode`)
  - C2 constancy: uniform `T`, divergent `w` with `w = 0` at surface and bottom,
    splits `(0.5,1.0)`, `(0.0,1.0)`, `(0.9,1.0)`: explicit LO (`adv_tra_ver_upw1(w_e)`) +
    implicit (`w_i`) → `max|T^{n+1} − T| < 1e-13·|T|` (measured ≤ 5.3e-16, both signs
    of `w`, non-uniform layers `[0.75,1.25]·h0`, max `CFL_z` 4.3/5.2; every class
    populated, e.g. 12726/25027/67136 faces below/bend/capped for `(0.5,1)`)
  - C2b constancy with `w(surface) ≠ 0` of both signs (`hnode_new` consistent): same bound
    (the only unsigned row) (measured ≤ 5.3e-16; the uniform-`T` surface budget
    `−dt·w(1)·T·area` closes to ≤ 7.2e-16)
  - C3 conservation: non-uniform `T`, `w = 0` at surface/bottom:
    `Σ hnode_new·T^{n+1}·areasvol == Σ hnode·T·areasvol` to 1e-13 relative (measured
    ≤ 1.0e-15 per column, owned total ≤ 2.8e-16)
  - C3b with `w(surface) ≠ 0`: content changes by exactly `−dt·w(1)·T_top·area` (1e-13)
    (⚠️ the exact form is `−dt·area·(w_e(1)·T_old(1) + w_i(1)·T_new(1))` — the explicit
    part carries the old, the implicit part the new surface value; the plan's form is its
    uniform-`T` case, asserted in C2b; measured ≤ 9.5e-16)
  - C4 large Courant, uniform-`w` column (one inflow + one outflow face per cell),
    `CFL_z ≈ 10`: `T^{n+1}` within `[min T, max T]`, finite (precondition asserted:
    per-cell explicit outflow `≤ hnode`) (one thin cell `0.08·h0`, `|w|dt = 0.8·h0`:
    `CFL_z` 10.8 at its faces, 14.0 for `w > 0` where the drained bottom cell thins to
    `0.2·h0`; explicit outflow/hnode ≤ 0.957; overshoot 1.3e-15 of the range; positive
    control: the uncapped explicit step overshoots by 0.95 / 5.6 of the range)
  - C5 fully implicit limit: `w_e = 0, w_i = w`, two-cell closed form in flux form with the
    `hnode_new` mass: error < 1e-12 (kernel alone vs its two-row form 1.4e-16; the
    sequence vs the flux form 2.8e-16; two-cell mass 3.7e-16; both signs)
  - C6 driver-level assembly: call `do_oce_adv_tra` (FCT, QR4C vertical) with
    `use_wsplit` on, a linear `T(z)` in a column with uniform interior `w` so the limiter is
    inactive (assert `fct_plus/minus` leave the antidiffusive flux unlimited), and compare
    with the test's own assembly of the correct sequence from the public parts
    (`adv_tra_ver_upw1(w_e)` → `adv_tra_vert_impl(w_i)` → `HO(w) − LO(w)` →
    `oce_tra_adv_flux2dtracer(use_lo)`); positive control: the WRONG assembly
    `HO(w) − LO(w_e)` must differ from the driver's result by more than the tolerance
    (that is the wiring bug a conservation gate cannot see: it double-counts the `w_i`
    transport yet still telescopes)
    (⚠️ "uniform interior `w`" with `w = 0` at the bottom face is clipped by the limiter
    at its last cell — the drained end cell keeps `lo = T` and is the cluster extremum;
    the configuration where the limiter is provably inactive is upward `w0` through the
    open surface down to the second-last cell with the LAST cell inert, `Cu = 1.2`;
    derivation in the test header. Measured: every factor b3 applies to a nonzero face
    == 1 and the clipped flux == the unclipped one; driver `fct_LO` and `del_ttf_advvert`
    == the test assembly BITWISE (0.0) at np 1/2; the wrong assembly differs by
    29.9 / 58.8 K·m (slope ±0.02 K/m) == the predicted double-counted `w_i` transport
    `(−w_i(nz)·T(nz) + w_i(nz+1)·T(nz+1))·area·dt/areasvol` to 5e-14; plus the closed
    form of the whole step (LO recursion `d_n = ((Cu/2)·s + r_i(n+1)·d_{n+1})/(1 + r_i(n))`
    + the HO increments `Cu·s/4` at the surface, `−(Cu/4)·s/(1 − Cu/2)` at the drained
    cell) reproduced to 1.4e-14 K)
- [x] port `adv_tra_vert_impl` (f3 area rule, owned loop, no `zbar_n`, 2-layer guard)
- [x] replace the driver error stop with the FESOM2 sequence; update the module header
- [x] confirm `tools/run_pressuredump_pi.sh` still runs (it forces `use_wsplit=.true.`
  through this path) (FESOM2 side: exit 0, 130 MB dump into `/scratch/a/a270029/
  pressuredump_pi`; FESOM3 `fesom_pressuredump` exit 0 with `ulimit -s unlimited` as the
  script sets — without it the `wr_r3` stack temporary segfaults before any `w_split`
  record, an environment matter; the retired byte-gate FAILs by design. ⚠️ that driver
  calls `diff_tracers_ale` only, never `do_oce_adv_tra`, so it does not reach
  `adv_tra_vert_impl`; the Task 4 `fesom_conserve` configs are the first end-to-end run)
- [x] run `test_wimpl_tra` np 1/2 + full `ctest` — must pass before Task 4 (np 1 0.9 s,
  np 2 0.7 s; full ctest 38/38 in 153 s)

### Task 4: Drivers, diagnostics, gate

**Files:**
- Modify: `src/drivers/fesom_conserve.F90`, `src/drivers/fesom_lifecycle_native_mr.F90`
- Modify: `tools/run_conserve_pi.sh`, `test/CMakeLists.txt`

- [x] env hooks (both drivers): `FESOM3_WSPLIT` (presence → `use_wsplit`),
  `FESOM3_WSPLIT_MINCFL`, `FESOM3_WSPLIT_MAXCFL` (values; defaults 0.5 / 1.0), validated
  with `wsplit_check_params` (error stop); banner with both numbers (the `MINCFL` default
  is `0.5·maxcfl` in the drivers, i.e. 0.5 at the default cap, so a config that sets only
  `MAXCFL` keeps the type's onset/cap ratio; the inadmissible pair `(0.9, 0.5)` measured
  to error-stop before init)
- [x] `fesom_conserve`: last-step split statistics + `FESOM3_WSPLIT_EXPECT_SPLIT=1` error
  stop on zero split faces (`wsplit_statistics`: owned faces, levels
  `ulevels_nod2D..nlevels_nod2D`, `w_i ≠ 0`, `|w_i| ≥ 0.5|w|`, `max |w_i|/|w|` guarded at
  `w = 0`, plus `max CFL_z` and the counts above `mincfl` / `maxcfl` / `2·maxcfl` — the
  breakpoints at which `f > 0`, the cap starts and `f ≥ 0.5`; MPI-summed/maxed, rank 0;
  printed for EVERY config; positive control: `EXPECT_SPLIT=1` with the cap 1.0 gives
  0 split faces and exit 1) (⚠️ review 2026-10-05: the cap does NOT start at `maxcfl` —
  `Cu_e` saturates at `Cu_cut = 2·maxcfl − mincfl`; the "`> maxcfl`" column counted the
  faces FESOM2's hard switch would have split. Rebuilt as one line (Technical Details →
  Diagnostics): `w_i ≠ 0` / capped `CFL_z ≥ Cu_cut` / `f ≥ 0.5` / max `f` / max `CFL_z`;
  measured capped faces 1130 at the gate cap. `EXPECT_SPLIT` is value-based, the trip
  path is a ctest)
- [x] measure `max CFL_z` of the 20-step pi conserve run (cold start from rest: expected
  far below 0.5) and choose `FESOM3_WSPLIT_MAXCFL` for the gate config so that a
  substantial number of faces split, some with `f ≥ 0.5`; record the measured counts here
  (➕ measured, default zstar config, split off, last step, owned faces: max CFL_z =
  2.757e-2; faces with CFL_z > 0.005 / 0.01 / 0.02 / 0.025 / 0.04 = 2568 / 499 / 26 / 5 / 0.
  Chosen maxcfl = 0.005 (mincfl = 0.0025 by the 0.5·maxcfl default); the alternative 0.01
  gave only 26 faces at f ≥ 0.5. With the split ON (wsplit config): faces f>0 = 8632,
  f≥0.5 = 500, max f = 0.818, max CFL_z 2.755e-2 at np 1 (np 2: 8633 / 500; np 8: 8628 /
  496); momadv-vinv-wsplit: 8647 / 497, max f 0.819. Consistency: f>0 count == faces
  above mincfl and f≥0.5 count == faces above 2·maxcfl in every run, as derived.)
- [x] gate configs: `wsplit FESOM3_WSPLIT=1 FESOM3_WSPLIT_MAXCFL=0.005
  FESOM3_WSPLIT_EXPECT_SPLIT=1`, `momadv-vinv-wsplit` (same + `FESOM3_MOMADV_OPT=1`), an
  np=8 block for `wsplit`; ctest `fesom_conserve_zstar_wsplit_np{1,2}` and
  `fesom_conserve_zstar_vinv_wsplit_np{1,2}` (the halo path of `w_e/w_i` is read by the
  next step's momentum advection — np ≥ 2 required) (the heredoc loop's `tail -7` became
  `tail -9` and the np=8 block uses `tail -8` so the two statistics lines are visible for
  every config; ctest now 42 tests)
- [x] run the full gate — GATE OK required before Task 5 (GATE OK, 21 configs; wsplit
  drift heat/salt/vol = 1.7e-14 / -9.5e-15 / -2.6e-14 at np 1, -8.5e-16 / 3.5e-16 /
  3.9e-15 at np 8; momadv-vinv-wsplit -1.1e-14 / -4.1e-15 / 1.5e-14; all pre-existing
  configs' drifts identical to the Task 3 gate log, i.e. the `use_wsplit=.false.` path is
  unchanged; full ctest 42/42)

### Task 5: Verify acceptance criteria

- [x] every requirement in Overview implemented; `use_wsplit=.false.` path bit-identical
  (suite + gate results unchanged) (Acceptance list below. Off path: a fresh gate run
  has all 21 drift rows (heat/salt/vol) and both statistics lines of every config
  identical to the Task 4 gate log, whose 18 pre-existing rows are in turn identical to
  the pre-plan reference log (⚠️ the Task 4 entry's "22 configs / 19 pre-existing" is a
  miscount; the logs hold 21 / 18). `git diff 66177c4..HEAD -- src/` touches the off path
  only structurally: `compute_Wvel_split`'s `.not. use_wsplit` branch is the previous
  loop body (`w_e = w`, `w_i = 0`, same node/level range) ahead of an early return; the
  FCT driver change sits inside `if (dynamics%use_wsplit)`; `adv_tra_vert_impl` is a new
  routine called only there; `t_dyn` gains `wsplit_mincfl` (read on the on path only,
  serialized after `wsplit_maxcfl`, round-tripped only by `test_types`); with no
  `FESOM3_WSPLIT*` set the drivers pin `use_wsplit=.false.`, `maxcfl=1.0` as before, and
  `wsplit_statistics` is a read-only print after the step loop)
- [x] full `ctest` and full gate green at np 1/2/8 (ctest 42/42 in 231 s incl. the np 1/2
  wsplit / vinv-wsplit conserve tests; `run_conserve_pi.sh` GATE OK over 21 configs:
  zstar np 1/2/8, wsplit np 1/8, momadv-vinv-wsplit np 1, every physics combination;
  logs `gate_task5.log` / `ctest_task5.log` in the session scratchpad)
- [x] split statistics of the `wsplit` gate config reported in the commit message (the
  Task 4 commit 772456b carries a title only, so they go into the Task 5 commit: max
  CFL_z 2.755e-2, cap 0.005 / onset 0.0025, faces with w_i≠0 8632, f≥0.5 500, max f
  0.818, drift heat/salt/vol 1.7e-14 / -9.5e-15 / -2.6e-14 at np 1; np 8: 8628 / 496 /
  0.818, drift -8.5e-16 / 3.5e-16 / 3.9e-15; momadv-vinv-wsplit np 1: 8647 / 497 /
  0.819, drift -1.1e-14 / -4.1e-15 / 1.5e-14; identical to the Task 4 run)

Acceptance (Overview requirements against the code, 2026-10-02):
- ✓ the split lives where FESOM2 has it: `compute_Wvel_split` (`src/oce/oce_ale.F90:
  490-558`) at the end of `vert_vel_ale` (:443), `w_i = f·w`, `w_e = w − w_i`, owned+halo
- ✓ the Shchepetkin C¹ function is the ONLY split: `oce_wsplit::wsplit_implicit_fraction`
  (`src/oce/oce_wsplit.F90:40-58`, three branches); the `dd` hard switch is gone and
  `wsplit_maxcfl` is read nowhere else in `src/oce`; `mincfl = maxcfl` is a test control
- ✓ parameters: `t_dyn%wsplit_mincfl` (`src/types/mod_dyn.F90:133`, serialized :207/:236,
  `test_types` round trip); `wsplit_check_params` (:60-66) guards both env-hook drivers
  (`fesom_conserve.F90:174`, `fesom_lifecycle_native_mr.F90:237`)
- ✓ consumers: momentum TDMA `w_i` (`oce_dyn_ivertvisc.F90:124`), momentum advection
  `w_e` (`oce_dyn_velrhs.F90:197`, `oce_dyn_vinv.F90:171`), tracer-diffusion TDMA `w_i`
  (`oce_ale_tracer.F90:532`); the FCT path ported: `adv_tra_vert_impl`
  (`oce_adv_tra_ver.F90:140-241`) + LO flux on the full `w` (`oce_adv_tra_driver.F90:
  149-150`), no error stop left
- ✓ nothing else in the discretization changed: `git diff 66177c4..HEAD -- src/` was 8
  files at Task 5 (new module, the split, the FCT branch, the `t_dyn` field, three
  drivers); +3 comment-only files in Task 6 (`fesom_stepdump`, `oce_ale_tracer`,
  `oce_dyn_ivertvisc`); the review fixes below touch the same set
- ✓ diagnostics/hooks: `FESOM3_WSPLIT` / `_MINCFL` / `_MAXCFL` + banners in both drivers;
  `fesom_conserve` `wsplit_statistics` (:510-561) + `FESOM3_WSPLIT_EXPECT_SPLIT`;
  `fesom_pressuredump` counts `w_i ≠ 0`; `pressure_diff.py` skips `w_split_e/w_split_i`
- ✓ tests: `test_wsplit` W/S/X, `test_wimpl_tra` C1-C6, `test_types`, ctest wsplit
  configs np 1/2, gate wsplit np 1/8 + momadv-vinv-wsplit np 1 — all green above

### Task 6: [Final] Update documentation

- [x] `docs/HANDOFF.md`: `use_wsplit` section (function, parameters, env vars, where the
  implicit parts act, the FCT sequence, the retired pressure byte-gate fields) (the
  "Implicit vertical advection (`use_wsplit`)" paragraph under "Momentum advection
  options" — which still said no end-to-end gate was possible and the correction was
  dormant — rewritten into the subsection "Implicit vertical advection: the smooth
  `use_wsplit` split" with the measured net numbers of Tasks 1-5; the M2-era "unported
  kernel" bullet marked ported)
- [x] `docs/LESSONS.md`: one entry (hard switch = degenerate case of the C¹ function;
  the single-face LO explicit/implicit pair is exactly split-invariant, so a smoothness
  test needs operators that differ at first order; the constancy test is the consistency
  proof) (L58, plus the measured-cap and the review's tolerance lessons; L57's last bullet
  and the two M1.4/M2.9b-era "unported" mentions carry a superseded note pointing to L58)
- [x] stale comments: `src/oce/oce_adv_tra_driver.F90:26-28`, `src/oce/oce_ale.F90:489-496`,
  `src/drivers/fesom_stepdump.F90:147-152`, `fesom_pressuredump.F90:498-501, 563-565`,
  `src/oce/oce_ale_tracer.F90:536-537` (the driver header, the `compute_Wvel_split`
  header and the pressuredump comments were already rewritten in Tasks 2/3 — verified, no
  change; rewritten now: the `fesom_stepdump` note, the `oce_ale_tracer` module header
  and its `do_wimpl` comment, the `oce_dyn_ivertvisc` header (the split is active above
  `wsplit_mincfl`, not `maxcfl`; end-to-end net named); tree grep for "unported" /
  "dormant" / "not yet supported" clean outside `docs/plans/` and `HANDOFF-archive.md`;
  comment-only edits, build clean, smoke ctest green)
- [x] move this plan to `docs/plans/completed/` (moved by the harness at the end of the run)

### Review fixes (2026-10-05, after the five-agent review)

- [x] guards as ctests: `fesom_conserve_zstar_wsplit_vacuous_np1` (cap 0.5 splits nothing →
  "vacuous config" and no drift report; the pass regex also pins the driver default
  `mincfl = 0.5·maxcfl` = 0.2500), `fesom_conserve_wsplit_badparams_np1` ((0.9, 0.5)
  rejected), `fesom_conserve_wsplit_badvalue_np1` (`MAXCFL=0.5x` → "not a real"; the
  `read(env,*,iostat=)` results were unchecked in both env-hook drivers),
  `test_wimpl_tra_onelayer_np1` (the kernel's < 2-layer `error stop`)
- [x] `FESOM3_WSPLIT_EXPECT_SPLIT=N` value-based: ≥ N split faces AND ≥ 1 face at `f ≥ 0.5`
  (gate/ctest configs `N = 1000`)
- [x] halo contract: `test_wsplit` S5 (global-index inputs → `cfl_z/w_e/w_i` at every halo
  node == owner bitwise against an `exchange_nod`'d copy, 1429 faces at np 2, 0
  mismatches; positive control with local-index inputs: 1429/1429 mismatch). The np 2
  conserve tests run the halo consumers but cannot see a wrong halo value.
- [x] `compute_Wvel_split` validates on the on path: the pair on every call (the `t_dyn`
  default `mincfl = 0.5` is absolute), the ≥ 2-layer column requirement once with a
  collective verdict (the kernel's rank-local `error stop` stays as a cheap assertion)
- [x] cavity (`ulevels = 3`) and 2-layer columns synthesised by global index in both tests
  (309 / 29 owned on pi); `test_wimpl_tra` C1-C5 run on the mix (restored for C6: the
  FCT limiter needs element-consistent `ulevels`); C6 now also assembles w < 0 with
  `oce_tra_adv_fct` in the test's own assembly (limiter active, wiring bitwise; the
  wrong assembly differs by 3.9 / 1.4 K·m) and counts the nonzero antidiffusive faces
  (95469) so (i) cannot pass vacuously
- [x] `test_wsplit`: one representative column per `(ulevels, nlevels)` class (10 s / 6 s
  at np 1/2, was 65-92 / 38-41 s), X1 bound 5, X3a not re-run on the X2 scan, the X3b
  tautology and W4's grid comparison against the test's own formula copies dropped, W7
  reworded (a compiler-path check under `-no-prec-div -fimf-use-svml`, kept), the
  `apply_chain(dyn%w_e, dyn%w_i, …)` aliasing (F2008 12.5.2.13) removed
- [x] statistics: one line, breakpoints `mincfl` / `Cu_cut` / `2·maxcfl` (capped faces
  1130 at the gate cap); gate tails fixed so the line is visible for every config;
  `GM+wsplit` gate config added (22 configs); `fesom_pressuredump` no longer dumps
  `w_split_e/w_split_i`
- [x] docs: the dominance claim → precondition + combined-step convexity (kernel header,
  above), FESOM2 line refs (:120, :142-150), module header (in-place kernel, owned loop),
  consumer timing (tracers same step owned; momentum next step at owned elements' halo
  vertices), GM `fer_w` comment, HANDOFF/LESSONS breakpoint wording, all plan citations
  on the `docs/plans/completed/` path

### Review fixes, round 2 (2026-10-05, style/convention pass)

- [x] validation moved to SETUP, kernels stateless (house convention: `error stop` in
  `src/oce` is config-time only; mesh invariants are walked once in `mod_mesh_read`):
  `oce_wsplit::wsplit_assert_params(mincfl, maxcfl)` is the one call every driver that
  enables `use_wsplit` makes (obligation documented in `t_dyn`; `fesom_conserve`,
  `fesom_lifecycle_native_mr`, now also `fesom_pressuredump`); the ≥ 2-layer column
  requirement is `mod_mesh_read::assert_min_layers`, called by `derive_vertical_bounds`
  on every mesh path with a collective integer verdict (`mod_halo` gained
  `allreduce_sum_i0`); `compute_Wvel_split` lost its per-call pair check, the `save`
  flag and the real-valued count, `adv_tra_vert_impl` its per-column `error stop` — both
  headers state the preconditions like `adv_tra_ver_upw1`
- [x] `test_wimpl_tra_onelayer_np1` (hidden env-var mode + raw `add_test`) replaced by
  `test/test_minlayers.F90` via `add_fesom_test` (pi passes, one column shrunk to one
  layer → the walk must stop with "fewer than 2 layers")
- [x] guard ctests keep the PASS-regex (right reason) + FAIL-regex (did not continue) pair:
  `WILL_FAIL` cannot make the exit code load-bearing alongside a regex — CMake inverts
  the verdict AFTER the regexes, so `WILL_FAIL` + a FAIL regex PASSES a driver that
  prints the drift line and exits 0 (verified with CMake 3.26); the vacuous test's banner
  regex no longer counts `f8.4` blanks
- [x] naming/style: `use_wsplit` (not `_env`) for the presence toggle in both drivers, one
  banner shape `'<driver>: … ENABLED (use_wsplit; wsplit_mincfl=…, wsplit_maxcfl=…)'`,
  named statistics counters `n_split/n_capped/n_half` reduced with `allreduce_sum`,
  `n_split_faces` in `fesom_pressuredump`, `0.0_WP` throughout `adv_tra_vert_impl`,
  `partit=partit` keyword in the driver call, the header table of `fesom_conserve`
  re-aligned, `kprobe` / `layer_kind` / `mod_constants::pi` / named selector thresholds /
  `ctl_margin` in the tests, C1/C5 assert the `hnode_new > 0` precondition
- [x] the gate triple is hoisted once per file (`WSPLIT_CFG` in `run_conserve_pi.sh`,
  `_wsplit_cfg` in `test/CMakeLists.txt`); plan citations use the CURRENT path (the
  final move commit updates them); `run_conserve_pi.sh` cites the moved
  bottom-at-vertices plan at `completed/`

## Post-Completion

*No checkboxes — informational.*

- **Production**: `job_levante` decides `FESOM3_WSPLIT=1` and `FESOM3_WSPLIT_MINCFL`
  (default `0.5·maxcfl`). Use a fresh RUNID; do not switch it on together with other
  changes.
- **Evaluation**: compare a short run with/without the split on core2 — split-face
  statistics per step, `w` and tracer fields near thin z* surface layers and in
  convective columns, conservation drift.
- **Later options**: NEMO-style interface Courant number / horizontal-Courant coupling as
  an alternative argument of the same function; `cfl_z` and `f` as output means.
