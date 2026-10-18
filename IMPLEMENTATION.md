# FESOM_F — Implementation and Design

This document explains how FESOM_F is built and how it differs from FESOM2. It is written for
developers who know FESOM2 and want to understand the reimplementation's structure and the reasoning
behind it. For building and running, see [README.md](README.md); for how the model is validated, see
[TESTING.md](TESTING.md).

> Naming, as in the README: the project is **FESOM_F** in prose, but real identifiers keep the
> repository's working name `fesom3` (env-var prefix `FESOM3_`, build dirs `build_*`, drivers
> `fesom_*`, git tags `m0…m10`).

---

## Part I — Overview

### The guiding idea

FESOM_F has one organizing principle: **first reproduce exactly, then optimize.** The numerical
kernels — the equation of state, pressure gradient, advection, viscosity, the free-surface solver,
sea-ice rheology, vertical mixing, and so on — are transcribed essentially line-for-line from FESOM2
v2.7.3. What is *redesigned* is everything around the kernels: how data is organized, how the model
runs in parallel, how it is configured, and how it does I/O.

The payoff of this separation is a sharp, testable claim: for a fixed configuration (double
precision, triangular mesh, the Intel toolchain), FESOM_F produces **bit-for-bit identical** results
to FESOM2. Because the mathematics is unchanged, any difference between the two codes is a bug in the
plumbing, not a question of scientific interpretation. The validation methodology built on this claim
is the subject of [TESTING.md](TESTING.md).

So the "differences from FESOM2" described below are, by design, **not numerical**. They are
structural: derived types instead of global modules, an optional-argument parallelism pattern, a
generic halo layer, compile-time precision selection, mesh-arity scaffolding, write-once
configuration, and a distributed Zarr I/O stack.

### FESOM2 vs FESOM_F at a glance

| Aspect | FESOM2 v2.7.3 | FESOM_F |
|---|---|---|
| Numerical kernels | reference | transcribed verbatim → bit-identical through M10 |
| Bottom / topography | defined at **elements** (`elvls.out`); a scalar cell has its bottom at several levels | defined at **vertices** (`nlvls.out`); the element bottom is derived as the shallowest of its 3 or 4 corners (§13) |
| State | global module arrays (`MOD_DYN`, `o_ARRAYS`, …), `use`d everywhere | derived types (`t_mesh`, `t_dyn`, `t_tracer`, `t_ice`, `t_partit`) passed as arguments |
| Parallelism | per-kernel multi-rank code throughout | one code path; an **optional** `partit` argument adds the parallel case |
| Halo exchange | many hand-written per-array routines | generic `exchange_nod` / `exchange_elem` + `allreduce_*` |
| Precision | hardcoded double | compile-time `WP` (+ mesh precision `MP`); double is the anchor |
| Element arity | triangles, loops hardcoded `1:3` | `MAX_NV=4`, 4 vertex slots (triangle 4th = 1st): triangle, quad and mixed meshes (§7) |
| Configuration | namelists spread across modules | write-once `mod_config` (today: drivers use `FESOM3_*` env vars) |
| Output | gather to rank 0 → netCDF | distributed Zarr writers, no rank-0 gather, partition-independent |
| Restart | rank-0 gather → netCDF | partition-independent Zarr checkpoints (same I/O stack) |
| Free surface | semi-implicit, CG solve | the same CG solve (default) or split-explicit barotropic subcycling (`FESOM3_SSH_SE=1`, §14) |

### v1 scope

**In scope and complete (M0–M10):** the ocean dynamical core, sea-ice EVP/mEVP, GM/Redi, KPP and
TKE vertical mixing, the zstar vertical coordinate, multi-year forced runs (CORE2 and JRA55-do),
Zarr output, and restart/checkpoint — all byte-exact against FESOM2, on 1 rank and up to 864 ranks.

**After M10:** the bottom moved from elements to vertices (§13). That is a deliberate departure
from FESOM2, so bit-identity no longer applies to the ocean; §13 explains what replaces it.

**Hybrid meshes:** triangle, quad and mixed triangle+quad meshes (e.g. converted NEMO grids) run
through the same code path (§7); every kernel was converted to 4-slot formulas, so triangle results
are no longer bit-identical to FESOM2 either (roundoff-gated).

**Deferred (foundations laid, not exercised):** mixed/reduced precision, performance tuning, and
ports to other languages. These are discussed in §12.

---

## Part II — Reference

### 1. Goals and the validation contract

The project's contract is **`max|Δ| = 0`**: on a fixed configuration, every model field matches an
instrumented FESOM2 v2.7.3 reference ("the oracle") to the last bit, checked substep by substep,
on a single rank and across many. This is a much stronger bar than statistical agreement, and it is
what justifies transcribing the kernels verbatim rather than rewriting them: a clean rewrite that
merely "agreed well" could not be distinguished from a subtle bug. The full method, the reference
setup, and the results are in [TESTING.md](TESTING.md).

The anchor configuration is **Intel + double precision + Release**, with compiler flags copied
verbatim from FESOM2 (see [README.md](README.md) §2 and `cmake/fesom_flags.cmake`). The GNU build is
a portability check, not a second bit-identity anchor.

### 2. Source organization

`src/` holds 105 Fortran files: **72 library modules across nine subsystems**, plus **33 driver
programs** (entry points and byte-gate harnesses).

| Subsystem | Modules | Contents |
|---|---:|---|
| `params/` | 6 | precision, physical constants, write-once configuration, version |
| `types/` | 5 | the core derived types: mesh, partition, dynamics, tracers, ice |
| `infra/` | 10 | MPI halo layer, partitioning, loop bounds, clock, timers, dump I/O |
| `mesh/` | 4 | mesh read, geometry/areas, rotation |
| `oce/` | 28 | ocean physics kernels (advection, dynamics, pressure, mixing, ALE, GM/Redi, the SSH solvers) |
| `ice/` | 6 | sea-ice dynamics (EVP), thermodynamics, FCT advection, ocean coupling |
| `forcing/` | 3 | atmospheric forcing read, bulk fluxes, ancillary forcing |
| `io/` | 8 | Zarr writer, I/O decomposition, mesh diagnostics, restart, POSIX shims, coords |
| `step/` | 2 | timestep orchestration (`mod_step_oce`, `mod_model`) |
| `drivers/` | 33 | program entry points + per-subsystem byte-gate / smoke drivers |

The split between `oce/` (28 modules) and the rest mirrors FESOM2's own decomposition, which keeps
the kernel transcription one-to-one. The new subsystems relative to FESOM2's flat layout are
`types/` (the derived types that used to be global modules), a dedicated `io/` (the Zarr stack), and
`infra/` (the parallel and diagnostic plumbing).

### 3. Data model: derived types and dependency injection

FESOM2 keeps state in global modules (`MOD_DYN`, `o_ARRAYS`, `MOD_ICE`, …) that routines pull in
with `use`. FESOM_F instead groups state into **derived types passed explicitly as arguments**
(`src/types/`):

- `t_mesh` — static mesh: connectivity, geometry, vertical structure, control-volume areas,
  the SSH stiffness matrix.
- `t_partit` — the parallel decomposition: owned/halo dimensions, the communication structure, the
  MPI communicator.
- `t_dyn` — ocean dynamics.
- `t_tracer` — the tracer set (temperature, salinity as separate entries).
- `t_ice` — sea ice (prognostic concentration/thickness/snow plus the EVP stress tensor).

A routine receives exactly the types it touches, so its dependencies are visible in its signature
rather than hidden in module scope. Two consequences are worth calling out:

**Prognostic vs. work state.** Each type separates the genuinely prognostic state from the
recomputed-every-step scratch. In `t_dyn`, prognostic velocity (`uv`, `uv_rhs`, `uvnode`, `eta_n`)
lives directly in the type, while the auxiliary 3-D fields — in-situ density, `N²`, hydrostatic
pressure, the PGF, the mixing coefficients `Kv`/`Av`, the KPP and GM/Redi intermediates — live in a
nested `t_dyn_work` and are *not* serialized: they are diagnostics of the tracer state, restored by
recomputation rather than from a restart. This split is exactly what makes the restart set small and
well-defined (§10).

**Config rides in the types.** Per-entity settings (per-tracer advection scheme, per-run viscosity
parameters, the solver tolerances in `t_solverinfo`) live in the type they belong to, not in a
global config module — so they travel with the data and there is no global state to keep in sync.

### 4. One code path, serial and parallel: the optional `partit` pattern

The single most useful structural decision is how FESOM_F handles "1 rank" versus "N ranks". Rather
than maintaining separate serial and parallel kernels, every kernel takes an **optional** `partit`
argument and asks a small helper, `owned_bounds` (in `src/infra/mod_part_bounds.F90`), for its loop
limits:

```fortran
pure subroutine owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
    type(t_mesh),   intent(in)            :: mesh
    integer,        intent(out)           :: nNodO, nNodL, nEdgeO, nElemO
    type(t_partit), intent(in), optional  :: partit
    if (present(partit)) then
        if (partit%npes > 1) then
            nNodO = partit%myDim_nod2D                       ! owned
            nNodL = partit%myDim_nod2D + partit%eDim_nod2D   ! owned + halo
            ...
            return
        end if
    end if
    nNodO = mesh%nod2D; nNodL = mesh%nod2D                   ! 1-rank: global counts
end subroutine
```

When `partit` is **absent or `npes==1`**, the bounds are the global mesh counts and the kernel runs
its proven serial path byte-for-byte unchanged; the serial drivers simply omit `partit`. When
`npes>1`, the bounds become the partition's owned and owned+halo dimensions, and the caller adds the
matching halo exchanges. Because an "owned+halo" loop collapses to the global loop at one rank, the
same source serves both cases, and the serial result remains the bit-identity anchor while the
multi-rank case is a surgical addition. The companion helpers `local_dims` (the full set of
node/edge/element bounds) and `is_multirank` (which guards the exchanges) complete the pattern.

This is threaded through the whole timestep: `step_oce` (§9) carries an optional `partit` to every
kernel it calls.

### 5. The generic halo / MPI layer

FESOM2 has many hand-written exchange routines, one per array shape and purpose. FESOM_F has a single
generic interface in `src/infra/mod_halo.F90`:

```fortran
public :: exchange_nod, exchange_elem, exchange_elem_full, allreduce_sum, allreduce_max
public :: exchange_elem_shared, shared_copy_max_elem      ! shared elements, see below
```

`exchange_nod`/`exchange_elem` are generic over 2-D, 3-D, integer, and block field shapes; the
implementation is one pattern — pack owned values, `Isend`/`Irecv`, unpack into halo cells — driven
by the `com_struct` communication lists in `t_partit`. The halo is **broadcast-only**: each halo
entry is overwritten with its owner's value, never accumulated.

**Shared elements.** FESOM2's partition rule gives an element to every rank that owns one of its
vertices, so an element on a partition border has two or three *owned* copies (24,798 of 244,659
elements at CORE2 `dist_512`). Each owner updates its own copy, and order-dependent sums make the
copies differ; the ranks then compute different fluxes across their common edges, and with the linear
free surface the global volume leaks (LESSONS L70). `partit%com_elem2D_shared`, built at the end of
`read_mesh_local` by `src/infra/mod_shared_elem.F90` from data every rank already has, names one
designated owner per shared element: the highest rank among its vertex owners, whose copy the restart
and output writers keep. `exchange_elem_shared` sends that copy to the other owners, and `update_vel`
calls it on `uv` every step before `exchange_elem(UV)` (`FESOM3_SHARED_SYNC`, default 1; timer
"(of which) shared sync"). The other per-element fields (`uv_rhsAB`, the EVP stresses) are computed
from equal inputs and stay equal without a sync; `shared_copy_max_elem` (exchange-and-compare, like
the stale-halo probe) measures the spread of any element field, and the step diagnostic and the
guard tests use it.

For the free-surface solver, `allreduce_sum` wraps the conjugate-gradient dot-products. Its
determinism is load-bearing: it issues the same `MPI_Allreduce(MPI_IN_PLACE, …, MPI_SUM, …)` over
the same communicator and datatype as FESOM2, so OpenMPI selects the same reduction tree and the
global sum is bit-identical given bit-identical per-rank partial sums. `allreduce_max` is exact by
construction (a selection, no rounding) and only drives loop counts. GPU/OpenACC hooks present in the
structural reference were intentionally left out; one inert argument (`luse_g2g`) remains as a
foundation marker.

### 6. Precision scaffolding

Precision is a compile-time parameter, with a two-tier design (`src/params/mod_precision.F90`):

```fortran
#else
    integer, parameter :: WP = 8            ! double (anchor)
#endif
    integer, parameter :: MP = max(WP, 4)   ! mesh precision: at least single
```

`WP` is the **working precision** — the precision under test, switchable to single (`-DUSE_SINGLE_PRECISION`)
or half (`-DUSE_HALF_PRECISION`). `MP` is the **mesh precision** used for geometry, coordinates, and
work arrays; it is held at no less than single so that a future reduced-`WP` build does not lose the
mesh to overflow. `MPI_WP` is the matching MPI datatype.

At the v1 anchor, `WP = MP = 8`, so the scaffolding collapses to plain double precision and the build
is bit-identical to FESOM2 (which hardcodes double). The mixed-precision machinery is therefore
present but unexercised — a foundation for later work (§12), not a v1 feature.

### 7. Mesh-arity generalization

FESOM2 is triangle-only, with element loops hardcoded to three vertices. FESOM_F keeps the FESOM2
array names (so kernels transcribe verbatim) but generalizes the element arity
(`src/types/mod_mesh.F90`):

```fortran
integer, parameter :: MAX_NV = 4   ! max element vertices (3 = tri, 4 = quad)
...
integer, allocatable :: elem2D_nodes(:,:)    ! (MAX_NV, elem2D)
integer, allocatable :: elem2D_nnodes(:)     ! (elem2D) vertices per element
```

Connectivity arrays are dimensioned `(MAX_NV, elem2D)` and carry a per-element vertex count
(3 or 4). **Mixed triangle + quad meshes are supported** (plan
`docs/plans/completed/2026-10-08-hybrid-meshes.md`), in one code path with no triangle/quad branch in
any kernel:

- **4-slot convention.** Every element has 4 vertex slots, clockwise; a triangle repeats its 1st
  vertex in slot 4. `elem2d.out` may have 3 or 4 columns.
- **Kite weights.** `elem_w(MAX_NV,·)` (median-dual share of each vertex, a triangle exactly
  1/3,1/3,1/3,0) and `nod_in_elem2D_w`; element means are `sum(elem_w*f)`, node gathers weight each
  element by `elem_area·nod_in_elem2D_w` instead of FESOM2's `/3`.
- **Gradients.** Green–Gauss `gradient_sca(8,·)` (x in rows 1:4, y in 5:8; a triangle's slot 4 = 0),
  exact for linear fields on convex elements, equal to P1 on triangles.
- **Missing neighbour = self.** `elem_neighbors`/`elem_edges` and `edge_tri` never hold a zero: on a
  boundary edge `edge_tri(2) == edge_tri(1)` and boundary tests compare the two.

Every kernel (ocean dynamics, tracers, mixing, GM/Redi, forcing/coupling, sea-ice EVP and FCT) was
converted, so triangle meshes are **no longer bit-identical** to FESOM2 or to the pre-hybrid code:
they agree to roundoff, gated by `tools/run_tri_regression.sh` ([TESTING.md](TESTING.md) §6b). The
details (file formats, `fesom_ini`, exchanges, the SSH matrix, sea-ice mass matrix, open findings) are
in `docs/HANDOFF.md`, section "Hybrid (mixed triangle + quad) meshes". (Geometry arrays are `MP`, per
§6.)

### 8. Configuration

FESOM_F provides a **write-once** configuration module (`src/params/mod_config.F90`, decision D4):
module variables hold the run settings, populated once by `read_config()` from a single
`namelist.config` at startup and treated read-only thereafter, with namelist group names and defaults
transcribed from FESOM2's `g_config`. Per-entity settings deliberately live in the data types
(§3), not here.

In the current state, **the production lifecycle drivers do not call `read_config`** — they
configure physics from in-code defaults and read run-specific knobs from `FESOM3_*` environment
variables (mesh, initial state, forcing, physics toggles, output, restart). The authoritative list is
the environment block at the top of `src/drivers/fesom_lifecycle_native_mr.F90`; the common variables
are tabulated in [README.md](README.md) §4. The only namelist read at runtime today is
`namelist.io`, for output (§10).

So `mod_config` exists and is the intended home for run configuration; consolidating the driver
configuration into the single `namelist.config` file it already supports is a planned step, not a
gap in the module itself. (One honest wrinkle: because the drivers set physics in code, a few
defaults differ from FESOM2's namelist defaults — e.g. `use_sw_pene` defaults to `.false.` here as a
fail-safe — so a value is never silently assumed from a namelist the driver does not read.)

### 9. The timestep and lifecycle

The ocean step is assembled in `src/step/mod_step_oce.F90` as a faithful transcription of FESOM2's
`oce_timestep_ale`, with one driver running the leaf kernels in sequence and live data flow (each
kernel reads the previous kernel's output):

```
compute_vel_nodes → pressure_bv (EOS, hydrostatic pressure, N²) → pressure_force →
mixing (PP / KPP / TKE) + mo_convect → compute_vel_rhs (Coriolis AB2, PGF, momentum advection) →
viscosity_filter → impl_vert_visc_ale → compute_ssh_rhs_ale → solve_ssh_ale (free-surface CG) →
update_vel → compute_hbar_ale → update_eta_n → vert_vel_ale → solve_tracers_ale → update_thickness_ale
```

With `FESOM3_SSH_SE=1` the block from `compute_ssh_rhs_ale` to `update_eta_n` is replaced by the
split-explicit barotropic subcycling (§14); the rest of the step is unchanged.

`step_oce` follows the dependency-injection rule (§3): inputs the core does not itself produce — the
tracer diffusivity, the surface heat/freshwater/salt fluxes, the wind stress — are explicit
arguments, sourced by the caller from forcing and mesh. It also carries the optional `partit` (§4),
so the same routine drives both the serial byte-gate driver and the multi-rank production driver.

The production driver `fesom_lifecycle_native_mr` wraps this step in a forced, coupled loop: it
advances the model clock, reads atmospheric forcing, computes air–sea and air–ice fluxes (the
"native" flux path), steps the sea ice (EVP/mEVP → FCT advection → thermodynamics), steps the ocean
via `step_oce`, and periodically writes output and checkpoints.

### 10. I/O and checkpointing

Both output and restart are built on one principle: **no rank-0 gather, and files that do not depend
on the rank count.**

**Output (M9).** The writer is a hand-rolled Zarr v2 implementation (`src/io/mod_io_zarr.F90`):
C-order chunks, partial-chunk fill-padding, optional LZ4 compression, consolidated metadata.
`src/io/mod_io_decomp.F90` redistributes a field from its owning ranks to a subset of *writer* ranks
using a single `MPI_Alltoallv` keyed on the canonical global index (`decomp_init`,
`decomp_redistribute`), so the resulting store is **partition-independent** — a run on 2 ranks and a
run on 8 ranks produce byte-identical stores. `mod_io_means` handles field registration,
mean/snapshot accumulation, per-field output cadence, node and element fields, and vector rotation to
geographic coordinates. Output is one Zarr store per variable per period, plus a
`fesom.mesh.diag.zarr` describing the mesh; everything opens directly in xarray. Variables and writer
knobs come from `namelist.io` (`config/namelist.io`).

**Restart/checkpoint (M10).** `src/io/mod_io_restart.F90` reuses that same stack. A checkpoint is an
immutable folder holding one single-variable snapshot Zarr store per prognostic field (no time
dimension, no accumulation), plus a small `checkpoint.json`. Fields are **registered by live pointer**
into the model arrays (`restart_register_state` maps the full ocean + ice prognostic state, including
the EVP stress tensor `sigma`), and the reader reads back into those same pointers. Writes are
**atomic**: a checkpoint is staged into a `.tmp` folder, then atomically renamed into place, and a
one-line `restart.latest` pointer is flipped — so a crash leaves either the old or the new
checkpoint, never a half-written one. A keep-N prune trims old checkpoints. The cross-rank file
inversion uses `decomp_gather` (the inverse of the output redistribution), and on read each field's
halo is reconstructed with the exchange variant its in-step consumer expects. POSIX operations
(rename, fsync, rmtree) go through small `bind(C)` shims in `src/io/mod_io_posix.F90` rather than
`execute_command_line`. The result: a run can stop and resume byte-exactly, and a checkpoint written
on one rank count can be read on another. (The exact reproducibility guarantees — and their one
FESOM2-inherited limit — are in [TESTING.md](TESTING.md).)

### 11. Correctness decisions that make byte-identity possible

Bit-for-bit agreement between two independently compiled programs is fragile; a handful of specific
decisions are what hold it. These are documented in detail in `docs/LESSONS.md`; the load-bearing
ones:

- **Compiler flags copied verbatim.** The Intel anchor flags are transcribed exactly from FESOM2
  v2.7.3, including the *absence* of `-march` on Levante (FESOM2 comments it out → an SSE2 baseline
  with no FMA contraction). Matching this is required for bit-identity; the strings live in
  `cmake/fesom_flags.cmake`.
- **Reduction determinism.** The CG dot-product all-reduces mirror FESOM2's exactly (§5), so the
  parallel free-surface solve is bit-identical to the serial one.
- **Preconditioner vectorization (LESSONS L29).** A multi-rank "reproducibility floor" on CORE2
  turned out not to be a floor at all but an auto-vectorized preconditioner divide reordering the
  arithmetic; pinning it (a `!DIR$ NOVECTOR`) restored `max|Δ| = 0`.
- **Flush-to-zero at runtime (LESSONS L51).** A 2-year run diverged at day 107 because the FESOM_F
  *process* ran with flush-to-zero (FTZ) **off** while FESOM2 had it **on**: FESOM_F kept a denormal
  snow thickness (~1e-309 m) that FESOM2 flushed to exactly zero, which flipped an ice-albedo branch.
  The fix forces FTZ on at startup (`ieee_set_underflow_mode`), matching the FESOM2 process — a
  reminder that bit-identity depends on the floating-point *runtime mode*, not only the source.
- **A genuine, FESOM2-inherited floor.** At multiple ranks, element fields can differ by ~1 ULP on
  the ~562 elements that FESOM2's partitioning lets more than one rank "own"; the partition-
  independent stores deduplicate these. This is inherent to FESOM2 (its own netCDF restart shows it),
  not a FESOM_F bug — see [TESTING.md](TESTING.md).

### 12. Scope, limitations, and roadmap

**Done (M0–M10, tagged `m0`, `m1`, `m2-mvp`, `m3` … `m10`):** dynamical core; sea-ice EVP/mEVP;
GM/Redi; KPP and TKE mixing; the zstar vertical coordinate; multi-year forced runs (CORE2,
JRA55-do); Zarr output; restart/checkpoint. All byte-exact against FESOM2 on 1 rank and up to 864
ranks. After M10: bottom at vertices (§13), triangle/quad/mixed meshes (§7) and a split-explicit
barotropic solver (§14, its stability defaults still open). The next milestone is open.

**Foundations in place but not yet exercised:**

- **Mixed / reduced precision.** The `WP`/`MP` split (§6) is built for it; the next step is to flip
  `WP` to single and protect the cancellation-sensitive computations.
- **Performance.** The "reproduce exactly" phase deliberately preceded optimization; one targeted
  win (the surface-forcing read) is already done. Systematic profiling and throughput (SYPD) work
  are future.

**Beyond v1 (new ports, not byte-ports):** scientific multi-decade validation against observations
and FESOM2 climatologies; and additional physics — IDEMIX, backscatter, biogeochemistry, ice-shelf
cavities, icebergs, and coupling — which would be fresh implementations rather than bit-identity
transcriptions.

---

*See also:* [README.md](README.md) (build & run) · [TESTING.md](TESTING.md) (validation & results) ·
`docs/HANDOFF.md` (development log) · `docs/LESSONS.md` (reproducibility traps) · `docs/plans/`.

---

### 13. Bottom at vertices

In FESOM2 the bottom depth is defined at **elements** while the elevation is at **vertices**, so a
scalar control volume has its bottom at several different levels. That complicates every surface and
bottom exchange process — ice-sheet coupling, sediment resuspension — because there is no single
depth to attach them to. FESOM_F puts the bottom at vertices instead: the scalar cell becomes a
straight prism with one bottom level, and velocities touching topography vanish by construction.

**The contract.** The vertex column is authoritative; the element bounds are derived and are never
read from a mesh file:

```
layers of vertex v :  nz = ulevels_nod2D(v) .. nlevels_nod2D(v)-1
layers of element e:  nz = ulevels(e)       .. nlevels(e)-1

ulevels(e) = maxval(ulevels_nod2D(elem2D_nodes(:,e)))
nlevels(e) = minval(nlevels_nod2D(elem2D_nodes(:,e)))
```

An element is wet only where **all** of its 3 or 4 corners are, so its layer range is exactly its
fully wet prisms. The design note's `tlayer`/`blayer` are `ulevels_nod2D`/`nlevels_nod2D-1`, and
`tlayer_elem`/`blayer_elem` are `ulevels`/`nlevels-1`; the FESOM2 names were kept, so no kernel was
renamed. Level (not layer) indexing was kept too. The full contract is written out at the vertical
block of `src/types/mod_mesh.F90`.

**Why this is nearly free.** Every velocity kernel already loops `ulevels(e)..nlevels(e)-1`, and
bottom drag is already applied at `nlevels(elem)-1` (`oce_dyn_ivertvisc.F90:177`). Inverting the
producer therefore restricts velocity work to full prisms, moves bottom drag to the last full cell,
and confines the stiffness integration to the full element interval — with no kernel edits at all.

**What did have to change.**

- **`compute_node_areas`** — the scalar cell is now a straight prism, so its horizontal area is the
  full median-dual area at *every* wet layer rather than a depth-gathered sum. This is the design
  note's `area(1:myDim+eDim)`. The array stays 2-D so that `area(nlevels_nod2D(n),n) == 0` keeps
  expressing a closed bottom.
- **`tr_xynodes`** (`oce_ale_tracer.F90`) — a node *average* of element gradients, so it is
  normalised by the area that actually contributed, not by `areasvol` (which is now the full prism
  area). `momentum_adv_scalar` is a flux divergence over the control volume and is deliberately
  left on the full area.
- **zstar** — the stretch spans the whole vertex column, since that is now where
  `area(nz) == area(1)`. That in turn required the `helem` rebuild to cover the element's deepest
  layer: FESOM2 could skip it because `nlevels_nod2D_min(n) <= nlevels(e)`, an inequality the
  vertex bottom reverses.
- **`edge_dxdy`** (R7) — stored in metres rather than radians, with the `r_earth*mean(elem_cos)`
  factor folded in at construction instead of applied at each point of use. `edge_len` (metres) was
  added alongside; nothing consumes it yet.

**A trap worth knowing.** `nlevels_nod2D_min` and `ulevels_nod2D_max` look like aliases of the
vertex column under this scheme and are not — they are 2-ring quantities bounding work that reaches
*adjacent elements* from a node, and they sit several levels away from the vertex column over most
of a real mesh. In particular the MUSCL scheme (`adv_tra_hor_muscl`, `oce_adv_tra_hor.F90`) reads
`tr_xy` at the up/downwind elements of an edge, and `tr_xy` is uninitialised below an element's
bottom; that read therefore tests the up/downwind element's own wet range (`ulevels..nlevels-1`).

**The required invariant**, asserted at mesh setup:

```
maxval over e in adj(n) of nlevels(e) == nlevels_nod2D(n)
```

Every vertex's deepest scalar cell must have at least one wet adjacent element. It holds because
`nlvls.out` is exactly the max over adjacent `elvls.out` (verified on pi and core2). It is not a
quality metric: `oce_ale.F90:88`, `oce_ale.F90:377` and `oce_pressure_bv.F90:310` all divide without
guarding, so a mesh violating it yields NaN rather than a wrong-but-finite answer.

**Bathymetry consequence.** With `blayer(v) = nlvls.out`, the derived element bottom composes as
`min ∘ max` — a morphological closing — so it is never shallower than `elvls.out`: on core2 4.3% of
elements deepen, mean `+0.067` levels, `+0.38 %` ocean volume, concentrated at island edges and
shelf breaks rather than at overflow sills. The alternative `blayer = min over adjacent elvls` is a
double erosion with nothing to undo it (`-9.27 %` volume, 3966 dead bottom cells) and was rejected.

**Deferred.** Horizontal diffusion, the Redi adjustment, and the optional inflation of boundary
scalar-cell areas. (The boundary self-duplication of `edge_tri` and the element self-neighbours are
done — §7.)

---

### 14. Split-explicit barotropic solver

An alternative to the semi-implicit CG free-surface solve, switched on with `FESOM3_SSH_SE=1`
(module `src/oce/oce_ssh_se.F90`, plan `docs/plans/2026-10-17-split-explicit.md`). It needs no
global reduction per step (two small halo exchanges per substep), which is the point for the later
GPU port. With the switch unset the CG path runs and its results are byte-identical to the build
before the SE work ([TESTING.md](TESTING.md) §6c). SE is not bitwise to CG; it is gated by
invariants, determinism and restart exactness.

**Scheme.** Sergey Danilov's notes (`~/port_kokkos/ssh_sergey/subcycling.tex`): the dissipative
AB3–AM4 subcycling of Shchepetkin & McWilliams (2005), as in Banerjee et al. (2024, GMD 17, 7051).
Per baroclinic step τ the 2-D system (η, Ū = Σₖ helem·u) takes M substeps of dtbt = τ/M:

```
U*      = (3/2+β) U^m − (1/2+2β) U^{m−1} + β U^{m−2}                          (AB3)
η^{m+1} = η^m + dtbt·(I(U*) − W·areasvol)/areasvol                          (W: zstar only)
η^A     = δ η^{m+1} + (1−δ−γ−ζ) η^m + γ η^{m−1} + ζ η^{m−2}                     (AM4)
U^{m+1} = U^m + dtbt·(−f e_z×U* − g H^A ∇η^A + F + V(U^m − U^n))
⟨⟨Ū⟩⟩    = Σ_m U*·(1/M)
β = 0.281105, ζ = 0.00976186 − 0.13451357χ, γ = 0.083445 − 0.513584χ, δ = ½ + 2ζ + γ + 2χ
```

I is the inflow edge scatter of `compute_hbar_ale` (positive = convergence) and χ the dissipation
parameter. There is no averaging filter: the new elevation is η^M. F is formed once per step from the
predictor, F = G/τ + g·Hⁿ·∇ηⁿ + f e_z×(a₂šⁿ + a₁š^{n−1}) with G = Σₖ helemⁿ·UV_rhs and š = Σₖ helem·uv.
It removes from the 3-D increment the pressure gradient and the Coriolis term that the substeps
compute themselves. Coriolis uses the stored sums š because `compute_vel_rhs` keeps Coriolis and
momentum advection in one AB2 array; this is exact under linfs and second order under zstar.

**What it replaces.** In `step_oce`, `se_bt_forcing → se_subcycle → se_trim_finish` take the place of
`update_stiff_mat_ale → compute_ssh_rhs_ale → solve_ssh_ale → update_vel → compute_hbar_ale →
update_eta_n` (the old block is unchanged, in the `else` branch). The trim corrects the 3-D velocity
depth-uniformly to the mean transport, uvₖ = u**ₖ − (Σⱼ helemⁿⱼ u**ⱼ − ⟨⟨Ū⟩⟩)/Hⁿ_e, followed by the
shared-element sync and halo exchange of `uv` as in `update_vel`. It then sets hbar_old = hbar and
hbar = eta_n = η^M on owned and halo nodes, and ssh_rhs_old = I(⟨⟨Ū⟩⟩) − W·areasvol. The predictor
(`compute_vel_rhs`, `viscosity_filter`, `impl_vert_visc_ale`) and the tail (GM, `vert_vel_ale`,
tracers, `update_thickness_ale`) are untouched. The tail reads the result through hbar, hbar_old and
ssh_rhs_old. SE does not use d_eta, ssh_rhs, dhe or the stiffness matrix. It is wired into
`fesom_lifecycle_native_mr` and `fesom_conserve`; any other driver stops with `FESOM3_SSH_SE=1`. It
requires the AB2 predictor and no cavities.

**Depth law under the vertex bottom.** One element depth function serves both modes:

```
H_e(η) = Hⁿ_e + Σ_v elem_w(v,e)·c_{e,v}·(η_v − ηⁿ_v),     Hⁿ_e = Σₖ helemⁿ(k,e)
```

Under linfs c ≡ 0, so H is the cold Σhelem. Under zstar c is the share of a vertex's Δη that
`vert_vel_ale` puts into the layers inside the element's column (the stretch spans the vertex column
except its bottom layer, §13):
c_{e,v} = (zbar_3d_n(ulevels(e),v) − zbar_3d_n(K+1,v))/H̃_v with K = min(nlevels(e)−1,
nlevels_nod2D(v)−2) and H̃_v = zbar_3d_n(1,v) − zbar_3d_n(nlevels_nod2D(v)−1,v). It is 1 at vertices
whose column ends at most one level below the element's and below 1 at deeper vertices. c is
recomputed once per step. Through the real zstar tail Σhelem^{n+1} = H_e(η^{n+1}) holds to 4.5e-12 m
of 5650 m (test_ssh_se Z3); the law H0 + η (c ≡ 1) misses by 0.24–0.26 m. Freshwater W enters the
η equation only when `which_ALE ≠ linfs`.

**Shared elements.** A border element has two or three owned copies (§5). SE keeps them bitwise
equal by construction:

- F is synced once per step from the designated owner (`exchange_elem_shared`).
- Every per-element operator of the substeps (gradient, Coriolis, viscosity, H^A) is a gather over
  the element's own vertex and neighbour slots in fixed slot order, so equal inputs give equal
  copies of Ū and ⟨⟨Ū⟩⟩ without a per-substep sync.
- The one scatter, I into the nodes, writes owned nodes from owned edges.
- The arithmetic of a per-element loop must not depend on the element's local index. Under Intel's
  `-no-prec-div` a division by a loop invariant rounds differently in the vectorised body and in the
  scalar remainder, so ⟨⟨Ū⟩⟩ accumulates U*·(1/M) with the reciprocal computed once (LESSONS L71).

Per substep this costs `exchange_nod(η)` and `exchange_elem(Ū)`, the latter only with viscosity.
The copy spreads of Ū and ⟨⟨Ū⟩⟩ are exactly 0 in every test and gate (with the syncs off they are not,
test_ssh_se P3).

**Barotropic viscosity.** The notes' linear harmonic flux form, V_e = Σₖ κ_{e,k}·H_edge·(ΔŪ_nb/Hⁿ_nb −
ΔŪ_e/Hⁿ_e) with κ = A_h·l/(d·A_e), A_h = `se_visc_ah`·l (l the edge length, d the centroid distance),
applied to ΔŪ = Ūᵐ − Ūⁿ. It damps only the change within the step and leaves the slow flow alone.
There is no flow-dependent γ₀/γ₁ term and no barotropic bottom drag (bottom drag stays in the 3-D
predictor and so in F).

**CFL guard.** `se_init` estimates the fastest gravity mode of the substeps' own linear operator,
A η = −I(Hⁿ·(−g∇η))/areasvol, by 100 power iterations (ω_max; one allreduce and one `exchange_nod` per
iteration). It then sets M_min = ⌈τ·ω_max/(s·a_crit(χ))⌉, where a_crit is the AB3–AM4 stability
limit for one oscillator (Schur–Cohn test of the characteristic polynomial: 1.853 / 1.579 / 1.329
at χ = 0 / 0.1 / 0.25) and s = 0.9. CORE2 at Δt = 1800 s: ω_max = 3.15e-2 s⁻¹, M_min = 40 at χ = 0.1
and 48 at χ = 0.25 (0.06 s at np 8). A smaller M stops the run unless `FESOM3_SE_M_FORCE=1`. The
resolution estimate 0.5·res/√(gH) used before is not a stability limit (LESSONS L72).

**Knobs** (environment only, value-checked; the `FESOM3_SE_*` knobs are read only with SE on, and SE
prints nothing when off):

| env | default | meaning |
|---|---|---|
| `FESOM3_SSH_SE` | 0 | 1 = split-explicit, 0 = CG; any other value stops |
| `FESOM3_SE_M` | 50 | substeps per baroclinic step (≥ the CFL-guard minimum) |
| `FESOM3_SE_M_FORCE` | 0 | 1 = run below the CFL-guard minimum (with a warning) |
| `FESOM3_SE_CHI` | 0.1 | dissipation parameter χ, 0 ≤ χ ≤ 0.25 |
| `FESOM3_SE_VISC_AH` | 0.01 | barotropic viscosity A_h = value·l [m/s]; 0 = off |
| `FESOM3_SE_CHECK` | 0 | 1 = per-step invariant line `SE_CHECK` (η compatibility, trim, mean sea level, copy spreads; zstar adds `SE_CHECK_TAIL`: layer sum, w(1), depth law) |

The barotropic substeps have their own timer, "(of which) SE barotropic" under "dynamics ssh". At
CORE2 np 128 SE costs 12.9 ms/step in the SSH phase against SI's 9.1 (loop 146 vs 144 ms/step); the
planned CPU and GPU measurements are open.

**Restart.** With SE on, the checkpoint carries 10 more stores: `se_eta1`, `se_eta2` (η^{−1/M},
η^{−2/M}; node), `se_ubt{0,1,2}_{u,v}` (Ūⁿ, Ū^{−1/M}, Ū^{−2/M}; element, halo ELEM) and
`se_sbar_old_{u,v}` (š^{n−1}; element, halo NONE). The CORE2 lifecycle registers 37 instead of 27
fields. A resumed SE run equals the straight run (`max|Δ| = 0`, linfs and zstar, np 1 and 2, np 2 → 8
round trip). Limits:

- An SE run that finds no SE state in the checkpoint stops with "checkpoint has no split-explicit
  state: written with FESOM3_SSH_SE off; SI<->SE restarts are not supported".
- The other direction is not detected: an SE checkpoint read with SE off silently ignores the `se_*`
  stores. SE → SI is not supported.
- M and χ are not stored. Keep them fixed across a restart chain: the rings hold levels dtbt = τ/M
  apart.
- zstar: after `restart_read` the driver now rebuilds the derived thicknesses (`hnode_new`,
  `zbar_3d_n`, `Z_3d_n`, `helem` through `update_thickness_ale`). Every zstar restart needed this.
  An SI-zstar restart is still not exact: the SSH stiffness matrix accumulates −dhe every step, and
  neither dhe nor the accumulated values are in the checkpoint (the resumed run departs at its first
  SSH solve). SE does not use the matrix, so SE-zstar is exact.

**Open: stability at the default χ = 0.1 on CORE2.** With the defaults (χ 0.1, M 50, `se_visc_ah`
0.01), a cold-started CORE2 run with production physics blows up at step 331–340 (day 7), at np 8 and
np 128, under linfs and zstar. The cause is a grid-scale gravity mode with a period of 3 baroclinic
steps on the 20 m Antarctic shelf near 74°W 72°S under compact sea ice; it grows by 0.9 % per step.
The feedback runs through the sea ice: the ice reads ηⁿ, and its stress on the ocean enters F frozen
over the step. Removing the ice–ocean stress or the ice's view of η removes the growth. SI's
backward-Euler damping hides the same coupling. χ = 0.25 and `se_visc_ah` = 0.1 run the 480 screened
steps stably; the default is not decided yet. Details: the plan's Task 5.1 note, LESSONS L74 and
`docs/se_question_packet.md`. CORE2 gate runs longer than ~300 steps use `FESOM3_SE_CHI=0.25`.
