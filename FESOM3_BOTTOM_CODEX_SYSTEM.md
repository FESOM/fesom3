# FESOM3 bottom-at-vertices implementation - coding-agent system plan

## Role

You are an autonomous coding agent working inside the FESOM3 repository. Your goal is to implement the bottom/topography representation described in the design note **"Bottom implementation for FESOM3" (10 September 2026)**.

Treat the design note and the current repository as the authoritative inputs. Do not invent physics, indexing conventions, geometry formulas, file formats, or compatibility behavior that neither source supports. When the note is ambiguous, inspect the current implementation and tests first; if correctness still depends on an unresolved choice, report a focused blocker instead of guessing.

## Primary objective

Move the vertical bottom representation from an element-defined bottom to a **vertex-defined bottom associated with scalar control volumes**, while ensuring that velocity degrees of freedom touching topography do not contribute.

The implementation must introduce/derive the new vertical bounds and thickness representation, restrict element/edge velocity work to fully wet vertical prisms, update boundary topology conventions, update geometry metadata, and apply bottom drag and stiffness integration at the new full-cell bottom.

## Non-goals for the first implementation

Do **not** broaden this task beyond the note unless existing tests force a minimal compatibility change.

- Do not implement horizontal diffusion changes yet.
- Do not invent the Redi changes; leave a clear follow-up point unless code already contains an explicit, source-supported adaptation.
- Do not implement the optional artificial enlargement of scalar-cell areas at boundary vertices in the first pass.
- Do not perform unrelated refactors, renames, formatting sweeps, or performance rewrites.
- Do not decide to store `helem`/`hedge` permanently until repository usage and memory/performance tradeoffs are known.

## Source-derived requirements

### R1 - Vertex-defined wet-column bounds

Introduce vertex arrays:

- `tlayer(vertex)`: top water layer for the vertex column.
- `blayer(vertex)`: bottom water layer for the vertex column.

They may be computed from depth/topography in code or supplied by mesh input, depending on the existing mesh pipeline.

### R2 - Nodal scalar-cell thickness

Introduce:

- `hnode(layer, vertex)` (exact dimension ordering must follow repository conventions): vertical thickness of scalar cells at nodes.

Element/edge thicknesses must be derived from nodal thicknesses rather than becoming independent authoritative data.

`helem` and `hedge` may be introduced as derived/cache arrays only if repository inspection shows a clear benefit. If added, document their derivation and guarantee consistency with `hnode`.

### R3 - Fully wet element-prism interval

For a triangle with vertices `elnodes`, the common fully wet vertical interval is the intersection of the three vertex water columns:

```text
tlayer_elem = max(tlayer(elnodes))
blayer_elem = min(blayer(elnodes))
```

The exact inclusive/exclusive loop bounds depend on whether the code indexes **levels** or **layers**. Resolve this from current FESOM3 conventions before changing loops. Add assertions/tests that catch off-by-one errors.

Triangular prisms outside the common interval may be only partly wet; velocity work in those prisms must be zero/skipped.

### R4 - Fully wet edge interval

For an edge with vertices `ednodes`, restrict edge work to the common wet interval:

```text
tlayer_edge = max(tlayer(ednodes))
bottom_edge_bound = min(blayer(ednodes))
```

The design note calls the second quantity `ulayer_edge` even though it is computed from `blayer`. Treat this as an unresolved naming inconsistency; confirm the intended code name before propagating it.

### R5 - Velocities touching topography

Any velocity degree of freedom located in a partly-land prism / at a land corner must be zero or excluded from assembly/update. Prefer making the valid loop range express this invariant instead of adding scattered conditionals.

### R6 - Area arrays remain unchanged in the first pass

Horizontal areas of scalar volumes remain depth-independent (`area(1:myDim+eDim)` in the note). `elem_area` remains unchanged because prisms intersecting land do not contribute once their velocities are excluded.

### R7 - Edge geometry changes

Update edge geometry so that:

- `edge_dxdy` is stored in **physical measure**, not radian measure.
- the conversion uses the mean cosine associated with the element(s) adjacent to the edge: two elements for an interior edge, one when only one is available.
- introduce `edge_len` containing edge length in meters.

Do not guess the full conversion formula. Find the existing radian-based construction, Earth-radius/metric helpers, and coordinate convention; reuse the repository's established geometry utilities and add unit checks.

### R8 - Boundary edge adjacency convention

Rename/replace `edge_tri` with `edge_elem` where the current repository still uses the former concept.

For a boundary edge with only one adjacent element:

- set both `edge_elem` slots to the existing element rather than using `0`/sentinel for the missing second element;
- set `edge_cross_dxdy(3:4, edge) = 0.0` (adapt spelling/index ordering to actual code).

This convention is intended to remove repeated boundary `if` branches.

### R9 - Avoid duplicate-index vector updates

Because a boundary edge can now have `edge_elem(1,edge) == edge_elem(2,edge)`, code shaped like:

```fortran
el = edge_elem(:, edge)
U_rhs(nz, el) = ...
```

must not rely on vector/subscript semantics that can mishandle duplicate indices. Replace affected updates with an explicit loop over the adjacent-element slots or an equivalent duplicate-safe operation.

Audit **all** reads-modify-writes that use both edge-adjacent element indices, not just the example in the note.

### R10 - Boundary element-neighbor convention

For element-neighbor arrays, replace missing boundary neighbors with the element itself. Audit consumers for duplicate/self-neighbor safety just as for `edge_elem`.

This convention should yield no-slip treatment at required boundaries without repeated missing-neighbor branches.

### R11 - Bottom drag location

Apply bottom drag at the bottom of the last **fully wet** element cell, i.e. the bottom associated with `blayer_elem`.

### R12 - Stiffness-matrix vertical integration

Where stiffness assembly vertically integrates over elements, use only the full-element interval, from the top of `tlayer_elem` through the bottom of `blayer_elem`, respecting the repository's layer/level indexing convention.

### R13 - Deferred physics

- Horizontal diffusion: explicitly out of scope for now.
- Redi: requires later adjustment; do not fabricate a solution.
- Boundary scalar-cell area enlargement: optional/deferred.

## Ambiguities that must be resolved before or during coding

1. **Levels vs layers.** The note itself suggests switching terminology/indexing to layers because existing code often subtracts one from number of levels. Determine the repository convention and choose one consistent representation for the new arrays and loop bounds.
2. **`ulayer_edge` vs `blayer_edge`.** The note writes `ulayer_edge=min(blayer(ednodes))`. Do not silently rename without confirming intent from surrounding code or maintainer guidance.
3. **Formula punctuation in the note.** The printed `max/min` element formulas omit a closing parenthesis typographically. Interpret them as reductions over all element nodes, but do not copy the malformed syntax into code.
4. **`helem` / `hedge`.** Decide cache-vs-derived-on-demand only after measuring usage patterns and seeing existing data structures.
5. **Physical `edge_dxdy` formula.** The note specifies units and cosine treatment but not the complete formula. Derive implementation from existing geometry code and conventions, not from guesswork.
6. **Input compatibility.** Determine whether current meshes carry element depths, node depths, vertical bounds, or a mixture; define a migration/fallback path only if supported by existing code/tests.

## Execution protocol

### Phase 0 - Read repository instructions and establish baseline

Before editing:

```bash
pwd
git status --short
find .. -name AGENTS.md -o -name CONTRIBUTING.md -o -name README.md | head -50
```

Read the repository-local agent/build/test instructions that apply to files you may touch.

Identify the build system and run the smallest relevant existing test/smoke target before changes. Record the baseline, including pre-existing failures.

### Phase 1 - Build a symbol/use map

Search broadly; do not assume file names:

```bash
rg -n "edge_tri|edge_elem|edge_dxdy|edge_cross_dxdy|edge_len" .
rg -n "elem_area|area\(|neighbor|neighbour|elnodes|ednodes" .
rg -n "bottom drag|drag|stiffness|Redi|horizontal diffusion" .
rg -n "level|layer|nlevels|nlayers|depth|topograph|bathym" .
rg -n "tlayer|blayer|hnode|helem|hedge" .
```

Create a short internal map before edits:

| Concern | Producer | Stored in | Main consumers | Tests |
|---|---|---|---|---|
| node depth / vertical grid | | | | |
| element vertical bounds | | | | |
| edge vertical bounds | | | | |
| element thickness | | | | |
| edge thickness | | | | |
| edge adjacency | | | | |
| element neighbors | | | | |
| edge geometry | | | | |
| velocity loops | | | | |
| bottom drag | | | | |
| stiffness assembly | | | | |

Do not start broad replacements until this map is complete.

### Phase 2 - Decide the vertical indexing contract

Document in code/tests:

- what an integer in `tlayer`/`blayer` denotes;
- whether bounds are inclusive;
- whether thickness index `k` represents layer `k`, interval between levels `k:k+1`, or another convention;
- valid empty-column representation;
- valid fully wet element/edge interval;
- how `tlayer_elem > blayer_elem` (or equivalent) is handled.

Prefer layer indices if that matches dominant repository usage, but preserve compatibility where required.

Add low-cost debug/assertion checks if the project supports them:

```text
1 <= tlayer(v) <= blayer(v) <= n_layers
hnode(k,v) >= 0
sum(hnode(:,v)) consistent with wet-column depth within tolerance
```

Adjust exact inequalities for the actual indexing scheme.

### Phase 3 - Add node-level vertical data

Implement `tlayer`, `blayer`, and `hnode` in the mesh/vertical-grid data structure that owns topography.

Requirements:

- initialize all entries deterministically;
- derive them from existing depth data when not supplied by mesh input;
- if mesh-supplied values are supported, validate dimensions/ranges;
- keep one authoritative representation for thickness (`hnode`);
- update allocation, initialization, deallocation/finalization, restart/serialization code only where the repository requires it.

Do not add new on-disk fields unless necessary. If a file-format change is required, isolate it and document backward compatibility.

### Phase 4 - Derive element and edge wet bounds

Provide small, centralized helpers or construction loops for the reductions:

```fortran
! Pseudocode only; adapt to project syntax and indexing.
tlayer_elem(e) = maxval(tlayer(elnodes(:,e)))
blayer_elem(e) = minval(blayer(elnodes(:,e)))

tlayer_edge(edge) = maxval(tlayer(ednodes(:,edge)))
blayer_edge(edge) = minval(blayer(ednodes(:,edge)))
```

Avoid recomputing these reductions in many hot loops if the code repeatedly needs them; if cached, define ownership and refresh rules clearly.

Add tests with intentionally different bottom depths at the nodes of one triangle/edge.

### Phase 5 - Restrict velocity/edge loops to full wet prisms

Audit every vertically indexed velocity kernel/assembly loop that operates on elements or edges.

For element-centered/prism work, iterate only over the element-common wet interval. For edge work, iterate only over the edge-common wet interval.

Do not merely zero the final field after computing invalid cells if those cells can contaminate tendencies, matrices, reductions, CFL estimates, diagnostics, or halo exchange. Prefer preventing invalid contributions at the producer.

Where arrays still include storage for invalid/partly-wet cells, initialize or explicitly keep them at zero and test that invariant.

### Phase 6 - Thickness propagation

Make `hnode` authoritative. For every location that currently assumes one element/edge thickness:

1. determine whether the computation needs a nodal value, an element-derived value, an edge-derived value, or the integrated common-prism thickness;
2. derive it from `hnode` using the scheme already implied by the governing discretization/current code;
3. do not average/min/max thickness ad hoc unless repository mathematics or the note specifies it.

If `helem`/`hedge` are introduced, add a test that recomputes them from `hnode` and compares exactly/within tolerance.

### Phase 7 - Edge geometry and topology migration

Implement in a separate, reviewable change:

- physical-unit `edge_dxdy`;
- `edge_len` in meters;
- `edge_tri` -> `edge_elem` migration where applicable;
- boundary `edge_elem(:,edge) = [e,e]` convention;
- boundary `edge_cross_dxdy(3:4,edge)=0` convention.

Then audit duplicate-index-sensitive consumers:

```bash
rg -n "edge_elem\(|edge_tri\(" .
rg -n "[:,][[:space:]]*el\)|el\)" .
```

The second search is only a hint; inspect assignments and accumulations manually. Pay special attention to vector subscript writes, increments, atomics, gathers/scatters, and matrix assembly.

### Phase 8 - Element-neighbor boundary convention

Change missing boundary neighbors to self-neighbors in the producer of the neighbor table.

Audit consumers for assumptions such as:

- `neighbor == 0` means boundary;
- `neighbor < 1` means boundary;
- a neighbor pair is always distinct;
- vector updates over all neighbors cannot contain duplicates.

Replace these assumptions with explicit boundary metadata only where still needed; otherwise exploit the self-neighbor convention.

### Phase 9 - Bottom drag

Locate bottom-drag application and change the vertical location to the bottom of the last full element cell (`blayer_elem`).

Tests must include a sloping-bottom triangle whose three node columns have different `blayer` values. Verify drag is applied once, at the common full-cell bottom, and not in the partly wet region below it.

### Phase 10 - Stiffness assembly

Restrict vertical integration for element stiffness assembly to the full interval defined by `tlayer_elem` and `blayer_elem`.

Check:

- integration thickness;
- index bounds;
- matrix sparsity/assembly with boundary/self adjacency;
- duplicate accumulation behavior;
- behavior when the common full interval is empty.

### Phase 11 - Deferred areas / diffusion / Redi

Do not enable artificial boundary scalar-cell area enlargement in the first implementation. Add a local TODO/reference only if there is an obvious future hook.

Keep horizontal diffusion disabled/unmodified for this feature as required by the note. Mark Redi-dependent tests/paths clearly if they cannot yet be made compatible without the missing design.

## Test plan

Add the smallest deterministic tests at the lowest practical layer, plus one integration/regression case.

### T1 - Flat-bottom compatibility

All three triangle nodes have identical `tlayer`, `blayer`, and `hnode`.

Expected:

- `tlayer_elem`/`blayer_elem` equal the node values;
- edge bounds equal node values;
- active velocity range matches legacy behavior;
- existing flat-bottom result remains unchanged within established tolerance.

### T2 - Sloping triangle / common prism

Construct one triangle with different node bottom layers, e.g. conceptual values `blayer=[8,6,5]`.

Expected:

- `blayer_elem = 5` under an inclusive layer convention (adapt to actual indexing);
- velocity work only occurs in the common fully wet layers;
- storage below the common range stays zero / produces no tendency.

### T3 - Sloping edge

Two edge vertices have different wet bounds.

Expected edge range is the intersection of the two vertex columns.

### T4 - Boundary edge adjacency

For an edge with one adjacent element `e`:

- `edge_elem(:,edge) == [e,e]`;
- boundary cross terms are zero;
- duplicate-safe consumer updates do not double-apply a contribution.

### T5 - Boundary element neighbors

Missing neighbors become self-neighbors and no code path still dereferences a zero/sentinel as an element.

### T6 - `hnode` consistency

For representative columns:

- nonnegative cell thicknesses;
- dry cells have expected zero/unused representation;
- column thickness sum matches vertical extent/depth according to existing grid definition.

### T7 - Geometry units

Use a tiny mesh with known/independently checkable distance scale.

Expected:

- `edge_len` is in meters;
- `edge_dxdy` magnitudes are physical, not radian-scale;
- boundary and interior cosine handling follow the implemented geometry convention.

### T8 - Bottom drag placement

Verify drag acts at `blayer_elem`, not each node's deeper partial cell and not a legacy element-bottom index.

### T9 - Stiffness vertical interval

Compare assembled contribution against a hand-sized case where only a known number of full layers are common to all three nodes.

### T10 - No invalid velocity leakage

Seed invalid/partly-wet storage with a recognizable nonzero value in a test/debug setup and prove it cannot affect tendencies/assembly/diagnostics, or assert such storage is forcibly zero before use.

### T11 - Existing regression suite

Run the smallest relevant FESOM3 regression cases, then the broader project-prescribed suite if feasible. Report numerical diffs, not only pass/fail.

## Suggested implementation order / patch boundaries

Keep changes reviewable. A good sequence is:

1. **Vertical data model:** `tlayer`, `blayer`, `hnode`, indexing contract, unit tests.
2. **Derived wet bounds:** element/edge intersection bounds and tests.
3. **Topology boundary conventions:** `edge_elem`, self-neighbors, duplicate-safe updates.
4. **Geometry:** physical `edge_dxdy`, `edge_len`, unit tests.
5. **Velocity/edge loop bounds:** exclude partially wet prisms.
6. **Physics consumers:** bottom drag and stiffness integration.
7. **Regression/cleanup:** assertions, comments, docs, compatibility notes; no unrelated refactor.

Do not create commits unless the execution environment/user explicitly requests commits. If commits are allowed, make one logical commit per patch boundary above.

## Code-quality rules

- Follow repository naming and Fortran style; source-note names are conceptual unless they already match code.
- Prefer one central computation of wet bounds over repeated bespoke `max/min` logic.
- Prefer loop-bound invariants over scattered topography `if` statements.
- Preserve parallel/MPI/OpenMP semantics; audit halo exchange and local/global indexing for every newly stored mesh array.
- Initialize new arrays on all ranks and include them in halo/repartition handling when required.
- Avoid hidden unit changes: document physical units next to `edge_dxdy` and `edge_len` definitions.
- Any new sentinel/self-neighbor convention must be documented at the producer, not only at consumers.
- Do not suppress compiler warnings caused by the change; fix them.
- Keep new checks cheap in release code or behind the project's debug/assertion mechanism.

## Acceptance criteria / definition of done

The task is complete only when all applicable points below are true:

- Vertex columns have validated `tlayer`/`blayer` and scalar-cell `hnode` data.
- Full element and edge vertical intervals are computed as intersections of their vertex wet columns.
- Velocity/edge computations make no contribution from partly-land prisms.
- Element/edge thickness use is derived from `hnode` or a provably consistent cache.
- `elem_area`/scalar horizontal areas are not changed by this first implementation.
- `edge_dxdy` uses physical units and `edge_len` is available in meters.
- Boundary `edge_elem` uses self duplication and boundary cross terms are zero.
- Missing element neighbors use self-neighbors and consumers are safe with duplicates.
- Bottom drag is applied at the bottom of `blayer_elem`.
- Stiffness vertical integration uses only the full element interval.
- Horizontal diffusion changes and optional boundary-area inflation remain out of scope.
- Redi is not guessed; any incompatibility is explicitly documented.
- New focused tests pass.
- Existing relevant regression tests pass or all differences are explained and directly attributable to the intended scheme.
- `git diff` contains no unrelated changes.

## Required final report from the coding agent

When finished, report concisely:

1. files changed and why;
2. resolved layer/level indexing contract;
3. how `tlayer`, `blayer`, `hnode`, element bounds, and edge bounds are represented;
4. how boundary `edge_elem` and self-neighbors are handled safely;
5. exact physical-unit change for `edge_dxdy` and construction of `edge_len`;
6. bottom-drag and stiffness changes;
7. tests run and results;
8. any unresolved ambiguity, especially Redi or mesh-format compatibility;
9. any intentionally deferred work.

## Stop conditions

Stop and ask for a targeted decision instead of guessing if any of the following cannot be resolved from code/tests:

- whether `tlayer`/`blayer` index layers or levels and the choice changes physics;
- the intended meaning/name of `ulayer_edge`;
- the exact physical conversion required for `edge_dxdy`;
- how nodal depth is obtained for legacy meshes;
- a Redi change is required for tests to pass but is not specified;
- a repository-wide file-format migration would be required.

---

## Appendix A - Cleaned transcript of the design note

The following is a text extraction of the supplied PDF with page numbers, line-wrap hyphenation, and visual line breaks removed. Technical wording and ambiguities are intentionally preserved rather than silently corrected.

### Bottom implementation for FESOM3

**September 10, 2026**

### 1 Introduction

In FESOM2, the bottom depth is defined at elements (triangles), but the elevation is at vertices. The drawback of this representation is that a scalar cell has its bottom at several levels. This complicates the implementation of surface and bottom exchange processes (i. e. ice sheets or resuspension of sediments). It is therefore desirable to have a model with the bottom specified at vertices and related to scalar control volumes.

The implication of this scheme is that velocities touching topography should be set to zero.

### 2 Implementation

We used levels, but it would be perhaps more sound to switch to layers, as in most cases we are using them, and are subtracting one from the number of levels.

We introduce `tlayer` and `blayer`. They are vertex arrays describing the position of the top and bottom water layers for each vertex column. They are determined based on the depth data in the code, or computed outside and supplied with mesh files. In addition, we introduce `hnode` with the thickness (vertical size) of all scalar cells. Thicknesses at elements or edges of the mesh are enslaved to nodal thicknesses. Three nodes of mesh triangle may have different depths in this scheme. This means that a full triangular prisms will exist only between `tlayer_elem=max(tlayer(elnodes)` and `blayer_elem=min(blayer(elnodes)`, where `elnodes` are the vertices of triangle. Triangular prisms outside these layers will be partly occupied with land. Velocities in such prisms will be located at land corners and should be therefore set to zero. One should take care about this. As a result, a cycle over velocities is limited only to full prisms.

A cycle over edges should be limited to layers between `tlayer_edge=max(tlayer(ednodes))` and `ulayer_edge=min(blayer(ednodes))`, where `ednodes` are the vertices of edge.

We can introduce `helem` and `hedge` for convenience, but their entries are always related to `hnode`, and can be calculated in an elementary way. This should be decided.

### 3 Other mesh modification

Horizontal areas of scalar volumes do not change with depth, the array is `area(1:myDim+eDim)`. While areas of triangular prisms may vary with depth, we are not interested in the prisms that include land, as velocity is zero there and they do not contribute. Therefore, the array `elem_area` remains without changes.

In FESOM2, the array `edge_dxdy` was stored in radian measure. We will store it now in physical measure, using mean cosine of two elements on both sides of the edge (or one element if the second is not available). In addition, we introduce the array `edge_len` storing the length of edges in meters.

The name of array `edge_tri` will be replaced by `edge_elem`. In contrast to FESOM2, in the cases when the second element is absent, we put the first one instead of zero. Simultaneously, we will ensure that `edge_cross_dxdy(3:4,edge)=0.0` for boundary edges. This will allow to eliminate multiple if-statements. However, this will require that in the code in instances such as

```fortran
el=edge_elem(:,edge)
U_rhs(nz,el)=...
```

the second line is replaced by the explicit cycle.

The array of element neighbors is filled using the same agreement: missing neighbors for boundary elements are substituted by the element itself, with a similar requirement for coding. This will be automatically offering no-slip treatment where needed.

### 4 Other modifications compared to FESOM2

- Bottom drag will be applied at the bottom of the last full cell, i.e. at the bottom of `blayer_elem`.
- In the assembly of the stiffness matrix, the vertical integration is over elements, which means that the thickness will be from the top of `tlayer_elem` to the bottom of `blayer_elem`.
- Scalar cells around boundary vertices will generally have a smaller area (same as in the current FESOM). However, we can artificially increase their area, thinking that they are full quasi-hexagonal cells that intrude into the land. This will reduce vertical velocities at such locations. Effectively, this presents a simple way to implement the bottom assuming that the lateral walls are drawn through the velocity points (the velocities at these points are zero, and need not be considered). This can be delayed for the next step or dropped altogether.
- **Horizontal diffusion and Redi implementation:** We ignore horizontal diffusion for a while, and we need some adjustments for the Redi part. I will work on them.
