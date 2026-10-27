# Split-explicit barotropic solver for FESOM_F — handoff for the brainstorm (2026-10-17)

Branch `split-explicit` (from main 814e5d0), worktree `/work/ab0995/a270088/fesom3_se`. Nothing implemented yet.
Workflow as for M9/M10: **brainstorm with Nikolay first, then the plan `docs/plans/YYYY-MM-DD-split-explicit.md`,
then implement.** This note is the grounded starting point, not the design.

## 1. Decision and constraints (Nikolay, 2026-10-17)

- Implement split-explicit (SE) subcycling of the barotropic (η, Ū) system in **FESOM_F first, on this branch**;
  the JAX port (`~/fesom3_jax`, stage B of its GPU speed plan) follows later, bitwise against this code.
- Follow **Sergey's notes**: `~/port_kokkos/ssh_sergey/subcycling.tex` (+ `.pdf`): the Shchepetkin–McWilliams
  (2005) AB3–AM4 dissipative subcycling, transports U_k = h_k u_k, R_k = h_k·(predictor increment), r-term removal,
  trimming, no filtering. Background: Banerjee et al. 2024, GMD 17, 7051.
- **linfs first** (Dmitry's production config). **zstar must be switchable later by flipping `which_ALE` alone**
  ("ты должен заложить возможность потом просто включить zstar туда"). zstar is designed in now, not bolted on
  later (section 5).
- **The bottom is at vertices in FESOM_F** (IMPLEMENTATION.md §13, LESSONS L53). The notes and the FESOM2 reference
  code assume FESOM2's bottom on elements. Every depth/thickness formula has to be re-derived for the vertex bottom
  (section 4).
- Not bitwise to the CG path (the notes say so). The CG path stays the default and must stay byte-identical.

## 2. Prior work to reuse

| what | where | use |
|---|---|---|
| Sergey's notes | `~/port_kokkos/ssh_sergey/subcycling.tex` | the spec |
| FESOM2.5 SE reference (Zenodo) | `~/port_kokkos/ssh_sergey/zenodo_se/fesom2-refactoring_addsshsubcycl/src/oce_ale_ssh_splitexpl_subcycl.F90` | routine layout (`compute_BT_rhs_SE_vtransp`, `compute_BT_step_SE_ale`, `update_trim_vel_ale_vtransp`, `compute_thickness_zstar`), namelist names (`use_ssh_se_subcycl`, `se_BTsteps`, `se_visc*`, `se_bottdrag`), the BT viscosity. ⚠️ It is the forward-backward (Demange) variant with semi-implicit Coriolis, not the notes' SM AB3–AM4; it has a linfs branch (no water flux into η: line ~1183) |
| port_kokkos M12 (C++/Kokkos SE, zstar, SM AB3–AM4, gated) | `~/port_kokkos/docs/SSH_SE_M12.md` (summary), `docs/plans/20260813-m12-split-explicit.md` (full gate record), `docs/report/M12_SERGEY_PACKET.md` (deviations D1–D4, measured answers, Sergey's feedback), `src/fesom_ssh_se.{h,cpp}` | measured defaults, the exactness rules, the gate suite |

Measured in port_kokkos (CORE2, dt = 1800 s):
- **χ = 0.1.** χ = 0.05 dies on real forcing by step ~300 (Antarctic-shelf setdown → vertical-CFL death); 0.15 gives
  the same result.
- **M = 40–50.** A CFL guard gave a minimum of 35. Sergey: "30 can work on uniform meshes, 50 is already a lot", so
  tune M per mesh.
- **Barotropic viscosity:** live and two-term, V[Ūᵐ] − V[Ūⁿ], with the Zenodo γ0/γ1 = 10/2750.
- **Results:** 1-year twin SE−SI annual-mean SSH rms 0.2 mm (field std 0.66 m). Transients are hotter than SI by
  physics: cold-start η peak ~3 m vs ~2 m.

## 3. Where SE plugs into the FESOM_F step (`src/step/mod_step_oce.F90`)

```
compute_vel_rhs → viscosity_filter → impl_vert_visc_ale        (:212-220)  predictor, KEPT: gives UV_rhs = u** − u^{n−1/2}
update_stiff_mat_ale (non-linfs) → compute_ssh_rhs_ale → solve_ssh_ale  (:229-233)  REPLACED by SE
update_vel → compute_hbar_ale → update_eta_n                    (:239-241)  REPLACED by trim + set η/hbar
(GM/Redi) → vert_vel_ale → solve_tracers_ale → update_thickness_ale  (:261-295)  KEPT, driven by hbar/hbar_old
```

- τR_k = helem_k · UV_rhs_k after `impl_vert_visc_ale` (the notes: "use the velocity rhs of the current code after the
  full predictor and multiply by h").
- The predictor already contains −g∇ηⁿ per layer, un-h-weighted (`oce_dyn_velrhs.F90:108`, `p_eta = g*eta_n`). The
  r-term removes it exactly only with H_e = Σ_k helem(k,e) (port_kokkos packet §3).
- SE ends with η^{n+1}; set `hbar_old = hbar; hbar = η^{n+1}; eta_n = η^{n+1}` (the Zenodo code does the same).
  `vert_vel_ale` / `update_thickness_ale` then do zstar's thickness redistribution from hbar − hbar_old unchanged, and
  linfs ignores it. port_kokkos reused its validated zstar tail in exactly this way.
- **Time-level shift:** η moves to integer steps (synced with tracers, the notes' first point). Consumers to check:
  - the ice reads `elevation = mesh%hbar` (`src/ice/mod_ice_dyn.F90:99`);
  - output means;
  - restart (`src/io/mod_io_restart.F90:294-301`: eta_n, d_eta, hbar, ssh_rhs_old). d_eta is the CG initial guess and
    is not needed under SE. Add the SE history: the η levels AM4 needs, the Ū levels AB3 needs, and anything the r-term
    Coriolis keeps (D4). Gate with the M10 split-vs-straight restart test.

## 4. Bottom at vertices: what changes against the notes / FESOM2 reference

Contract (IMPLEMENTATION.md §13):
- The vertex column is authoritative: layers `ulevels_nod2D(v)..nlevels_nod2D(v)-1`.
- Elements are derived: `nlevels(e) = min` over the corners, `ulevels(e) = max`.
- An element column is therefore never deeper than any of its vertex columns.
- The scalar cell is a straight prism with the full area at every wet layer.

Consequences:
1. **Element barotropic depth is the element column, H_e = Σ_k helem(k,e).** Do not use the reference's
   `hh = -zbar_e_bot + mean(hbar)` or node depths.
2. **zstar's K\* / H̃⁰ (notes: "stop before the fragmented bottom") is FESOM2-specific.** FESOM_F zstar stretches the
   whole vertex column except its bottom layer (`vert_vel_ale`, `oce_ale.F90` ~:433; `update_thickness_ale` rebuilds
   helem over the element's full range, LESSONS L53 item 3). Reuse that tail; do not port the notes' K\* formula.
3. **Under zstar, H_e(η) ≠ H0_e + η_e.** Part of each vertex's Δη goes into its layers below the element's bottom.
   The element's share is linear in η:

   ```
   H_e(η) = H0_e + Σ_v w_v c_{e,v} η_v,
   c_{e,v} = (stretched thickness of v's layers inside e's column) / H̃_v ≤ 1
   ```

   Use it for the subcycle H^{AM4}, and Σhelemⁿ for the r-term. Then H_e(ηⁿ) = Σhelemⁿ to rounding (the incremental
   zstar law telescopes, port_kokkos D2), so the operator and the r-term cancel exactly at η = ηⁿ. linfs: c ≡ 0, so
   H_e = Σhelem⁰ = const. **Confirm with Sergey** (question 3).
4. Partial cells, if on, are covered automatically by Σhelem.

## 5. zstar readiness: design requirements (so that zstar is a switch, not a project)

- **One function for the element barotropic depth.** Use it for the r-term (Σhelemⁿ) and the subcycle (H_e(η^{AM4})),
  with linfs as its c ≡ 0 case. No `if linfs` scattered through the subcycle loop.
- **Trim weights helem_kⁿ / Σhelemⁿ in both modes.** Trimming with hⁿ, not h^{n+1/2}, is port_kokkos D1: it keeps
  Σ_k uv·helem = ⟨⟨Ū⟩⟩ exact through the tracer chain, measured 3e-13 m²/s.
- **Freshwater in the η equation:** `− τ W` only when `which_ALE /= 'linfs'`, mirroring `compute_hbar_ale`
  (`oce_ale.F90` ~:253) and the Zenodo linfs branch. Under linfs the freshwater goes in as virtual salt, as now.
- **The thickness update stays in the existing zstar tail** (`vert_vel_ale` / `update_thickness_ale` via hbar/hbar_old).
  SE writes no thickness itself.
- **Not needed under SE:** `update_stiff_mat_ale`, the stiffness matrix and dhe. Leave them untouched for the SI path.
- **Gate zstar on the branch from the start**, cheap runs (section 7). That way "enable zstar" is already tested, not a
  promise.

## 6. Shared elements: keep the one-owner sync

- Under the vertex bottom, a border element is owned by every rank that owns one of its vertices.
- `update_vel` gives each such element one owner for `uv` (`exchange_elem_shared`, `oce_ale.F90:160-171`,
  `FESOM3_SHARED_SYNC`). Without that, roundoff differences between the copies (viscosity-filter sum order, amplified
  by the rvo_upwind blend) caused the linfs volume/salt leak of CORE2 results7/11/17/21.
- The SE trim replaces `update_vel`, so it must keep that sync.
- More: R̄ on shared elements already differs by roundoff between ranks, so the per-substep Ū and ⟨⟨Ū⟩⟩ would too. If
  the edge owner (η divergence) and the element owner (trim) see different ⟨⟨Ū⟩⟩, the same leak comes back at roundoff
  level. Decide in the brainstorm: a one-owner Ū per substep (one more shared exchange, like port_kokkos' "η node + Ū
  element" pair per substep) or a single sync of R̄ before the subcycles.
- Gate with the volume/salt invariants at np > 1 (section 7).

## 7. Gates (port_kokkos G0–G3 adapted to FESOM_F)

- **G0 knob off = byte-identical** to main: ctest Intel + GNU, `tools/run_conserve_pi.sh`, the restart gate.
- **G1 invariants at rounding, every step, linfs and zstar, np 1 and np > 1:**
  - η compatibility: η^{n+1} − ηⁿ + τ(∇·⟨⟨Ū⟩⟩ + W) = 0;
  - trim: Σ_k helem·uv = ⟨⟨Ū⟩⟩;
  - volume (linfs: Σ areasvol·η constant; zstar: change = freshwater);
  - zstar layer sum Σ hnode = H0 + η;
  - heat/salt content via the conservation gate.

  pi, then CORE2 short. port_kokkos levels: compatibility ≤ 2e-14 m, trim ≤ 3.4e-13 m²/s, layer sum 1.5e-11 m over 1000
  steps.
- **G2 determinism:** same binary and np → bitwise; np 1 vs np 8 at roundoff; restart split-vs-straight max|Δ| = 0 at
  np 1.
- **G3 physics:** CORE2 forced screens of 3000 steps (χ, M); then a 1-year SE-vs-SI twin (annual-mean SSH difference).
  NEMO after CORE2.
- **Perf, measured not guessed:** the SSH phase time SI vs SE on one CPU node (CORE2, NEMO). The main payoff is
  expected on the GPU (JAX: no global AllReduce per CG iteration), not on the CPU.

Cost: a CORE2 year on one CPU node is ~42 min (144 ms/step). G3 is a few node-hours.

## 8. Questions for Sergey / Dmitry (collect before or during the plan)

1. **`momadv_opt = 1` (vector-invariant, Dmitry's setting) with SE.** The notes' "multiply the rhs by h" covers it.
   The FESOM2.5 SE code allowed only `momadv_opt = 2` (with a transport variant `momentum_adv_scalar_transpv`). Is
   R_k = helem·UV_rhs acceptable with vinv?
2. **The Coriolis part of the r-term.** FESOM_F's AB2 array mixes Coriolis and advection (port_kokkos D4 rebuilt Ū_AB2
   from stored sums). Is that acceptable, or should Coriolis be split out of the AB array?
3. **The element barotropic depth under the vertex bottom** (section 4 item 3): H_e(η) via the zstar law vs H0+η.
4. **linfs: H = Σhelem⁰ (no η)** in the BT momentum. This is needed for exact r-term removal.
5. **Defaults:** χ = 0.1, M per mesh (CFL guard), the BT viscosity, BT bottom drag off for v1?
6. **SM AB3–AM4 (the notes) vs the Zenodo FB code with SI Coriolis:** which is canonical? This was port_kokkos packet
   question 2, still open.

## 9. Later JAX port: keep it cheap

- `jax-oracle` = main + 5 commits (gfortran oracle build, inert `FESOM3_JAXDUMP` dumps); main has no commits since. Keep the SE work as self-contained commits
  (one new module + small hooks in step/namelist/restart), so it rebases onto `jax-oracle` for the gfortran no-FMA
  oracle build.
- No order-dependent parallel sums, no atomics. SE has no global reductions.
- Add `FESOM3_JAXDUMP` stages for the SE call points when the JAX side starts.
- The JAX side ports linfs first.

## 10. Practicalities

- Knobs: follow FESOM_F's namelist + `FESOM3_*` env override pattern, e.g. `use_ssh_se_subcycl` / `FESOM3_SSH_SE`,
  `se_BTsteps`, `se_chi`, `se_visc*`. Reuse the Zenodo names where they exist.
- Build in this worktree (`configure.sh`, own build dirs; /work, not home).
- Runs via sbatch with `--hint=nomultithread` (and `--mem=0` for whole nodes); `env.sh` sets the knem workaround.
- Nothing is pushed without Nikolay's approval of the exact text. No session links or tool markers in commits.
