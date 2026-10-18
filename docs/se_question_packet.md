# Split-explicit barotropic solver in FESOM_F: status and questions for Sergey and Dmitry

Draft, 2026-10-18, **not sent**. Branch `split-explicit`; plan `docs/plans/2026-10-17-split-explicit.md`
(all numbers below are recorded there with job ids); design summary in `IMPLEMENTATION.md` §14.

## 1. What is implemented

FESOM_F is the Fortran reimplementation of FESOM2 with the bottom at vertices (an element is wet only
where all its vertices are). The split-explicit (SE) solver replaces the semi-implicit CG solve for one
step block; the predictor and the ALE/tracer tail are unchanged. Off by default.

- **Scheme:** the AB3–AM4 subcycling of Sergey's notes (Shchepetkin & McWilliams 2005; Banerjee et al.
  2024), β = 0.281105, ζ, γ, δ as functions of χ, M substeps of dtbt = τ/M. No averaging filter:
  ⟨⟨Ū⟩⟩ = Σ U*/M, η^{n+1} = η^M, and `eta_n` = `hbar` = η^M at integer levels. Defaults χ = 0.25
  (0.1 until 2026-10-18, see section 3), M = 50.
- **Transports and forcing:** Ū = Σₖ helem·u, R = Σₖ helemⁿ·UV_rhs, where UV_rhs is the full 3-D
  predictor increment (`compute_vel_rhs` with the vector-invariant momentum advection and
  `rvo_upwind` 1, `viscosity_filter`, the implicit vertical viscosity and the explicit bottom drag).
  The r-term removal is F = R/τ + g Hⁿ ∇ηⁿ + f e_z×(a₂šⁿ + a₁š^{n−1}), with š = Σₖ helem·uv the
  stored sums: FESOM's AB2 array mixes Coriolis with momentum advection, so Coriolis is rebuilt from
  the sums (exact under linfs, second order under zstar).
- **Depth under the vertex bottom:** H_e(η) = Hⁿ_e + Σ_v elem_w·c_{e,v}·(η_v − ηⁿ_v),
  Hⁿ_e = Σₖ helemⁿ. linfs: c ≡ 0 (H fixed). zstar: c_{e,v} is the share of the vertex's Δη that the
  zstar stretch puts into the layers inside the element's column, from `zbar_3d_n` differences
  (c = 1 where the vertex column ends at most one level below the element's). Through the real zstar
  tail Σhelem^{n+1} = H_e(η^{n+1}) to 4.5e-12 m of 5650 m; H0 + η would miss by 0.24 m.
- **Trim:** uvₖ = u**ₖ − (Σⱼ helemⁿⱼ u**ⱼ − ⟨⟨Ū⟩⟩)/Hⁿ_e (weights helemⁿ, depth-uniform correction).
- **Barotropic viscosity:** the notes' linear harmonic form on Ū/Hⁿ, A_h = 0.01 m/s × edge length,
  applied to Ūᵐ − Ūⁿ only. No flow-dependent γ₀/γ₁ term, no barotropic bottom drag.
- **Freshwater:** −τW in the η equation under zstar; under linfs as virtual salt (no W in η).
- **CFL guard:** ω_max of the discrete 2-D operator by power iteration and the AB3–AM4 limit
  a_crit(χ) (1.853 / 1.579 / 1.329 at χ = 0 / 0.1 / 0.25), M_min = ⌈τω_max/(0.9·a_crit)⌉. CORE2 at
  Δt 1800 s: ω_max = 3.15e-2 s⁻¹, M_min = 40 at χ = 0.1, 48 at χ = 0.25. The resolution-based
  estimate 0.5·res/√(gH) (port_kokkos) gave 35, and M = 35 blew up at step 5: it underestimates
  ω_max by 3.3× on CORE2.
- **Parallel:** border elements are owned by several ranks. F is synced once per step from one owner;
  every per-substep element operator is a gather in fixed order, so the copies stay bitwise equal.
  Per substep one node exchange (η) and one element exchange (Ū).
- **Restart:** the rings (η^{−1/M}, η^{−2/M}, three Ū levels, š^{n−1}) are in the checkpoint; resume
  is exact.

## 2. What is gated

| check | result |
|---|---|
| SE off vs the pre-SE build (CORE2 linfs and zstar, production physics; pi conservation suite) | byte-identical |
| η compatibility, max over 1000 pi steps / 480 CORE2 steps | 6.3e-15 m / 7.2e-14 m |
| trim Σhelemⁿuv = ⟨⟨Ū⟩⟩ | 5.7e-13 / 3.2e-12 m²/s |
| mean sea level (net of freshwater) | 7.4e-16 / 2.1e-15 m |
| zstar layer sum, w(1), depth law (CORE2) | 1.3e-11 m, 4.3e-17 m/s, 8.2e-12 m |
| zstar heat / salt / volume drift (pi, 1000 steps) | ≤ 6.4e-15 / 1.6e-14 / 4.3e-14 |
| shared-element copy spreads of Ū, ⟨⟨Ū⟩⟩ (np 2, 8, 128) | exactly 0 |
| determinism, same np (CORE2 np 8, 480 steps) | bitwise |
| restart split vs straight (linfs, zstar; np 1, 2; np 2 → 8 round trip) | max\|Δ\| = 0 |
| dtbt convergence (pi, Gaussian bump) | order 2.0–2.3 |

The CORE2 runs at 480 steps use χ = 0.25 (section 3). **3000-step screens at χ = 0.25** (2026-10-18, CORE2
production physics, cold start, np 128; the defaults M = 50 and `se_visc_ah` 0.01, M = 48, `se_visc_ah` 0, zstar,
each against SI): all run 62.5 days stably. The cold-start transient peaks at 4.7 m (SI 2.0 m) and is over by step
~500; afterwards SE's max|η| stays 3–6 cm above SI's (2.19 vs 2.14 m in steps 501–750, 1.96 vs 1.93 m in
2751–3000), max CFL_z 1.60 vs 1.59, max|uv| 2.98 vs 2.97 m/s; the runs with and without the barotropic viscosity
agree to 1.3 mm in max|η|. Not done yet: a 1-year SE-vs-SI twin and the timing comparison. Timing seen so far
(CORE2, one node, np 128): SSH phase 12.9 ms/step with SE vs 9.1 with SI, loop 146 vs 144 ms/step.

**Transients.** SE resolves the cold-start external gravity waves that SI's θ = 1 damps: max|η| in the
first day 4.3 m (SE, independent of Δt) vs 1.7 m (SI at Δt 1800 s), reaching 3.2 m at Δt 300 s. The
extremes sit in the resonant macrotidal gulfs (head of the Bay of Bengal, Gulf of Khambhat, Gulf of
California) at the 20 m minimum depth. port_kokkos saw the same (3 m vs 2 m).

## 3. The blow-up at χ = 0.1 on CORE2

**Observation.** CORE2, production physics (JRA55-do, EVP sea ice, TKE, GM/Redi, vector-invariant
momentum), cold start 1958-01-01, χ = 0.1, M = 50, `se_visc_ah` = 0.01: floating overflow at step
337–340 (day 7) at np 8 and np 128 alike (linfs); zstar at step 331. Up to the last step every invariant
is at rounding and the copy spreads are 0, so the scheme stays consistent; the instability is dynamical.

**Where and what.** A cluster of 20 m deep shelf nodes on the Bellingshausen coast (72–75°W, 71–73°S)
under compact ice (a = 1, ~2.6 m). The shelf noise (|η − local element mean|) grows from step ~150 at
**+0.9 % per step** (0.31 m at step 150, 1.37 m at step 330); the probe nodes oscillate with a **period
of 3.0 baroclinic steps** (ωτ ≈ 2.1), a grid-scale gravity wave in 20 m of water. At step 337 η reaches
21 m and uv 55 m/s. SI at step 336 has a smooth −1.6 m there.

**Damping estimate.** The SM substeps damp this mode by about χ(ωτ)²/M ≈ 0.9 % per step (∝ 1/M);
SI's backward Euler reduces its amplitude to 1/√(1 + (ωτ)²) ≈ 0.43 per step.

**On/off runs** (diagnostic build, env switches, np 8, 480 steps, χ = 0.1, M = 50):

| run | change | result |
|---|---|---|
| R0 | none | blows up (steps 337–340) |
| R1 | ice→ocean stress zeroed | runs; shelf noise decays −0.45 %/step |
| R2 | no bottom drag | growth reduced to +0.1 … +0.6 %/step, then decays; max\|η\| 8.8 m elsewhere |
| R3 | ice sees η = 0 (no tilt force) | runs; noise decays −0.45 %/step |

**Margin runs** (same setup): χ = 0.25: decays −0.3 … −0.55 %/step; `se_visc_ah` = 0.1: decays
−0.05 … −0.65 %/step; M = 100: grows +0.9 … +1.6 %/step, blows up at step 190–192; `se_visc_ah` = 0:
blows up at step 298.

**Mechanism.** Before each ocean step the ice reads hbar = ηⁿ; the grid-scale tilt −m g∇ηⁿ drives
the compact ice; the ice→ocean stress enters F and is frozen over [tⁿ, tⁿ⁺¹]. The explicit bottom drag,
also frozen in F from the previous step's velocity, adds to the growth. The same coupling exists under
SI; only the damping of the mode differs.

**Toy model.** One SE mode plus a pressure-like force of relative size ε = 0.03 taken from the
elevation at the start of the step and frozen over it reproduces the pattern: χ = 0.1, M = 50:
+0.9 %/step at ωτ = 2.2; `se_visc_ah` = 0: +1.4 %; M = 100: +1.5 %; χ = 0.25: +0.07 %;
`se_visc_ah` = 0.1: −0.02 %.

## 4. Remedy options

| option | status |
|---|---|
| χ = 0.25 (the default since 2026-10-18) | measured: stable over 480 steps (linfs np 8 and np 128, zstar np 8) and over 3000 steps (linfs at M = 50 and 48, with and without the viscosity; zstar; np 128). Needs M ≥ 43 at Δt 1800 s (guard: 48), damps the resolved fast waves more (max\|η\| 4.70 vs 4.74 m at step 67) |
| `se_visc_ah` = 0.1 (linear viscosity ×10) | measured: stable over 480 steps (linfs np 8; max\|η\| 4.72 m); acts only on Ūᵐ − Ūⁿ, explicit limit far away (ν·dtbt/l² ≈ 4e-3) |
| a filtered η for the ice instead of η^M | untested; targets the loop directly (R3 removes the growth) |
| the SM2005 averaging filter for ⟨⟨Ū⟩⟩ and the η handed to the slow components | untested (the notes use none) |
| the flow-dependent γ₀/γ₁ viscosity (Zenodo / port_kokkos) | untested here |
| larger M, χ < 0.1, no viscosity | not remedies (measured worse) |

## 5. Questions

1. **Damping.** Which remedy do you recommend: a larger χ (0.25?), the linear barotropic viscosity at
   0.1 m/s, the γ₀/γ₁ flow-dependent viscosity, or the SM2005 averaging filter? We made χ = 0.25 the
   default on 2026-10-18; is it acceptable for production, given that it damps the resolved external
   waves more?
2. **The ice coupling.** Should the sea ice see a time-filtered elevation (for example the substep mean
   of η) instead of η^M? Is there a standard choice in SE models with sea ice?
3. **Bottom drag in F.** The explicit 3-D bottom drag enters the barotropic forcing frozen over the
   step. Should the barotropic substeps carry their own (linearised) bottom drag on Ū instead, as
   the Zenodo code's `se_bottdrag` option does?
4. **Integer levels.** Is η^{n+1} = η^M at integer levels with `eta_n = hbar = η^M` (no averaged
   elevation) what you intended for FESOM, also for the tracer tail and the ice?
5. **Coriolis in the r-term** from the stored sums š (because the AB2 array mixes Coriolis with
   momentum advection): acceptable, or should Coriolis be split out of the AB array?
6. **Vector-invariant momentum.** R = Σhelem·UV_rhs with `momadv_opt` = 1 and `rvo_upwind` 1 (Dmitry's
   setting): acceptable? The FESOM2.5 SE code allowed only the flux form.
7. **Depth law.** H_e(η) through the zstar weights c under the vertex bottom (section 1): do you agree,
   and linfs with H fixed at Σhelem⁰?
8. **linfs or zstar** for production with SE? Both are gated; zstar adds the freshwater term to η.
9. **Canonical scheme.** The notes' SM AB3–AM4 or the Zenodo forward-backward variant with
   semi-implicit Coriolis (still open from the port_kokkos packet)?
