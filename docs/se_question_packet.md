# Split-explicit barotropic solver in FESOM_F: status and questions for Sergey and Dmitry

Written 2026-10-18; Nikolay relayed the questions to Sergey, whose answers are summarised in section 6. Branch `split-explicit`; plan `docs/plans/completed/2026-10-17-split-explicit.md`
(all numbers below are recorded there with job ids); design summary in `IMPLEMENTATION.md` §14.

## 1. What is implemented

FESOM_F is the Fortran reimplementation of FESOM2 with the bottom at vertices (an element is wet only
where all its vertices are). The split-explicit (SE) solver replaces the semi-implicit CG solve for one
step block; the predictor and the ALE/tracer tail are unchanged. Off by default.

- **Scheme:** the AB3–AM4 subcycling of Sergey's notes (Shchepetkin & McWilliams 2005; Banerjee et al.
  2024), β = 0.281105, ζ, γ, δ as functions of χ, M substeps of dtbt = τ/M. No averaging filter:
  ⟨⟨Ū⟩⟩ = Σ U*/M, η^{n+1} = η^M, and `eta_n` = `hbar` = η^M at integer levels. Defaults χ = 0.1
  (➕ since plan Task 5.8; 0.25 from Task 5.0 to Task 5.8 after the blow-up of section 3; see section 6), M = 50.
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

## 6. Answers

Sergey Danilov's answers of 2026-10-18 (relayed by Nikolay), summarised.

**Question 1.** The damping coefficient is chosen by experiment; his setups are stable at 0.2. Damping is better
than the averaging filter, which needs more than M substeps and costs performance. The barotropic stepping is best
built on the SESM scheme, which is more stable and should work with less dissipation.

- χ is tuned by experiment; his setups are stable at χ = 0.2. (FESOM_F default χ = 0.25; 0.2 not yet tested on
  CORE2.)
- Damping, not the SM2005 averaging filter: the filter needs more than M substeps and costs performance. This
  agrees with the no-filter choice (D1); the filter option is dropped.
- He recommends building the barotropic stepping on the "SESM" scheme (more stable, less dissipation needed).
  ⚠️ Open: which scheme "SESM" is (not in the notes, the Zenodo code or the literature search); ask for a
  reference.

**Question 2.** The sea ice should see η^M; the damping should keep the fast waves out of it. An averaged
elevation is what models with a filter use.

- The ice sees η^M, as implemented (hbar = η^M). The damping (χ) is what must keep the fast waves out of η^M;
  an averaged elevation belongs to models with a filter. So the filtered-η-for-the-ice option is dropped: on CORE2
  the shelf mode at χ = 0.1 is the case where the damping is too weak for that, and χ = 0.25 (or 0.2, his setups)
  is the fix.

**Question 3.** The bottom drag should go into the barotropic equation. The vertically integrated 3-D forcing is
made consistent with the barotropic equation approximately, by subtracting the barotropic momentum right-hand side
at the first substep (m = 1); the same can be done for Coriolis, which simplifies the scheme.

**Question 4.** Yes (η^{n+1} = η^M at integer levels, `eta_n = hbar = η^M`, as intended).

**Question 5.** It is simpler to keep the Coriolis term in r as in 3-D and make it consistent with 2-D as in
answer 3.

- Put the bottom drag into the barotropic substeps (as the Zenodo `se_bottdrag`: r_b = C_d·|u_bottom| from the
  3-D bottom velocity, frozen over the step, term −r_b·Ū/H in each substep; there optionally semi-implicit).
- Consistency of F with the substeps, approximately: subtract from the integrated 3-D forcing the barotropic
  right-hand side at the first substep (the drag, and the same for Coriolis). Coriolis stays in the 3-D AB2 array
  as it is; the stored-sum r-term (š, D4) is replaced by subtracting the barotropic Coriolis at the start of the
  step. ➕ Implemented and screened as plan Task 5.4 (commit 6a76b29, 2026-10-18); results below.

**➕ Results of answers 3 and 5 (plan Task 5.4, CORE2 production physics, cold start, M = 50, `se_visc_ah` 0.01).**
Implementation: the drag rate r/H = C_d|u_b|/Hⁿ from the 3-D bottom velocity, frozen over the step; each substep
adds −(r/H)(Ūᵐ − Ūⁿ), so the 3-D drag stays in F; F = G/τ + g Hⁿ∇ηⁿ + f e_z×U*⁰ with U*⁰ the AB3 transport of the
first substep (the stored sums š removed). All consistency tests and the restart gates pass.

| r-term for Coriolis | drag | χ = 0.1 | χ = 0.2 | χ = 0.25 |
|---|---|---|---|---|
| stored sums š (before) | frozen in F only | +0.91 %/step, blows up at step 340 | — | −0.36 %/step, stable 3000 steps |
| f e_z×U*⁰ (answer 5) | no | blows up at step 109 | — | — |
| f e_z×U*⁰ (answer 5) | yes (answer 3) | blows up at step 171 | blows up at step 243 | blows up at step 313 |
| f e_z×Ūⁿ | yes | — | — | blows up at step 336 |
| stored sums š | yes (answer 3) | −0.34 / −0.51 %/step (steps 150–330 / 330–480); stable 3000 steps | −0.29 / −0.26 | −0.30 / −0.19 |

(np 8, 480 steps, unless noted; growth of the grid-scale noise on the 20–40 m Antarctic shelf.) So the barotropic
drag stabilises χ = 0.1, with a faster decay than the flow-dependent viscosity, and the first-substep consistency
for Coriolis destabilises at every χ. A likely reason: without a filter, the trimmed 3-D velocity still carries the
fast shelf mode (period 3–5 steps), whose 3-D AB2 Coriolis is far off at that frequency; the stored sums removed it
exactly, the first-substep value does not, and the difference stays in F frozen over the step. Question for Sergey:
is the first-substep consistency for Coriolis meant for a scheme with the averaging filter, or should another 2-D
state (for example the averaged ⟨⟨Ū⟩⟩) enter it?

**Follow-up question 4 (what SESM is).** SESM is the AB3–AM4 scheme of Shchepetkin and McWilliams (2005), modified
by the dissipation of Demange et al. (2019) and Banerjee et al. (2024), eqs. 9 and 10; the parameter values of those
papers should be used. (Nikolay: the notes are Sergey's note for port_kokkos.)

- SESM is the scheme FESOM_F implements. Banerjee et al. (2024, GMD 17, 7051) eqs. (9)–(10) are the AB3/AM4
  extrapolations and the substep equations of `oce_ssh_se`. Their parameters: β = 0.281105 (largest stability
  limit), the dissipation condition 2χ = δ − γ − 2ζ − 1/2 (eq. 26), and the set without third-order phase accuracy
  ζ ≈ 0.010 − 0.135χ, γ = 0.083 − 0.514χ (eq. 28, the set of their Fig. 1). FESOM_F uses this set (the notes give
  it to 8 digits). The phase-accurate set of eq. 28 (ζ ≈ −0.123χ + 0.223 − 0.169β − 0.169χ², γ = 1/3 − β − 3ζ − χ²)
  is not used; the paper says the stability limit is much higher without that constraint.
- The χ of the papers: Banerjee et al. run the forward–backward scheme with θ = 0.14 (Demange et al.'s value) and
  compare the schemes in Fig. 1 at "the same dissipation χ = 0.07" (2χ = θ for that scheme, eq. 26). The notes say
  χ ≈ 0.05–0.1. FESOM_F's default χ = 0.25 is 3.6 times 0.07. ⚠️ Demange et al. (2019, JCP 398, 108875) not read:
  no open copy was reachable.

On the mechanism proposed for the Task 5.4 blow-ups: correct, but it should not cause a problem if the 3-D
velocities are made consistent with the barotropic ones before the next 3-D step starts.

- FESOM_F trims the 3-D velocities once, to ⟨⟨Ū⟩⟩^{n+1/2} (notes eq. 4_13, Banerjee eq. 13), and carries them so
  into the next step. ⟨⟨Ū⟩⟩ is required for the volume and tracer consistency, but it is the step average of the
  substep transports, not the barotropic transport at n+1/2. Banerjee et al. §2.4 (footnotes 3–4) describe the
  step FESOM_F lacks: after the tracers are advanced, re-trim the 3-D transports to the barotropic transport at
  n+1/2 (Ū at m = M/2, or (⟨Ū⟩ⁿ + ⟨Ū⟩ⁿ⁺¹)/2). They found it redundant in practice, with the forward–backward
  scheme and the Coriolis term omitted from R̄. Without it, the barotropic part of the 3-D velocity, which the 3-D
  AB2 Coriolis in R acts on, carries the step-averaged fast mode, while the first-substep subtraction uses the
  instantaneous Ū; with the stored sums (D4) the two cancel exactly whatever the 3-D velocity holds.
- ➕ Next test (not started): re-trim the 3-D velocities to Ū^{n+1/2} after the tracers, then re-screen the
  first-substep Coriolis of answer 5 with the drag, at χ = 0.07 (the papers' value), 0.1 and 0.25. Superseded by
  the next answer: with the Coriolis kept out of the barotropic forcing (D4) the re-trim is not needed.

**Follow-up question 1 (the first-substep Coriolis).** That form is unstable. Subtract a stable estimate of the
barotropic Coriolis instead, for example an AB2 from the barotropic transports at the end of the previous step and
at step n, or keep everything as in Banerjee et al.; his suggestion was meant as the simpler option.

- The first-substep value f e_z×U*⁰ is expected to be unstable; Task 5.4's blow-ups agree.
- Two acceptable forms: (i) subtract an AB2 estimate of the barotropic Coriolis from the end-of-step barotropic
  transports, f e_z×[(3/2+ε)Ūⁿ − (1/2+ε)Ūⁿ⁻¹] (one more stored 2-D field); or (ii) keep the Coriolis term out of
  R̄ as in Banerjee et al. (2024, §2.1). FESOM_F's stored sums (D4) are (ii): they remove the vertically integrated
  3-D AB2 Coriolis exactly (linfs; O(Δh·Δu) under zstar). Answer 5 was meant as a simplification, not a
  requirement; in FESOM_F (i) costs the same storage as D4 and removes less. This points to D4 + the barotropic
  drag (answer 3), the variant screened stable for 3000 steps at χ = 0.1 and 0.25 in Task 5.4 (Nikolay decides).

**Follow-up questions 2 and 3.** Yes, acceptable, and the dissipation should be minimal; it can depend on the
mesh, so it is best a user parameter.

- Question 2: the barotropic drag in the substeps with the Coriolis term from the stored sums (D4) is acceptable.
- Question 3: use the smallest χ that is stable; it can depend on the mesh, so χ stays a user parameter
  (`FESOM3_SE_CHI`), set per mesh. The default should be the smallest χ that is stable on the reference mesh (CORE2),
  not a value chosen with a margin; the documentation should say it is mesh-dependent.

**➕ Results (plan Tasks 5.5–5.6, 2026-10-18).** D4 is restored with the barotropic drag (commit 55f717b; it
reproduces the screened D4 + drag runs line for line). With the drag the shelf noise decays at χ = 0.05 / 0.07 /
0.1 / 0.25 (480 steps: −0.12 / −0.27 / −0.34 / −0.30 %/step over steps 150–330), and χ = 0.05 runs 3000 steps
under linfs and zstar and a full linfs year (annual-mean SSH SE − SI rms 0.110 mm, max 2.45 mm; SI's own np 256 vs
128 spread 0.72 mm, 39 mm). But the zstar year at χ = 0.05 blew up on day 229: a two-step oscillation of max|η|,
growing ~15 % per step, probably at 26°W 5°N in the equatorial Atlantic, where both χ = 0.05 years had a July
anomaly of 30–40 mm and 1.6 °C at 85 m that χ = 0.25 did not have. So on CORE2 the smallest χ stable for a year
under zstar lies above 0.05; a zstar year at χ = 0.1 is the next test. The default χ is still 0.25.

**The original questions 5–9.**
- Question 5: they take the Coriolis term from uv and do the AB interpolation on the vertically integrated
  velocities, storing the result of the previous step. This is D4 as implemented (s = Σhelem·uv, AB2 a₂sⁿ + a₁sⁿ⁻¹, sⁿ⁻¹
  stored in `se_sbar_old` and in the restart), restored in Task 5.5.
- Questions 6 and 7: yes and yes. The vector-invariant momentum advection in R and the element depth H_e(η) through the
  zstar weights under the vertex bottom are accepted.
- Question 8: z*. Production with SE is z*; FESOM_F's `tools/prod_env.sh` is linfs, so z* gets its own twin
  (Task 5.6).
- Question 9: AB3–AM4, modified, the SESM above. The scheme is SESM, as implemented.

**➕ Results (plan Task 5.7, 2026-10-18): the zstar blow-up at χ = 0.05 located; the zstar year at χ = 0.1.** A
checkpoint replay (exact, line for line) put the event on one 30 m node of the Kara shelf (75.65°E 74.53°N, in the
Ob–Yenisei plume, under 75 % ice), not in the equatorial Atlantic: a grid-scale barotropic mode at the baroclinic
Nyquist frequency (η alternating step by step, period-2 part 5–6 cm on day 220, growing 0.5–1 %/step). From the
day-220 checkpoint, χ 0.07 grows it slowly for 11 days, χ 0.1 decays it at −0.3 … −0.8 %/step, χ 0.25 at −3.4 %/step;
without the drag or with M = 100 it blows up sooner. A one-mode model of the SM substeps with a force of ≈ 0.8 % of the
restoring force frozen from the step-start elevation reproduces all six runs; switching off the ice tilt or the
ice→ocean stress does not stop it, so the sea ice is not the drive. The likely source (an estimate, not yet tested by
a switch) is the zstar baroclinic pressure of the stratification that the mode's layer stretching lifts, computed at
the step start and frozen over the step (≈ 0.5 %). The zstar year at χ = 0.1 ran clean: annual-mean SSH SE − SI rms
0.106 mm, max 1.74 mm (linfs at χ 0.05: 0.110, 2.45 mm), the largest July anomaly, near 25°W 5°N, 25 mm (40 mm at
χ 0.05), global-mean SSH at rounding. So on CORE2 with the drag the smallest χ that passes a year under both linfs
(0.05 passed) and zstar (0.05 failed, 0.07 marginal in the replay) is 0.1; ➕ the default since plan Task 5.8.

**Sergey, 2026-10-18, on the Task 5.7 result** (relayed by Nikolay, summarised). The theory of Demange et al. (2019)
shows that the split-explicit method has an instability that the choice of χ removes, so χ must be large enough that
the instability does not occur. The semi-implicit method has strong built-in damping, and there the instability does
not arise. This matches Task 5.7: χ = 0.1 is the smallest χ at which the Kara-shelf mode decays, and the SI years have
no such mode. Still open from the same message: how SE models handle the baroclinic pressure frozen over the step (the
hypothesis above), and whether the χ-dependent July anomaly near 25°W 5°N is expected.

**➕ Results (plan Task 5.8, 2026-10-19): the default χ = 0.1.** The gates pass at the new default. The linfs year at
χ = 0.1 with the drag is clean: annual-mean SSH SE − SI rms 0.111 mm, max 1.58 mm (in the Gulf of Carpentaria, where
the χ = 0.25 year has its maximum); the July maximum near 25°W 5°N is 5.8 mm (34 mm at χ = 0.05, 6.1 mm at χ = 0.25
without the drag, 25 mm in the zstar year at χ = 0.1); the global salt and temperature drifts equal SI's to 8e-8 psu
and 2e-6 °C.
