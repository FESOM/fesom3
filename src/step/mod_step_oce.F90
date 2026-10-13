module mod_step_oce
    ! M2.9b — the faithful ocean timestep sequence (FESOM2 oce_ale.F90:3530-4209
    ! `oce_timestep_ale`, plus the `compute_vel_nodes` that the main loop runs just
    ! before it — fesom_module.F90:654). This assembles the M2.1-M2.9a leaf kernels into
    ! one driver with LIVE data flow (each kernel reads the previous kernel's output, not
    ! a prescribed input) and emits the same per-substep validation dumps as FESOM2 so the
    ! byte-gate (tools/run_step_gate.sh) compares the WHOLE step substep-by-substep.
    !
    ! Sequence (reduced-M2: PP / no-GM / no-Redi / linfs / opt_visc=7):
    !   compute_vel_nodes     UV (elements) -> uvnode (nodes); PP reads it
    !   pressure_bv           EOS density + hydrostatic pressure + N^2 (smoothed)
    !   pressure_force_4_linfs hpressure -> pgf_x/pgf_y
    !   [sw_alpha_beta / compute_sigma_xy / compute_neutral_slope: DEAD in M2 (KPP/GM/Redi
    !    only) -> OMITTED; their producers return to the sequence at M4]
    !   oce_mixing_pp + mo_convect    Ri-number Kv (nodes) / Av (elements) + convective adj.
    !   compute_vel_rhs       Coriolis AB2 + PGF + SSH-grad + momentum advection -> UV_rhs
    !   viscosity_filter      biharmonic horizontal viscosity (opt_visc=7) += UV_rhs
    !   impl_vert_visc_ale    implicit vertical viscosity TDMA (Av now LIVE from PP) -> UV_rhs
    !   compute_ssh_rhs_ale + solve_ssh_ale   free-surface CG -> d_eta
    !   update_vel            UV += UV_rhs + [-g*theta*dt*grad(d_eta)]
    !   compute_hbar_ale      hbar += dt/areasvol * div(UV); dhe
    !   update_eta_n          eta_n = alpha*hbar + (1-alpha)*hbar_old
    !   vert_vel_ale          W = -cumsum(div(UV*h))/area; CFLz + Wvel split
    !   solve_tracers_ale     per tracer: advection + diffusion solve (Kv LIVE) + S-clamp
    !   update_thickness_ale  commit hnode=hnode_new (linfs: no-op)
    !
    ! D7 dependency injection: the inputs the core does not yet produce are explicit
    ! arguments — the tracer horizontal diffusivity Ki (needs mesh_resolution, M4), the
    ! surface fluxes heat_flux/water_flux/virtual_salt/relax_salt/real_salt_flux (forcing,
    ! M2.10) and the wind stress stress_surf (forcing, M2.10). When those features land the
    ! caller sources them (mesh / forcing) and step_oce is unchanged. Av/Kv are NO LONGER
    ! arguments — they are produced LIVE by oce_mixing_pp into dyn%work and consumed in place
    ! (the M2.9b wiring that M2.5/M2.8 deferred).
    !
    ! M2.12c: an OPTIONAL partit is threaded to every kernel. Absent (or npes==1) -> each
    ! kernel runs its proven 1-rank path VERBATIM (the 1-rank callers fesom_stepdump /
    ! fesom_lifecycle omit partit and are unchanged). Present + npes>1 -> the kernels use
    ! owned/halo loop bounds + the FESOM2 halo exchanges (M2.12c-1 pre-SSH dynamics, c-2 the
    ! SSH stiffness + free-surface CG, c-3 the post-SSH ALE update + tracer SOLVE). The
    ! per-substep dumps are gid-keyed (mod_dump), so the multi-rank whole-step driver
    ! (fesom_stepfull_mr) dumps the same per-rank owned probes the 1-rank gate compares.
    use mod_precision,      only: WP
    use mod_mesh,           only: t_mesh
    use mod_dyn,            only: t_dyn
    use mod_tracer,         only: t_tracer
    use mod_partit,         only: t_partit
    use mod_dump,           only: dump_node, dump_node_2d, &
                                  DUMP_SUBSTEP_PRESSURE_BV, DUMP_SUBSTEP_MIXING, &
                                  DUMP_SUBSTEP_SSH_RHS, DUMP_SUBSTEP_SSH_SOLVE, &
                                  DUMP_SUBSTEP_HBAR, DUMP_SUBSTEP_ETA_N, &
                                  DUMP_SUBSTEP_ALE, DUMP_SUBSTEP_TRACERS, &
                                  DUMP_SUBSTEP_THICKNESS
    use oce_pressure_bv,    only: pressure_bv, sw_alpha_beta, compute_sigma_xy, compute_neutral_slope
    use oce_fer_gm,         only: init_Redi_GM, fer_solve_Gamma, fer_gamma2vel
    use mod_param_phys,     only: Fer_GM, Redi, mix_scheme_nmb
    use mod_part_bounds,    only: owned_bounds
    use oce_pgf,            only: pressure_force_4_linfs_fullcell, pressure_force_4_zxxxx_shchepetkin
    use oce_ale_mixing_pp,  only: oce_mixing_pp
    use oce_mixing_kpp,     only: oce_mixing_kpp_driver
    use oce_mixing_tke,     only: calc_cvmix_tke
    use oce_mo_conv,        only: mo_convect
    use oce_vert_spline,    only: compute_shear2
    use oce_dyn_velrhs,     only: compute_vel_rhs
    use oce_dyn_visc,       only: viscosity_filter
    use oce_dyn_ivertvisc,  only: impl_vert_visc_ale
    use oce_ssh_rhs,        only: compute_ssh_rhs_ale, update_stiff_mat_ale
    use mod_config,         only: which_ALE
    use oce_ssh_solve,      only: solve_ssh_ale
    use oce_ale,            only: compute_vel_nodes, update_vel, compute_hbar_ale, &
                                  update_eta_n, vert_vel_ale, update_thickness_ale
    use oce_ale_tracer,     only: solve_tracers_ale
    use mod_timer,          only: timer_start, timer_stop, &
                                  TMR_OCE_MIXPRES, TMR_OCE_DYN, TMR_OCE_SSH, &
                                  TMR_OCE_SOLVE, TMR_OCE_GMREDI, TMR_OCE_TRACER
    implicit none
    private
    public :: step_oce

contains

    subroutine step_oce(n, dt, lfirst, dynamics, tracers, mesh, &
                        Ki, heat_flux, water_flux, virtual_salt, relax_salt, &
                        real_salt_flux, is_nonlinfs, stress_surf, partit, stress_node_surf)
        integer,        intent(in)            :: n        ! step number (dump key)
        real(kind=WP),  intent(in)            :: dt
        logical,        intent(in)            :: lfirst   ! first Euler step (ff=1.0)
        type(t_dyn),    intent(inout), target :: dynamics
        type(t_tracer), intent(inout), target :: tracers
        type(t_mesh),   intent(inout), target :: mesh
        ! external inputs not yet sourced by the core (M2.10 forcing / M4 resolution):
        real(kind=WP),  intent(in) :: Ki(mesh%nl-1, mesh%nod2D)
        real(kind=WP),  intent(in) :: heat_flux(mesh%nod2D), water_flux(mesh%nod2D)
        real(kind=WP),  intent(in) :: virtual_salt(mesh%nod2D), relax_salt(mesh%nod2D)
        real(kind=WP),  intent(in) :: real_salt_flux(mesh%nod2D), is_nonlinfs
        real(kind=WP),  intent(in) :: stress_surf(2, mesh%elem2D)
        ! M2.12c: optional multi-rank partition (absent => 1-rank verbatim).
        type(t_partit), intent(in), optional :: partit
        ! M5 KPP: the surface stress on NODES (oce_fluxes_mom output). Required when KPP
        ! (mix_scheme_nmb==1) -> ustar; absent ⇒ PP path, never read. Unforced ⇒ zero.
        real(kind=WP),  intent(in), optional :: stress_node_surf(2, mesh%nod2D)

        logical :: is_kpp, is_tke
        integer :: node, nNodO, nNodL, nEdgeO, nElemO

        is_kpp = (mix_scheme_nmb == 1)
        is_tke = (mix_scheme_nmb == 5)

        !_______________________________________________________________________
        ! nodal velocity (the REAL uvnode source, was prescribed at M2.8)
        call timer_start(TMR_OCE_MIXPRES)
        call compute_vel_nodes(dynamics, mesh, partit)

        !_______________________________________________________________________
        ! EOS density, hydrostatic pressure, N^2 (+ horizontal smoothing: the caller
        ! pins N2smth_h=.true.). Writes dyn%work%density_m_rho0/hpressure/bvfreq; the
        ! below-bottom rows keep their setup value (0), as in FESOM2 at n=1.
        ! KPP needs the surface-referenced buoyancy difference dbsfc (pressure_bv optional
        ! output, M5a-3); the PP path leaves it absent (byte-neutral). partit is positional
        ! arg 8 so dbsfc is passed by keyword.
        if (is_kpp) then
            call pressure_bv(tracers%data(1)%values, tracers%data(2)%values, &
                             dynamics%work%density_ref, mesh, &
                             dynamics%work%density_m_rho0, dynamics%work%hpressure, &
                             dynamics%work%bvfreq, partit, dbsfc=dynamics%work%dbsfc, &
                             bvfreq_raw=dynamics%work%bvfreq_raw)
        else
            call pressure_bv(tracers%data(1)%values, tracers%data(2)%values, &
                             dynamics%work%density_ref, mesh, &
                             dynamics%work%density_m_rho0, dynamics%work%hpressure, &
                             dynamics%work%bvfreq, partit, &
                             bvfreq_raw=dynamics%work%bvfreq_raw)
        end if
        call dump_node(DUMP_SUBSTEP_PRESSURE_BV, n, 'density',  dynamics%work%density_m_rho0, mesh%nlevels_nod2D)
        call dump_node(DUMP_SUBSTEP_PRESSURE_BV, n, 'pressure', dynamics%work%hpressure,      mesh%nlevels_nod2D)
        call dump_node(DUMP_SUBSTEP_PRESSURE_BV, n, 'bvfreq',   dynamics%work%bvfreq,         mesh%nlevels_nod2D)

        !_______________________________________________________________________
        ! hydrostatic pressure gradient force. linfs -> the hpressure-based full-cell PGF;
        ! non-linfs (zlevel/zstar) -> the self-contained Shchepetkin density-Jacobian PGF
        ! (which_pgf='shchepetkin', the default; M6a-1). FESOM2 oce_ale.F90:3656-3661.
        if (trim(which_ALE)=='linfs') then
            call pressure_force_4_linfs_fullcell(dynamics%work%hpressure, mesh, &
                                                 dynamics%work%pgf_x, dynamics%work%pgf_y, partit)
        else
            dynamics%work%pgf_x = 0.0_WP; dynamics%work%pgf_y = 0.0_WP   ! kernel writes ule..nle only
            call pressure_force_4_zxxxx_shchepetkin(dynamics%work%density_m_rho0, mesh, &
                                                 dynamics%work%pgf_x, dynamics%work%pgf_y, partit)
        end if

        call timer_stop(TMR_OCE_MIXPRES)

        !_______________________________________________________________________
        ! M4/M5 producers: sw_alpha_beta (EOS expansion coeffs) feeds KPP (Bo), the GM
        ! streamfunction (via sigma_xy) and, for Redi, compute_neutral_slope. FESOM2
        ! oce_ale.F90:3673-3682 calls it unconditionally; here it fires for KPP.or.GM.or.Redi
        ! (the reduced PP/no-GM step leaves sw_alpha untouched). sigma_xy/neutral_slope are
        ! GM/Redi-only (KPP does not read them) so they stay Fer_GM.or.Redi-guarded.
        call timer_start(TMR_OCE_GMREDI)
        if (Fer_GM .or. Redi .or. is_kpp) then
            call sw_alpha_beta(tracers%data(1)%values, tracers%data(2)%values, mesh, &
                               dynamics%work%sw_alpha, dynamics%work%sw_beta, partit)
        end if
        if (Fer_GM .or. Redi) then
            call compute_sigma_xy(tracers%data(1)%values, tracers%data(2)%values, &
                                  dynamics%work%sw_alpha, dynamics%work%sw_beta, mesh, &
                                  dynamics%work%sigma_xy, partit)
            if (Redi) call compute_neutral_slope(dynamics%work%sigma_xy, dynamics%work%bvfreq, mesh, &
                                  dynamics%work%neutral_slope, dynamics%work%slope_tapered, &
                                  dynamics%work%fer_tapfac, partit)
        end if
        call timer_stop(TMR_OCE_GMREDI)

        !_______________________________________________________________________
        ! vertical mixing + convective adjustment. FESOM2 oce_ale.F90:3713-3729 dispatches
        ! on mix_scheme_nmb. KPP (==1): the boundary-layer scheme writes Av (element momentum
        ! viscosity) + Kv_double (node T/S diffusivity); Av stays element-based so
        ! impl_vert_visc_ale is UNCHANGED, and Kv = Kv_double(:,:,1) (T channel) so the tracer
        ! TDMA is UNCHANGED (the same single Kv as PP). PP (else): Richardson-number Kv/Av.
        ! Both fill only interior levels; surface/bottom keep their setup 0. mo_convect runs after.
        ! TKE (mix_scheme_nmb==5): calc_cvmix_tke is the WHOLE producer — it does the Kv=tke_Kv
        ! and Av=avg(tke_Av) write-back INTERNALLY (oracle-faithful, unlike KPP's dispatch-level
        ! Kv=Kv_double loop), so the branch only calls it then mo_convect. Av stays element-based
        ! and Kv node-based -> impl_vert_visc_ale + the tracer TDMA are BYTE-UNCHANGED from M6.
        call timer_start(TMR_OCE_MIXPRES)
        ! ONE shear for all three schemes (they each built their own before). Must run
        ! after compute_vel_nodes (uvnode) and before any scheme reads dyn%work%shear2.
        call compute_shear2(dynamics, mesh, partit)
        if (is_kpp) then
            if (.not. present(stress_node_surf)) &
                error stop 'step_oce: KPP (mix_scheme_nmb==1) requires stress_node_surf'
            call oce_mixing_kpp_driver(dynamics, tracers, stress_node_surf, heat_flux, water_flux, mesh, partit)
            call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
            do node = 1, nNodL
                dynamics%work%Kv(:, node) = dynamics%work%Kv_double(:, node, 1)
            end do
            call mo_convect(dynamics, mesh, partit)
        else if (is_tke) then
            if (.not. present(stress_node_surf)) &
                error stop 'step_oce: TKE (mix_scheme_nmb==5) requires stress_node_surf'
            call calc_cvmix_tke(dynamics, stress_node_surf, dt, mesh, partit)
            call mo_convect(dynamics, mesh, partit)
        else
            call oce_mixing_pp(dynamics, mesh, partit)
            call mo_convect(dynamics, mesh, partit)
        end if
        call dump_node(DUMP_SUBSTEP_MIXING, n, 'Kv', dynamics%work%Kv, mesh%nlevels_nod2D)
        call timer_stop(TMR_OCE_MIXPRES)

        !_______________________________________________________________________
        ! momentum rhs: Coriolis AB2 + PGF + SSH-grad + momentum advection
        call timer_start(TMR_OCE_DYN)
        call compute_vel_rhs(dynamics, mesh, dt, lfirst, partit)

        !_______________________________________________________________________
        ! horizontal (biharmonic) viscosity
        call viscosity_filter(dynamics%opt_visc, dynamics, mesh, dt, partit)

        !_______________________________________________________________________
        ! implicit vertical viscosity TDMA (Av is now LIVE from PP mixing)
        call impl_vert_visc_ale(dynamics, mesh, dt, dynamics%work%Av, stress_surf, partit)
        call timer_stop(TMR_OCE_DYN)

        !_______________________________________________________________________
        ! free-surface solve. For non-linfs ALE (zlevel/zstar) the SSH stiffness 2nd term
        ! tracks the moving surface: update it by the (lagged) dhe from the previous step's
        ! compute_hbar_ale BEFORE assembling/solving (FESOM2 oce_ale.F90:3921; step-1 dhe=0).
        ! TMR_OCE_SOLVE (the CG solve) is nested inside TMR_OCE_SSH, mirroring FESOM2's "solve ssh".
        call timer_start(TMR_OCE_SSH)
        if (trim(which_ALE)/='linfs') call update_stiff_mat_ale(mesh, dt, partit)
        call compute_ssh_rhs_ale(dynamics, mesh, partit, water_flux=water_flux)
        call dump_node_2d(DUMP_SUBSTEP_SSH_RHS, n, 'ssh_rhs', dynamics%ssh_rhs)
        call timer_start(TMR_OCE_SOLVE)
        call solve_ssh_ale(dynamics, mesh, partit=partit)
        call timer_stop(TMR_OCE_SOLVE)
        call dump_node_2d(DUMP_SUBSTEP_SSH_SOLVE, n, 'd_eta', dynamics%d_eta)

        !_______________________________________________________________________
        ! velocity update + elevation
        call update_vel(dynamics, mesh, dt, partit)
        call compute_hbar_ale(dynamics, mesh, dt, partit, water_flux=water_flux)
        call dump_node_2d(DUMP_SUBSTEP_HBAR, n, 'hbar', mesh%hbar)
        call update_eta_n(dynamics, mesh, partit)
        call dump_node_2d(DUMP_SUBSTEP_ETA_N, n, 'eta_n', dynamics%eta_n)
        call timer_stop(TMR_OCE_SSH)

        !_______________________________________________________________________
        ! M4 GM diffusivity + streamfunction + bolus velocity (FESOM2 oce_ale.F90:4050-4058 —
        ! after the SSH/velocity/elevation update, before vert_vel_ale which then fills fer_w).
        ! Redi off (M4d). fer_uv feeds vert_vel_ale (fer_w) + the tracer bolus add/subtract.
        call timer_start(TMR_OCE_GMREDI)
        if (Fer_GM .or. Redi) then
            if (Redi) then
                call init_Redi_GM(mesh, dynamics%work%bvfreq, dynamics%work%fer_K, dynamics%work%fer_c, &
                                  dynamics%work%fer_scal, partit, dynamics%work%Ki, dynamics%work%fer_tapfac)
            else
                call init_Redi_GM(mesh, dynamics%work%bvfreq, dynamics%work%fer_K, &
                                  dynamics%work%fer_c, dynamics%work%fer_scal, partit)
            end if
        end if
        if (Fer_GM) then
            call fer_solve_Gamma(mesh, dynamics%work%sigma_xy, dynamics%work%bvfreq, &
                                 dynamics%work%fer_c, dynamics%work%fer_K, dynamics%work%fer_gamma, partit)
            call fer_gamma2vel(mesh, dynamics%work%fer_gamma, dynamics%fer_uv, partit)
        end if
        call timer_stop(TMR_OCE_GMREDI)

        !_______________________________________________________________________
        ! vertical velocity / ALE thickness (linfs: hnode_new = hnode)
        call timer_start(TMR_OCE_DYN)
        call vert_vel_ale(dynamics, mesh, dt, partit, water_flux=water_flux)
        call dump_node(DUMP_SUBSTEP_ALE, n, 'hnode_new', mesh%hnode_new, mesh%nlevels_nod2D)
        call dump_node(DUMP_SUBSTEP_ALE, n, 'w',         dynamics%w,     mesh%nlevels_nod2D)
        call timer_stop(TMR_OCE_DYN)

        !_______________________________________________________________________
        ! tracer solve (advection + diffusion; tracer TDMA consumes LIVE Kv)
        ! M4d Redi: the diffusion uses the computed Redi diffusivity dynamics%work%Ki (the Ki
        ! arg is the prescribed background, 0 in the reduced config). The diff routines' Redi
        ! terms are if(Redi)-guarded so the non-Redi path is byte-unchanged.
        call timer_start(TMR_OCE_TRACER)
        if (Redi) then
            call solve_tracers_ale(dt, dynamics, tracers, mesh, dynamics%work%Ki, &
                                   heat_flux, water_flux, virtual_salt, relax_salt, &
                                   real_salt_flux, is_nonlinfs, partit)
        else
            call solve_tracers_ale(dt, dynamics, tracers, mesh, Ki, &
                                   heat_flux, water_flux, virtual_salt, relax_salt, &
                                   real_salt_flux, is_nonlinfs, partit)
        end if
        call dump_node(DUMP_SUBSTEP_TRACERS, n, 'T', tracers%data(1)%values, mesh%nlevels_nod2D)
        call dump_node(DUMP_SUBSTEP_TRACERS, n, 'S', tracers%data(2)%values, mesh%nlevels_nod2D)

        !_______________________________________________________________________
        ! commit the new layer thicknesses (linfs: no-op)
        call update_thickness_ale(mesh, partit)
        call dump_node(DUMP_SUBSTEP_THICKNESS, n, 'hnode', mesh%hnode, mesh%nlevels_nod2D)
        call timer_stop(TMR_OCE_TRACER)
    end subroutine step_oce

end module mod_step_oce
