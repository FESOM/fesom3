module mod_step_oce_rk3
    ! The RK3 ocean step (FESOM3_RK3=1), the alternative to step_oce's AB2 sequence. Plan:
    ! docs/plans/2026-10-28-rk3.md, Technical Details §2; the scheme, the stage building blocks
    ! and the knobs are documented in src/oce/oce_rk3.F90. step_oce hands over to step_oce_rk3
    ! right after its first-call guards, with its own argument list.
    !
    ! One step, tau = dt (linfs; dyn%uv = u^n and the tracer values = T^n until stage 3):
    !   rk3_H = sum helem                               (every step: a restart rebuilds helem)
    !   stage 1 (tau/3)  w from u^n, split off (w_e = w, w_i = 0), no bolus
    !                    p(T^n) pressure only + PGF
    !                    tracers: valuesAB T^n -> T*       (HO, full w)
    !                    rk3_visc = the viscosity of u^n over tau
    !                    rk3_us = u^n + R1 - <R1>          (Coriolis, advection, PGF, -g grad eta^n)
    !                    SE: se_bt_forcing_rk3 + se_subcycle (eta^M, U^M, <<U>>; not committed)
    !   stage 2 (tau/2)  w from u*, split off, no bolus
    !                    p(T*) pressure only + PGF         (the textbook p(T*), plan D3: before T*
    !                                                       is overwritten)
    !                    tracers: T* -> T**
    !                    rk3_us = u^n + R2(u*) - <R2>
    !   stage 3 (tau)    p(T**) in full + PGF; sw_alpha_beta, sigma_xy, slopes from T**
    !                    compute_vel_nodes (u^n), shear, PP|KPP|TKE, mo_convect
    !                    w from the untrimmed u** with the split (w_e, w_i), no bolus
    !                    uv_rhs = R3(u**) + rk3_visc; impl_vert_visc_ale (base u^n, w_i, wind, drag)
    !                    CG: the SSH block of step_oce (update_vel: u^{n+1}; eta^{n+1})
    !                        rk3_us trimmed to the transport of u^{n+1}
    !                    SE: se_rk3_finish (u^{n+1} trimmed to U^M, rk3_us to <<U>>, eta^M committed)
    !                    GM/Redi from N^2(T**); w from the trimmed u** (bolus and split on)
    !                    solve_tracers_rk3 (advection from T^n with T**, diffusion, TDMA)
    !                    update_thickness_ale
    ! The per-substep dumps keep step_oce's ids, taken at the stage-3 equivalents (the restart
    ! gate compares them). FESOM3_RK3_CHECK adds the RK3_CHECK line (oce_rk3).
    use mod_precision,      only: WP
    use mod_mesh,           only: t_mesh
    use mod_dyn,            only: t_dyn
    use mod_tracer,         only: t_tracer
    use mod_partit,         only: t_partit
    use mod_part_bounds,    only: local_dims, owned_bounds
    use mod_dump,           only: dump_node, dump_node_2d, &
                                  DUMP_SUBSTEP_PRESSURE_BV, DUMP_SUBSTEP_MIXING, &
                                  DUMP_SUBSTEP_SSH_RHS, DUMP_SUBSTEP_SSH_SOLVE, &
                                  DUMP_SUBSTEP_HBAR, DUMP_SUBSTEP_ETA_N, &
                                  DUMP_SUBSTEP_ALE, DUMP_SUBSTEP_TRACERS, &
                                  DUMP_SUBSTEP_THICKNESS
    use oce_pressure_bv,    only: pressure_bv, sw_alpha_beta, compute_sigma_xy, compute_neutral_slope
    use oce_fer_gm,         only: init_Redi_GM, fer_solve_Gamma, fer_gamma2vel
    use mod_param_phys,     only: Fer_GM, Redi, mix_scheme_nmb
    use oce_pgf,            only: pressure_force_4_linfs_fullcell, pressure_force_4_zxxxx_shchepetkin
    use oce_ale_mixing_pp,  only: oce_mixing_pp
    use oce_mixing_kpp,     only: oce_mixing_kpp_driver
    use oce_mixing_tke,     only: calc_cvmix_tke
    use oce_mo_conv,        only: mo_convect
    use oce_vert_spline,    only: compute_shear2
    use oce_dyn_velrhs,     only: compute_vel_rhs
    use oce_dyn_ivertvisc,  only: impl_vert_visc_ale
    use oce_ssh_rhs,        only: compute_ssh_rhs_ale, update_stiff_mat_ale
    use mod_config,         only: which_ALE
    use oce_ssh_solve,      only: solve_ssh_ale
    use oce_ale,            only: compute_vel_nodes, update_vel, compute_hbar_ale, &
                                  update_eta_n, vert_vel_ale, update_thickness_ale
    use oce_ale_tracer,     only: solve_tracers_rk3
    use oce_ssh_se,         only: se_bt_forcing_rk3, se_subcycle, se_rk3_finish, se_check_tail
    use oce_rk3,            only: rk3_die, rk3_depth, rk3_transport, rk3_tracer_stage, rk3_viscosity, &
                                  rk3_momentum_stage, rk3_trim, rk3_check_begin, rk3_check_trim, &
                                  rk3_check_finish, RK3_CHK_TRIM1, RK3_CHK_TRIM2, RK3_CHK_TRIM3, RK3_CHK_TRIMM
    use mod_timer,          only: timer_start, timer_stop, &
                                  TMR_OCE_MIXPRES, TMR_OCE_DYN, TMR_OCE_SSH, &
                                  TMR_OCE_SOLVE, TMR_OCE_GMREDI, TMR_OCE_TRACER, &
                                  TMR_OCE_RK3_S1, TMR_OCE_RK3_S2, TMR_OCE_RK3_S3
    implicit none
    private
    public :: step_oce_rk3

contains

    subroutine step_oce_rk3(n, dt, dynamics, tracers, mesh, &
                            Ki, heat_flux, water_flux, virtual_salt, relax_salt, &
                            real_salt_flux, is_nonlinfs, stress_surf, partit, stress_node_surf)
        integer,        intent(in)            :: n
        real(kind=WP),  intent(in)            :: dt
        type(t_dyn),    intent(inout), target :: dynamics
        type(t_tracer), intent(inout), target :: tracers
        type(t_mesh),   intent(inout), target :: mesh
        real(kind=WP),  intent(in) :: Ki(mesh%nl-1, mesh%nod2D)
        real(kind=WP),  intent(in) :: heat_flux(mesh%nod2D), water_flux(mesh%nod2D)
        real(kind=WP),  intent(in) :: virtual_salt(mesh%nod2D), relax_salt(mesh%nod2D)
        real(kind=WP),  intent(in) :: real_salt_flux(mesh%nod2D), is_nonlinfs
        real(kind=WP),  intent(in) :: stress_surf(2, mesh%elem2D)
        type(t_partit), intent(in), optional :: partit
        real(kind=WP),  intent(in), optional :: stress_node_surf(2, mesh%nod2D)

        logical :: is_kpp, is_tke
        integer :: node, e, nz, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: tau3, tau2
        real(kind=WP), allocatable :: eta_old(:), target(:,:)

        if (.not. present(partit)) call rk3_die('step_oce_rk3: partit is required')
        is_kpp = (mix_scheme_nmb == 1)
        is_tke = (mix_scheme_nmb == 5)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        tau3 = dt/3.0_WP
        tau2 = 0.5_WP*dt

        call rk3_depth(dynamics, mesh, partit)
        if (dynamics%rk3_check) then
            call rk3_check_begin(dynamics, mesh, partit)
            allocate(eta_old(nNodL))
            eta_old = dynamics%eta_n(1:nNodL)
        end if

        !=======================================================================
        ! stage 1 (tau/3): from u^n, T^n
        call timer_start(TMR_OCE_RK3_S1)
        call timer_start(TMR_OCE_DYN)
        call vert_vel_ale(dynamics, mesh, dt, partit, split=.false., bolus=.false.)
        call timer_stop(TMR_OCE_DYN)
        call timer_start(TMR_OCE_MIXPRES)
        call stage_pressure(tracers%data(1)%values, tracers%data(2)%values, .true.)
        call timer_stop(TMR_OCE_MIXPRES)
        call timer_start(TMR_OCE_TRACER)
        call rk3_tracer_stage(1, tau3, dynamics%uv, dynamics%w, dynamics, tracers, mesh, partit)
        call timer_stop(TMR_OCE_TRACER)
        call timer_start(TMR_OCE_DYN)
        call rk3_viscosity(dynamics, mesh, dt, partit)
        call rk3_momentum_stage(1, tau3, dynamics, mesh, partit)
        call timer_stop(TMR_OCE_DYN)
        if (dynamics%use_ssh_se) then
            ! the barotropic step, once (rk3.tex's first-stage strategy): F from R1 (uv_rhs), the
            ! viscosity and the wind and drag of u^n, then the M substeps over tau
            call timer_start(TMR_OCE_SSH)
            call se_bt_forcing_rk3(dynamics, mesh, partit, dt, stress_surf)
            call se_subcycle(dynamics, mesh, partit, dt, water_flux)
            call timer_stop(TMR_OCE_SSH)
        end if
        if (dynamics%rk3_check) call rk3_check_trim(RK3_CHK_TRIM1, dynamics%rk3_us, dynamics, mesh, partit)
        call timer_stop(TMR_OCE_RK3_S1)

        !=======================================================================
        ! stage 2 (tau/2): from u*, T*
        call timer_start(TMR_OCE_RK3_S2)
        call timer_start(TMR_OCE_DYN)
        call vert_vel_ale(dynamics, mesh, dt, partit, uv_in=dynamics%rk3_us, split=.false., bolus=.false.)
        call timer_stop(TMR_OCE_DYN)
        call timer_start(TMR_OCE_MIXPRES)
        call stage_pressure(tracers%data(1)%valuesAB, tracers%data(2)%valuesAB, .true.)
        call timer_stop(TMR_OCE_MIXPRES)
        call timer_start(TMR_OCE_TRACER)
        call rk3_tracer_stage(2, tau2, dynamics%rk3_us, dynamics%w, dynamics, tracers, mesh, partit)
        call timer_stop(TMR_OCE_TRACER)
        call timer_start(TMR_OCE_DYN)
        call rk3_momentum_stage(2, tau2, dynamics, mesh, partit)
        call timer_stop(TMR_OCE_DYN)
        if (dynamics%rk3_check) call rk3_check_trim(RK3_CHK_TRIM2, dynamics%rk3_us, dynamics, mesh, partit)
        call timer_stop(TMR_OCE_RK3_S2)

        !=======================================================================
        ! stage 3 (tau): from u**, T**; the slow physics, the barotropic step, the tracer solve
        call timer_start(TMR_OCE_RK3_S3)
        call timer_start(TMR_OCE_MIXPRES)
        call stage_pressure(tracers%data(1)%valuesAB, tracers%data(2)%valuesAB, .false.)
        call dump_node(DUMP_SUBSTEP_PRESSURE_BV, n, 'density',  dynamics%work%density_m_rho0, mesh%nlevels_nod2D)
        call dump_node(DUMP_SUBSTEP_PRESSURE_BV, n, 'pressure', dynamics%work%hpressure,      mesh%nlevels_nod2D)
        call dump_node(DUMP_SUBSTEP_PRESSURE_BV, n, 'bvfreq',   dynamics%work%bvfreq,         mesh%nlevels_nod2D)
        call timer_stop(TMR_OCE_MIXPRES)

        ! EOS expansion coefficients, density gradient, neutral slopes from T** (step_oce:169-180)
        call timer_start(TMR_OCE_GMREDI)
        if (Fer_GM .or. Redi .or. is_kpp) then
            call sw_alpha_beta(tracers%data(1)%valuesAB, tracers%data(2)%valuesAB, mesh, &
                               dynamics%work%sw_alpha, dynamics%work%sw_beta, partit)
        end if
        if (Fer_GM .or. Redi) then
            call compute_sigma_xy(tracers%data(1)%valuesAB, tracers%data(2)%valuesAB, &
                                  dynamics%work%sw_alpha, dynamics%work%sw_beta, mesh, &
                                  dynamics%work%sigma_xy, partit)
            if (Redi) call compute_neutral_slope(dynamics%work%sigma_xy, dynamics%work%bvfreq, mesh, &
                                  dynamics%work%neutral_slope, dynamics%work%slope_tapered, &
                                  dynamics%work%fer_tapfac, partit)
        end if
        call timer_stop(TMR_OCE_GMREDI)

        ! vertical mixing from the u^n shear and N^2(T**), convective adjustment (step_oce:194-217)
        call timer_start(TMR_OCE_MIXPRES)
        call compute_vel_nodes(dynamics, mesh, partit)
        call compute_shear2(dynamics, mesh, partit)
        if (is_kpp) then
            if (.not. present(stress_node_surf)) &
                error stop 'step_oce_rk3: KPP (mix_scheme_nmb==1) requires stress_node_surf'
            call oce_mixing_kpp_driver(dynamics, tracers, stress_node_surf, heat_flux, water_flux, mesh, partit)
            do node = 1, nNodL
                dynamics%work%Kv(:, node) = dynamics%work%Kv_double(:, node, 1)
            end do
            call mo_convect(dynamics, mesh, partit)
        else if (is_tke) then
            if (.not. present(stress_node_surf)) &
                error stop 'step_oce_rk3: TKE (mix_scheme_nmb==5) requires stress_node_surf'
            call calc_cvmix_tke(dynamics, stress_node_surf, dt, mesh, partit)
            call mo_convect(dynamics, mesh, partit)
        else
            call oce_mixing_pp(dynamics, mesh, partit)
            call mo_convect(dynamics, mesh, partit)
        end if
        call dump_node(DUMP_SUBSTEP_MIXING, n, 'Kv', dynamics%work%Kv, mesh%nlevels_nod2D)
        call timer_stop(TMR_OCE_MIXPRES)

        ! momentum: w and its split from the untrimmed u**, R3 + the viscosity, the vertical TDMA
        call timer_start(TMR_OCE_DYN)
        call vert_vel_ale(dynamics, mesh, dt, partit, uv_in=dynamics%rk3_us, bolus=.false.)
        call compute_vel_rhs(dynamics, mesh, dt, .true., partit, uv_in=dynamics%rk3_us, no_history=.true.)
        do e = 1, nElemO
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                dynamics%uv_rhs(:, nz, e) = dynamics%uv_rhs(:, nz, e) + dynamics%rk3_visc(:, nz, e)
            end do
        end do
        call impl_vert_visc_ale(dynamics, mesh, dt, dynamics%work%Av, stress_surf, partit)
        call timer_stop(TMR_OCE_DYN)

        call timer_start(TMR_OCE_SSH)
        if (dynamics%use_ssh_se) then
            ! split-explicit: u^{n+1} to U^M, u** to <<U>>, eta^M committed (no ssh_rhs / d_eta)
            call se_rk3_finish(dynamics, mesh, partit, dt, n, water_flux)
            call dump_node_2d(DUMP_SUBSTEP_HBAR, n, 'hbar', mesh%hbar)
            call dump_node_2d(DUMP_SUBSTEP_ETA_N, n, 'eta_n', dynamics%eta_n)
            if (dynamics%rk3_check) then
                call rk3_check_trim(RK3_CHK_TRIM3, dynamics%rk3_us, dynamics, mesh, partit, &
                                    target=dynamics%se_umean)
                call rk3_check_trim(RK3_CHK_TRIMM, dynamics%uv, dynamics, mesh, partit, &
                                    target=dynamics%se_ubt(:, :, 0))
            end if
        else
            ! the CG corrector of step_oce (alpha = theta = 1), u^{n+1} in uv
            if (trim(which_ALE) /= 'linfs') call update_stiff_mat_ale(mesh, dt, partit)
            call compute_ssh_rhs_ale(dynamics, mesh, partit, water_flux=water_flux)
            call dump_node_2d(DUMP_SUBSTEP_SSH_RHS, n, 'ssh_rhs', dynamics%ssh_rhs)
            call timer_start(TMR_OCE_SOLVE)
            call solve_ssh_ale(dynamics, mesh, partit=partit)
            call timer_stop(TMR_OCE_SOLVE)
            call dump_node_2d(DUMP_SUBSTEP_SSH_SOLVE, n, 'd_eta', dynamics%d_eta)
            call update_vel(dynamics, mesh, dt, partit)
            call compute_hbar_ale(dynamics, mesh, dt, partit, water_flux=water_flux)
            call dump_node_2d(DUMP_SUBSTEP_HBAR, n, 'hbar', mesh%hbar)
            call update_eta_n(dynamics, mesh, partit)
            call dump_node_2d(DUMP_SUBSTEP_ETA_N, n, 'eta_n', dynamics%eta_n)
            ! the tracer transport: u** trimmed to the transport of u^{n+1}, which changed eta
            allocate(target(2, nElemL))
            call rk3_transport(dynamics%uv, mesh, nElemO, target)
            call rk3_trim(dynamics%rk3_us, target, dynamics, mesh, partit)
            if (dynamics%rk3_check) call rk3_check_trim(RK3_CHK_TRIM3, dynamics%rk3_us, dynamics, mesh, partit, &
                                                        target=target)
        end if
        call timer_stop(TMR_OCE_SSH)

        ! GM diffusivity, streamfunction and bolus velocity from N^2(T**) (step_oce:276-291)
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

        ! w, the split and the bolus w from the trimmed u** (the tracers' transport)
        call timer_start(TMR_OCE_DYN)
        call vert_vel_ale(dynamics, mesh, dt, partit, water_flux=water_flux, uv_in=dynamics%rk3_us)
        call dump_node(DUMP_SUBSTEP_ALE, n, 'hnode_new', mesh%hnode_new, mesh%nlevels_nod2D)
        call dump_node(DUMP_SUBSTEP_ALE, n, 'w',         dynamics%w,     mesh%nlevels_nod2D)
        call timer_stop(TMR_OCE_DYN)

        ! the stage-3 tracer solve (the Redi Ki switch of step_oce:307-315)
        call timer_start(TMR_OCE_TRACER)
        if (Redi) then
            call solve_tracers_rk3(dt, dynamics%rk3_lo_nsub, dynamics, tracers, mesh, dynamics%work%Ki, &
                                   heat_flux, water_flux, virtual_salt, relax_salt, &
                                   real_salt_flux, is_nonlinfs, partit)
        else
            call solve_tracers_rk3(dt, dynamics%rk3_lo_nsub, dynamics, tracers, mesh, Ki, &
                                   heat_flux, water_flux, virtual_salt, relax_salt, &
                                   real_salt_flux, is_nonlinfs, partit)
        end if
        call dump_node(DUMP_SUBSTEP_TRACERS, n, 'T', tracers%data(1)%values, mesh%nlevels_nod2D)
        call dump_node(DUMP_SUBSTEP_TRACERS, n, 'S', tracers%data(2)%values, mesh%nlevels_nod2D)
        call update_thickness_ale(mesh, partit)
        call dump_node(DUMP_SUBSTEP_THICKNESS, n, 'hnode', mesh%hnode, mesh%nlevels_nod2D)
        call timer_stop(TMR_OCE_TRACER)
        call timer_stop(TMR_OCE_RK3_S3)

        ! FESOM3_SE_CHECK: the zstar checks of the committed thicknesses (step_oce:328)
        if (dynamics%use_ssh_se .and. dynamics%se_check) call se_check_tail(dynamics, mesh, partit, n)

        if (dynamics%rk3_check) call rk3_check_finish(n, dt, eta_old, dynamics%rk3_us, dynamics, tracers, &
                                                      mesh, partit, water_flux=water_flux)

    contains

        subroutine stage_pressure(temp, salt, ponly)
            ! EOS + hydrostatic pressure (+ N^2 and, with KPP, dbsfc unless ponly) of (temp, salt),
            ! then the PGF as step_oce:130-158
            real(kind=WP), intent(in) :: temp(:,:), salt(:,:)
            logical,       intent(in) :: ponly
            if (ponly) then
                call pressure_bv(temp, salt, dynamics%work%density_ref, mesh, &
                                 dynamics%work%density_m_rho0, dynamics%work%hpressure, &
                                 dynamics%work%bvfreq, partit, pressure_only=.true.)
            else if (is_kpp) then
                call pressure_bv(temp, salt, dynamics%work%density_ref, mesh, &
                                 dynamics%work%density_m_rho0, dynamics%work%hpressure, &
                                 dynamics%work%bvfreq, partit, dbsfc=dynamics%work%dbsfc, &
                                 bvfreq_raw=dynamics%work%bvfreq_raw)
            else
                call pressure_bv(temp, salt, dynamics%work%density_ref, mesh, &
                                 dynamics%work%density_m_rho0, dynamics%work%hpressure, &
                                 dynamics%work%bvfreq, partit, &
                                 bvfreq_raw=dynamics%work%bvfreq_raw)
            end if
            if (trim(which_ALE) == 'linfs') then
                call pressure_force_4_linfs_fullcell(dynamics%work%hpressure, mesh, &
                                                     dynamics%work%pgf_x, dynamics%work%pgf_y, partit)
            else
                dynamics%work%pgf_x = 0.0_WP; dynamics%work%pgf_y = 0.0_WP
                call pressure_force_4_zxxxx_shchepetkin(dynamics%work%density_m_rho0, mesh, &
                                                     dynamics%work%pgf_x, dynamics%work%pgf_y, partit)
            end if
        end subroutine stage_pressure

    end subroutine step_oce_rk3

end module mod_step_oce_rk3
