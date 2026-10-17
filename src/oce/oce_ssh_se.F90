module oce_ssh_se
    ! Split-explicit (SE) barotropic solver, an alternative to the semi-implicit CG SSH solve
    ! (oce_ssh_rhs + oce_ssh_solve). Plan: docs/plans/2026-10-17-split-explicit.md. Scheme:
    ! the Shchepetkin-McWilliams (2005) AB3-AM4 barotropic subcycling of Sergey Danilov's notes
    ! (~/port_kokkos/ssh_sergey/subcycling.tex; Banerjee et al. 2024, GMD 17, 7051). Per
    ! baroclinic step tau = dt the 2-D system (eta, U = sum_k helem*u) is advanced in M substeps
    ! dtbt = tau/M:
    !     U*        = (3/2+beta) U^m - (1/2+2 beta) U^{m-1} + beta U^{m-2}                    (AB3)
    !     eta^{m+1} = eta^m + dtbt*(I(U*) - W*areasvol)/areasvol
    !     eta^A     = delta eta^{m+1} + (1-delta-gamma-zeta) eta^m + gamma eta^{m-1}
    !                 + zeta eta^{m-2}                                                       (AM4)
    !     U^{m+1}   = U^m + dtbt*(-f e_z x U* - g H^A grad(eta^A) + F + V(U^m - U^n))
    ! with (notes, subcycling.tex:81-91)
    !     beta = 0.281105, zeta = 0.00976186 - 0.13451357 chi, gamma = 0.083445 - 0.513584 chi,
    !     delta = 1/2 + 2 zeta + gamma + 2 chi,
    ! chi the dissipation parameter. Afterwards the 3-D velocity is trimmed to the mean transport
    ! <<U>>, and the existing ALE tail consumes the result through hbar/hbar_old.
    !
    ! IN PLACE: the knobs, the state in t_dyn, se_init (coefficients, cold start, CFL guard) and
    ! the step_oce guard (Task 0.3); the element depth H^n and the barotropic forcing F
    ! (se_elem_depth, se_bt_forcing; Task 1.1); the substeps (se_subcycle) with their operators
    ! (se_inflow, se_visc, se_tendency, se_depth_at) and the viscosity coefficients (se_kappa_build,
    ! from se_init; Task 1.2). Still to come: the trim (se_trim_finish, Task 1.3).
    !
    ! DETERMINISM (plan D2): every per-element operator of the substeps is a GATHER over the
    ! element's own vertex / neighbour slots in fixed slot order, never a scatter into elements.
    ! A shared border element (owned by every rank that owns one of its vertices) is then
    ! computed from bitwise-equal inputs on every owner, once F is synced (se_bt_forcing), so its
    ! copies of U stay bitwise equal without a per-substep sync. The one scatter, I(U) into the
    ! nodes, writes owned nodes only from owned edges (one owner per node).
    !
    ! KNOBS (env only, like every run switch of FESOM_F):
    !   FESOM3_SSH_SE      0 (default) = semi-implicit CG, 1 = split-explicit
    !   FESOM3_SE_M        substeps per baroclinic step, an integer >= 1              default 50
    !   FESOM3_SE_M_FORCE  1 = run with M below the CFL-guard minimum (0/1)           default 0
    !   FESOM3_SE_CHI      SM dissipation parameter, 0 <= chi <= 0.25                 default 0.1
    !   FESOM3_SE_VISC_AH  BT viscosity A_h = value*l_e [m/s], >= 0; 0 = off          default 0.01
    !   FESOM3_SE_CHECK    1 = per-step invariant suite (0/1)                         default 0
    ! Value-checked like FESOM3_SHARED_SYNC: anything outside the range stops the run, so a run
    ! never carries a setting other than the one its job file asked for. The FESOM3_SE_* knobs
    ! are read only when FESOM3_SSH_SE=1. SE is SILENT when off: run_conserve_pi.sh and
    ! run_tri_regression.sh parse fixed output lines.
    !
    ! CFL GUARD (port_kokkos src/fesom_ssh_se.cpp:255-296): the substep must resolve the fastest
    ! external gravity wave, dtbt <= 0.5*mesh_resolution/sqrt(g*H) at every owned surface node,
    ! H the resting depth of the vertex column. The limit is MIN-reduced over the ranks, so
    ! M_min = ceiling(tau/limit) is the same on every rank. 0.5 is a conservative safety factor
    ! for the stability bound, not a calibrated number (port_kokkos, CORE2 dt 1800 s: M_min 35).
    !
    ! The parsers (se_parse_knob) and the guard decision (se_cfl_decision) return their verdict
    ! instead of stopping, so test/test_ssh_se.F90 checks every rejection; the driver-facing
    ! wrappers (se_env_requested, se_read_env, se_init, se_step_guard) stop through se_die
    ! (MPI_Abort, so no rank is left waiting in a collective).
    use mpi
    use, intrinsic :: iso_fortran_env, only: output_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use mod_precision,   only: WP, MPI_WP
    use mod_constants,   only: g, r_earth
    use mod_config,      only: ab_epsilon, which_ALE
    use mod_mesh,        only: t_mesh, MAX_NV
    use mod_dyn,         only: t_dyn
    use mod_partit,      only: t_partit
    use mod_part_bounds, only: local_dims, is_multirank
    use mod_halo,        only: exchange_elem_shared, exchange_elem, exchange_nod
    use mod_timer,       only: timer_start, timer_stop, TMR_OCE_SHSYNC, TMR_OCE_SE_BT
    use mod_mesh_rotate, only: trim_cyclic
    use mod_mesh_areas,  only: elem_center
    implicit none
    private
    public :: se_env_requested, se_read_env, se_parse_knob, se_init, se_step_guard
    public :: se_elem_depth, se_bt_forcing, se_subcycle
    public :: se_inflow, se_visc, se_tendency, se_depth_at, se_kappa_build
    public :: se_sm_coeffs, se_cfl_dtbt_limit, se_cfl_m_min, se_cfl_decision
    public :: SE_CFL_OK, SE_CFL_FORCED, SE_CFL_DIE

    ! se_cfl_decision verdicts
    integer, parameter :: SE_CFL_OK     = 0   ! M >= M_min
    integer, parameter :: SE_CFL_FORCED = 1   ! M <  M_min, FESOM3_SE_M_FORCE=1: warn and run
    integer, parameter :: SE_CFL_DIE    = 2   ! M <  M_min: stop

    ! the knobs se_read_env reads (FESOM3_SSH_SE itself: se_env_requested)
    character(len=17), parameter :: SE_KNOBS(5) = [character(len=17) :: &
        'FESOM3_SE_M', 'FESOM3_SE_M_FORCE', 'FESOM3_SE_CHI', 'FESOM3_SE_VISC_AH', 'FESOM3_SE_CHECK']

contains

    !===========================================================================
    logical function se_env_requested(partit)
        ! FESOM3_SSH_SE, parsed ONCE and cached: unset or 0 = the semi-implicit CG solve, 1 =
        ! split-explicit, any other value stops. The one place that decides whether SE is
        ! wanted: se_init and the step_oce guard both ask here. partit (optional, as in
        ! step_oce) only selects the communicator of the stop.
        type(t_partit), intent(in), optional :: partit
        integer, save :: state = -1           ! -1 = not parsed yet, 0 = off, 1 = on
        type(t_dyn)        :: scratch
        character(len=256) :: val, msg
        logical :: is_set, ok

        if (state < 0) then
            state = 0
            call get_knob('FESOM3_SSH_SE', val, is_set)
            if (is_set) then
                call se_parse_knob('FESOM3_SSH_SE', val, scratch, ok, msg)
                if (.not. ok) call se_die(trim(msg), partit)
                if (scratch%use_ssh_se) state = 1
            end if
        end if
        se_env_requested = (state == 1)
    end function se_env_requested

    !===========================================================================
    subroutine se_read_env(dyn, partit)
        ! The FESOM3_SE_* knobs into dyn; an unset knob keeps its t_dyn default (M 50, chi 0.1,
        ! visc_ah 0.01, check and force off). Stops on a bad value. Called by se_init.
        type(t_dyn),    intent(inout) :: dyn
        type(t_partit), intent(in)    :: partit
        character(len=256) :: val, msg
        logical :: is_set, ok
        integer :: i

        do i = 1, size(SE_KNOBS)
            call get_knob(trim(SE_KNOBS(i)), val, is_set)
            if (.not. is_set) cycle
            call se_parse_knob(trim(SE_KNOBS(i)), val, dyn, ok, msg)
            if (.not. ok) call se_die(trim(msg), partit)
        end do
    end subroutine se_read_env

    !===========================================================================
    subroutine se_parse_knob(name, str, dyn, ok, msg)
        ! Parse the text str of the SE knob name (FESOM3_SSH_SE or one of FESOM3_SE_*) into its
        ! t_dyn field. A bad value leaves the field unchanged and returns ok = .false. with msg =
        ! '<name> must ...: "<value>"'. No output and no stop: the caller decides.
        ! Switches accept exactly '0' or '1'; FESOM3_SE_M digits only; the reals the characters
        ! of a plain number (no NaN/Inf text, no list separators, so "0.1 0.2" is not read as
        ! 0.1) and a finite result.
        character(len=*), intent(in)    :: name, str
        type(t_dyn),      intent(inout) :: dyn
        logical,          intent(out)   :: ok
        character(len=*), intent(out)   :: msg
        character(len=40) :: need
        integer       :: iv
        real(kind=WP) :: rv
        logical       :: lv

        ok = .false.
        select case (name)
        case ('FESOM3_SSH_SE')
            need = 'must be 0 or 1'
            call parse_01(str, lv, ok)
            if (ok) dyn%use_ssh_se = lv
        case ('FESOM3_SE_M')
            need = 'must be an integer >= 1'
            call parse_int(str, iv, ok)
            if (ok) ok = (iv >= 1)
            if (ok) dyn%se_M = iv
        case ('FESOM3_SE_M_FORCE')
            need = 'must be 0 or 1'
            call parse_01(str, lv, ok)
            if (ok) dyn%se_m_force = lv
        case ('FESOM3_SE_CHI')
            need = 'must be a real in [0, 0.25]'
            call parse_real(str, rv, ok)
            if (ok) ok = (rv >= 0.0_WP .and. rv <= 0.25_WP)
            if (ok) dyn%se_chi = rv
        case ('FESOM3_SE_VISC_AH')
            need = 'must be a real >= 0'
            call parse_real(str, rv, ok)
            if (ok) ok = (rv >= 0.0_WP)
            if (ok) dyn%se_visc_ah = rv
        case ('FESOM3_SE_CHECK')
            need = 'must be 0 or 1'
            call parse_01(str, lv, ok)
            if (ok) dyn%se_check = lv
        case default
            need = 'is not a split-explicit knob'
        end select
        msg = ''
        if (.not. ok) msg = name//' '//trim(need)//': "'//trim(adjustl(str))//'"'
    end subroutine se_parse_knob

    !===========================================================================
    pure subroutine se_sm_coeffs(chi, c_beta, c_zeta, c_gamma, c_delta)
        ! The AB3-AM4 weights of the notes (subcycling.tex:81-91) for the dissipation parameter
        ! chi. AB3: (3/2+beta, -(1/2+2 beta), beta); AM4: (delta, 1-delta-gamma-zeta, gamma,
        ! zeta). Both sum to 1 for every chi.
        real(kind=WP), intent(in)  :: chi
        real(kind=WP), intent(out) :: c_beta, c_zeta, c_gamma, c_delta
        c_beta  = 0.281105_WP
        c_zeta  = 0.00976186_WP - 0.13451357_WP*chi
        c_gamma = 0.083445_WP   - 0.513584_WP*chi
        c_delta = 0.5_WP + 2.0_WP*c_zeta + c_gamma + 2.0_WP*chi
    end subroutine se_sm_coeffs

    !===========================================================================
    function se_cfl_dtbt_limit(mesh, partit) result(lim)
        ! The largest admissible substep: min over the OWNED surface nodes of
        ! 0.5*mesh_resolution/sqrt(g*H), H = zbar(ulevels_nod2D) - zbar(nlevels_nod2D) the
        ! resting depth of the vertex column (the vertex bottom is authoritative), MIN-reduced
        ! over all ranks -- identical everywhere. huge() if no rank owns a wet surface node.
        ! Collective at npes > 1.
        type(t_mesh),   intent(in) :: mesh
        type(t_partit), intent(in) :: partit
        real(kind=WP) :: lim
        integer       :: n, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, ierr
        real(kind=WP) :: H

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        lim = huge(1.0_WP)
        do n = 1, nNodO
            if (mesh%ulevels_nod2D(n) /= 1 .or. mesh%nlevels_nod2D(n) < 2) cycle
            H = mesh%zbar(mesh%ulevels_nod2D(n)) - mesh%zbar(mesh%nlevels_nod2D(n))
            if (.not. (H > 0.0_WP)) cycle
            lim = min(lim, real(0.5_WP*mesh%mesh_resolution(n)/sqrt(g*H), WP))
        end do
        if (partit%npes > 1) call MPI_Allreduce(MPI_IN_PLACE, lim, 1, MPI_WP, MPI_MIN, &
                                                partit%MPI_COMM_FESOM, ierr)
    end function se_cfl_dtbt_limit

    !===========================================================================
    pure integer function se_cfl_m_min(tau, lim)
        ! The fewest substeps with tau/M <= lim (se_cfl_dtbt_limit); 1 without a limit.
        real(kind=WP), intent(in) :: tau, lim
        if (lim >= huge(lim)) then
            se_cfl_m_min = 1
        else
            se_cfl_m_min = max(1, ceiling(tau/lim))
        end if
    end function se_cfl_m_min

    !===========================================================================
    pure integer function se_cfl_decision(M, M_min, force)
        ! SE_CFL_OK for M >= M_min; below it SE_CFL_FORCED with FESOM3_SE_M_FORCE=1, else
        ! SE_CFL_DIE. For an integer M, M < M_min is the same as tau/M > lim.
        integer, intent(in) :: M, M_min
        logical, intent(in) :: force
        if (M >= M_min) then
            se_cfl_decision = SE_CFL_OK
        else if (force) then
            se_cfl_decision = SE_CFL_FORCED
        else
            se_cfl_decision = SE_CFL_DIE
        end if
    end function se_cfl_decision

    !===========================================================================
    subroutine se_init(dyn, mesh, partit, dt)
        ! Called once by a driver that supports SE (fesom_lifecycle_native_mr, fesom_conserve),
        ! after its cold init and before a restart read (which then overwrites the cold rings).
        ! A silent no-op unless FESOM3_SSH_SE=1. Otherwise: the knobs, the guards (AB2 momentum,
        ! no cavity), the SM coefficients, the state of the plan's Technical Details §1, the
        ! cold start, the viscosity coefficients (se_kappa_build), the CFL guard, and se_ready.
        ! Cold start: the eta ring holds eta_n (eta^{-1/M} = eta^{-2/M} = eta^n), the U ring the
        ! depth-integrated velocity sum_k helem*uv on owned + eDim elements (helem and uv are
        ! halo-valid), and s_old = 0 (the first step is lfirst: its Coriolis weight on s_old is
        ! 0 then, as compute_vel_rhs's ff = 1).
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: dt
        integer       :: n, e, nz, M_min
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: lim, ub(2)
        character(len=320) :: msg

        if (.not. se_env_requested(partit)) return
        if (dyn%se_ready) call se_die('se_init called twice', partit)
        call se_read_env(dyn, partit)
        dyn%use_ssh_se = .true.
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)

        ! ---- guards ----
        ! F carries the AB2 Coriolis weights of compute_vel_rhs; the SE eta equation and the
        ! CFL guard assume an open surface at every vertex.
        if (dyn%AB_order /= 2) &
            call se_die('split-explicit needs the AB2 momentum predictor (AB_order == 2)', partit)
        if (any(mesh%ulevels_nod2D(1:nNodL) > 1)) &
            call se_die('split-explicit does not support cavities (ulevels_nod2D > 1)', partit)

        call se_sm_coeffs(dyn%se_chi, dyn%se_beta, dyn%se_zeta, dyn%se_gamma, dyn%se_delta)

        ! ---- state ----
        allocate(dyn%se_eta(nNodL, 2), dyn%se_etam(nNodL), dyn%se_etanew(nNodL))
        allocate(dyn%se_ubt(2, nElemL, 0:2), dyn%se_sbar_old(2, nElemL), dyn%se_ubtn(2, nElemL))
        allocate(dyn%se_umean(2, nElemL), dyn%se_force(2, nElemL), dyn%se_H(nElemL))
        allocate(dyn%se_c(MAX_NV, nElemO), dyn%se_kappa(MAX_NV, nElemO))
        dyn%se_etam  = 0.0_WP; dyn%se_etanew = 0.0_WP
        dyn%se_ubtn  = 0.0_WP; dyn%se_umean  = 0.0_WP; dyn%se_force = 0.0_WP
        dyn%se_H     = 0.0_WP; dyn%se_c      = 0.0_WP; dyn%se_kappa = 0.0_WP

        ! ---- cold start ----
        do n = 1, nNodL
            dyn%se_eta(n, 1) = dyn%eta_n(n)
            dyn%se_eta(n, 2) = dyn%eta_n(n)
        end do
        do e = 1, nElemL
            ub = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                ub(1) = ub(1) + mesh%helem(nz, e)*dyn%uv(1, nz, e)
                ub(2) = ub(2) + mesh%helem(nz, e)*dyn%uv(2, nz, e)
            end do
            dyn%se_ubt(:, e, 0) = ub
            dyn%se_ubt(:, e, 1) = ub
            dyn%se_ubt(:, e, 2) = ub
        end do
        dyn%se_sbar_old = 0.0_WP

        ! ---- BT viscosity coefficients (geometry, once) ----
        call se_kappa_build(dyn, mesh, partit)

        ! ---- CFL guard ----
        lim   = se_cfl_dtbt_limit(mesh, partit)
        M_min = se_cfl_m_min(dt, lim)
        if (partit%mype == 0) then
            write(*,'(a)') 'oce_ssh_se: split-explicit barotropic solver ON (Shchepetkin-McWilliams AB3-AM4)'
            write(*,'(a,i0,a,f0.3,a,f6.4,a,es9.2,a,l1)') 'oce_ssh_se:   M = ', dyn%se_M, &
                ', dtbt = ', dt/real(dyn%se_M, WP), ' s, chi = ', dyn%se_chi, &
                ', visc_ah = ', dyn%se_visc_ah, ' m/s, check = ', dyn%se_check
            write(*,'(a,4(a,f12.9))') 'oce_ssh_se:  ', ' beta = ', dyn%se_beta, ', zeta = ', &
                dyn%se_zeta, ', gamma = ', dyn%se_gamma, ', delta = ', dyn%se_delta
            write(*,'(a,f0.3,a,i0)') 'oce_ssh_se:   CFL guard: dtbt <= ', lim, &
                ' s (0.5*res/sqrt(g*H), min over the mesh) -> M_min = ', M_min
        end if
        select case (se_cfl_decision(dyn%se_M, M_min, dyn%se_m_force))
        case (SE_CFL_FORCED)
            if (partit%mype == 0) write(*,'(a,i0,a,i0,a)') 'oce_ssh_se:   WARNING: M = ', &
                dyn%se_M, ' is below the CFL-guard minimum M_min = ', M_min, &
                '; FESOM3_SE_M_FORCE=1, running anyway'
        case (SE_CFL_DIE)
            write(msg,'(a,i0,a,i0,a,f0.3,a,f0.3,a,i0,a)') 'M = ', dyn%se_M, &
                ' is below the CFL-guard minimum M_min = ', M_min, ' (dtbt = ', &
                dt/real(dyn%se_M, WP), ' s > ', lim, ' s): set FESOM3_SE_M >= ', M_min, &
                ', or FESOM3_SE_M_FORCE=1 to run anyway'
            call se_die(trim(msg), partit)
        end select

        dyn%se_ready = .true.
    end subroutine se_init

    !===========================================================================
    subroutine se_step_guard(dyn, partit)
        ! step_oce's first-call check. FESOM3_SSH_SE=1 needs a driver that ran se_init
        ! (fesom_lifecycle_native_mr, fesom_conserve): any other driver would silently run the
        ! CG solve. Until step_oce has its SE branch (plan Task 1.3) an initialised SE run stops
        ! here as well, for the same reason.
        type(t_dyn),    intent(in)           :: dyn
        type(t_partit), intent(in), optional :: partit
        if (.not. se_env_requested(partit)) return
        if (.not. dyn%se_ready) &
            call se_die('FESOM3_SSH_SE=1 but this driver does not initialise split-explicit', partit)
        call se_die('FESOM3_SSH_SE=1: step_oce has no split-explicit step yet '// &
                    '(docs/plans/2026-10-17-split-explicit.md, Task 1.3)', partit)
    end subroutine se_step_guard

    !===========================================================================
    subroutine se_elem_depth(dyn, mesh, partit)
        ! H^n_e = sum_k helem^n over the layers ulevels(e)..nlevels(e)-1 (the loop of se_init's
        ! cold-start transport), on the OWNED + eDim elements: helem is halo-valid, and the
        ! viscosity gather reads H at eDim neighbours. The substeps use the depth law of plan D3,
        ! H_e(eta) = H^n_e + sum_v elem_w(v,e)*c(v,e)*(eta_v - eta^n_v). Under linfs c == 0 (se_c
        ! keeps the zero of se_init), so H_e(eta) = H^n_e. The zstar weights c come with Task
        ! 3.1; until then a non-linfs run stops here rather than run with c = 0.
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        integer       :: e, nz
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: H

        if (trim(which_ALE) /= 'linfs') &
            call se_die('split-explicit supports which_ALE = linfs only so far (zstar depth law: '// &
                        'docs/plans/2026-10-17-split-explicit.md, Task 3.1)', partit)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        do e = 1, nElemL
            H = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                H = H + mesh%helem(nz, e)
            end do
            dyn%se_H(e) = H
        end do
    end subroutine se_elem_depth

    !===========================================================================
    subroutine se_bt_forcing(dyn, mesh, partit, dt, lfirst)
        ! Once per baroclinic step, after the predictor (compute_vel_rhs, viscosity_filter,
        ! impl_vert_visc_ale), when uv = u^{n-1/2}, UV_rhs = u** - u^{n-1/2}, helem = helem^n and
        ! eta_n = eta^n. H^n (se_elem_depth), then on the OWNED elements the forcing F = Rbar - r
        ! of the notes (subcycling.tex:55-58):
        !     G   = sum_k helem^n*UV_rhs          = tau*Rbar, the depth-integrated predictor
        !     s^n = sum_k helem^n*uv              the stored Coriolis transport (plan D4)
        !     F   = G/tau + g H^n grad(eta^n) + f e_z x (a2 s^n + a1 s^{n-1})
        ! The r-term removes what compute_vel_rhs put into UV_rhs and the substep operator
        ! (se_tendency: -f e_z x U* - g H^A grad(eta^A)) adds back:
        !   - pressure: compute_vel_rhs adds dt*Fx to every layer, Fx = gradient_sca . (-g*eta_n)
        !     (oce_dyn_velrhs.F90:105-111). F subtracts H^n*Fx with Fx formed by the same
        !     expression, so the two cancel to rounding; no second discretisation of grad(eta).
        !   - Coriolis: compute_vel_rhs stores (v*ff, -u*ff) = -f e_z x u (times elem_area) in
        !     uv_rhsAB and blends ab2*(this step) + ab1*(previous step), ab2 = 1.5+eps (1 on
        !     lfirst), ab1 = -(0.5+eps). The same weights over the stored transports, a2 = ab2,
        !     a1 = ab1, cancel it exactly under linfs (helem constant) and to second order under
        !     zstar (uv_rhsAB weighs the previous layer velocities with helem^{n-1}). On lfirst
        !     s^{n-1} = 0 (se_init), as compute_vel_rhs's cold uv_rhsAB.
        ! Momentum advection (also in uv_rhsAB), viscosity, the baroclinic pressure and the
        ! wind stress stay in F.
        ! Then s^{n-1} <- s^n; U^n (ring level 0, owned + eDim) is copied to se_ubtn for the
        ! viscosity V(U^m - U^n); and F is synced over the shared elements, one owner as uv in
        ! update_vel (plan D2): the copies of G differ by roundoff (the viscosity filter's
        ! order-dependent sums). A no-op at 1 rank.
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: dt       ! tau, the baroclinic step
        logical,        intent(in)    :: lfirst   ! first Euler step (compute_vel_rhs's ff = 1)
        integer       :: e, nz, elnodes(MAX_NV)
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: a1, a2, f, Fx, Fy, pre(MAX_NV), gb(2), sn(2), sc(2)   ! gb: G (g is gravity)

        call se_elem_depth(dyn, mesh, partit)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)

        ! the AB2 weights of compute_vel_rhs (oce_dyn_velrhs.F90:97-100, :141-142)
        a1 = -(0.5_WP + ab_epsilon)
        a2 =  (1.5_WP + ab_epsilon)
        if (lfirst) a2 = 1.0_WP

        do e = 1, nElemO
            gb = 0.0_WP
            sn = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                gb(1) = gb(1) + mesh%helem(nz, e)*dyn%uv_rhs(1, nz, e)
                gb(2) = gb(2) + mesh%helem(nz, e)*dyn%uv_rhs(2, nz, e)
                sn(1) = sn(1) + mesh%helem(nz, e)*dyn%uv(1, nz, e)
                sn(2) = sn(2) + mesh%helem(nz, e)*dyn%uv(2, nz, e)
            end do
            ! grad(eta^n) exactly as compute_vel_rhs: Fx = -g d(eta)/dx
            elnodes = mesh%elem2D_nodes(:, e)
            pre = -(g*dyn%eta_n(elnodes))
            Fx  = sum(mesh%gradient_sca(1:4, e)*pre)
            Fy  = sum(mesh%gradient_sca(5:8, e)*pre)
            ! f e_z x sc = f*(-sc(2), sc(1))
            sc(1) = a2*sn(1) + a1*dyn%se_sbar_old(1, e)
            sc(2) = a2*sn(2) + a1*dyn%se_sbar_old(2, e)
            f = mesh%coriolis(e)
            dyn%se_force(1, e) = gb(1)/dt - dyn%se_H(e)*Fx - f*sc(2)
            dyn%se_force(2, e) = gb(2)/dt - dyn%se_H(e)*Fy + f*sc(1)
            dyn%se_sbar_old(:, e) = sn
        end do

        do e = 1, nElemL
            dyn%se_ubtn(:, e) = dyn%se_ubt(:, e, 0)
        end do

        if (is_multirank(partit)) then
            ! (nested ifs: Fortran does not short-circuit)
            if (partit%sync_shared_elem) then
                call timer_start(TMR_OCE_SHSYNC)
                call exchange_elem_shared(dyn%se_force, partit)
                call timer_stop(TMR_OCE_SHSYNC)
            end if
        end if
    end subroutine se_bt_forcing

    !===========================================================================
    subroutine se_kappa_build(dyn, mesh, partit)
        ! The BT viscosity coefficients (plan D5), once from the geometry. Slot k of an OWNED
        ! element e is the edge between its vertex slots k and k+1, nb = elem_neighbors(k,e) the
        ! element across it:
        !     kappa(k,e) = A_h*l/(d*A_e),  A_h = se_visc_ah*l     (the notes' A_h = 0.01 l_e)
        ! l the edge length, d the distance between the centroids of e and nb, A_e = elem_area.
        ! se_visc then forms V_e = sum_k kappa(k,e)*H_edge*(dU_nb/H_nb - dU_e/H_e), the flux
        ! A_h*H_edge*grad(U/H) through each edge divided by the element area. kappa = 0 on a
        ! boundary slot (nb == e: no flux through a coast) and on a triangle's slot 4 (never
        ! visited by se_visc).
        ! Geometry from the element's own vertices: edge_dxdy exists for owned edges only and an
        ! owned element can have one halo edge (build_elem_adjacency). Centroids by elem_center for
        ! the owned elements, the eDim neighbours' through exchange_elem (elem2D_nodes is stored
        ! for owned elements only), as compute_geometry does (mod_mesh_areas.F90:80-89). The
        ! longitude metric is the mean elem_cos of the two elements, as edge_len's
        ! (compute_edge_geometry), so l equals edge_len for an owned edge, and l, d and H_edge
        ! come out bitwise the same seen from either element: A_e*kappa(k,e) and the neighbour's
        ! A_nb*kappa agree to rounding, which is what makes sum_e A_e*V_e = 0.
        ! Collective at npes > 1 (the centroid exchange).
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        integer       :: e, k, nv, nb, v1, v2
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: a1, a2, b1, b2, cosm, l, d
        real(kind=WP), allocatable :: cx(:), cy(:)

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        allocate(cx(nElemL), cy(nElemL))
        cx = 0.0_WP; cy = 0.0_WP
        do e = 1, nElemO
            call elem_center(mesh, e, cx(e), cy(e))
        end do
        if (is_multirank(partit)) then
            call exchange_elem(cx, partit)
            call exchange_elem(cy, partit)
        end if

        dyn%se_kappa = 0.0_WP
        do e = 1, nElemO
            nv = mesh%elem2D_nnodes(e)
            do k = 1, nv
                nb = mesh%elem_neighbors(k, e)
                if (nb == e) cycle                                   ! boundary slot
                v1 = mesh%elem2D_nodes(k, e)
                v2 = mesh%elem2D_nodes(mod(k, nv) + 1, e)
                cosm = 0.5_WP*(mesh%elem_cos(e) + mesh%elem_cos(nb))
                a1 = mesh%coord_nod2D(1, v2) - mesh%coord_nod2D(1, v1)
                call trim_cyclic(a1)
                a2 = mesh%coord_nod2D(2, v2) - mesh%coord_nod2D(2, v1)
                l  = sqrt((a1*cosm*r_earth)**2 + (a2*r_earth)**2)  ! [m], = edge_len
                b1 = cx(nb) - cx(e)
                call trim_cyclic(b1)
                b2 = cy(nb) - cy(e)
                d  = sqrt((b1*cosm*r_earth)**2 + (b2*r_earth)**2)  ! [m]
                dyn%se_kappa(k, e) = dyn%se_visc_ah*l*l/(d*mesh%elem_area(e))
            end do
        end do
        deallocate(cx, cy)
    end subroutine se_kappa_build

    !===========================================================================
    subroutine se_inflow(mesh, partit, U, out)
        ! I(U): the inflow edge-scatter of compute_hbar_ale (oce_ale.F90:216-246) with the 2-D
        ! transport U(2, elem) in place of sum_k helem*uv. Over the OWNED edges in their order:
        !     c = (U2(el1)*dX1 - U1(el1)*dY1) - (U2(el2)*dX2 - U1(el2)*dY2)  [el2: off a boundary]
        !     out(n1) += c,  out(n2) -= c
        ! (dX, dY = edge_cross_dxdy). Positive = convergence, the ssh_rhs_old sign: eta
        ! advances by dt*out/areasvol. out is zeroed on owned + halo nodes and complete on the
        ! OWNED nodes (every edge of an owned node is owned); its halo entries are partial sums.
        ! U is read on owned elements only (both elements of an owned edge are owned).
        type(t_mesh),   intent(in)  :: mesh
        type(t_partit), intent(in)  :: partit
        real(kind=WP),  intent(in)  :: U(:,:)       ! (2, >= nElemO)
        real(kind=WP),  intent(out) :: out(:)       ! (>= nNodL)
        integer       :: ed, el(2), enodes(2), n
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: c1, c2

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        do n = 1, nNodL
            out(n) = 0.0_WP
        end do
        do ed = 1, nEdgeO
            enodes = mesh%edges(:, ed)
            el     = mesh%edge_tri(:, ed)
            c1 = U(2, el(1))*mesh%edge_cross_dxdy(1, ed) - U(1, el(1))*mesh%edge_cross_dxdy(2, ed)
            c2 = 0.0_WP
            if (el(2) /= el(1)) &
                c2 = -(U(2, el(2))*mesh%edge_cross_dxdy(3, ed) - U(1, el(2))*mesh%edge_cross_dxdy(4, ed))
            out(enodes(1)) = out(enodes(1)) + (c1+c2)
            out(enodes(2)) = out(enodes(2)) - (c1+c2)
        end do
    end subroutine se_inflow

    !===========================================================================
    subroutine se_visc(dyn, mesh, partit, dU, out)
        ! V(dU) on the OWNED elements, a gather over the slots k = 1..elem2D_nnodes(e) in fixed
        ! order (plan D2):
        !     V_e = sum_k kappa(k,e)*H_edge*(dU_nb/H_nb - dU_e/H_e),   H_edge = (H_e + H_nb)/2
        ! nb = elem_neighbors(k,e), H = H^n (se_H, frozen over the substeps). The linear
        ! harmonic flux form A_h*H_edge*grad(U/H) of the notes (plan D5): V of a uniform velocity
        ! U/H is 0, and sum_e A_e*V_e = 0 (an interior edge enters from both sides with opposite
        ! signs; se_kappa_build). Reads dU and H at eDim neighbours: dU must be halo-valid.
        type(t_dyn),    intent(in)  :: dyn
        type(t_mesh),   intent(in)  :: mesh
        type(t_partit), intent(in)  :: partit
        real(kind=WP),  intent(in)  :: dU(:,:)      ! (2, >= nElemL)
        real(kind=WP),  intent(out) :: out(:,:)     ! (2, >= nElemO)
        integer       :: e, k, nb
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: He, Hnb, kh, qx, qy, vx, vy

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        do e = 1, nElemO
            He = dyn%se_H(e)
            qx = dU(1, e)/He
            qy = dU(2, e)/He
            vx = 0.0_WP
            vy = 0.0_WP
            do k = 1, mesh%elem2D_nnodes(e)
                nb  = mesh%elem_neighbors(k, e)
                Hnb = dyn%se_H(nb)
                kh  = dyn%se_kappa(k, e)*(0.5_WP*(He + Hnb))
                vx  = vx + kh*(dU(1, nb)/Hnb - qx)
                vy  = vy + kh*(dU(2, nb)/Hnb - qy)
            end do
            out(1, e) = vx
            out(2, e) = vy
        end do
    end subroutine se_visc

    !===========================================================================
    subroutine se_depth_at(dyn, mesh, partit, eta, HA)
        ! H_e(eta) = H^n_e + sum_v elem_w(v,e)*c(v,e)*(eta_v - eta^n_v) on the OWNED elements:
        ! the one depth function of the substeps (plan D3), no ALE branch in the loop. Under
        ! linfs c == 0 and H_e(eta) = H^n_e exactly. eta and eta^n (eta_n) are read at the
        ! element's vertices (owned + halo).
        type(t_dyn),    intent(in)  :: dyn
        type(t_mesh),   intent(in)  :: mesh
        type(t_partit), intent(in)  :: partit
        real(kind=WP),  intent(in)  :: eta(:)       ! (nNodL)
        real(kind=WP),  intent(out) :: HA(:)        ! (>= nElemO)
        integer :: e, elnodes(MAX_NV)
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        do e = 1, nElemO
            elnodes = mesh%elem2D_nodes(:, e)
            HA(e) = dyn%se_H(e) + sum(mesh%elem_w(:, e)*dyn%se_c(:, e)*(eta(elnodes) - dyn%eta_n(elnodes)))
        end do
    end subroutine se_depth_at

    !===========================================================================
    subroutine se_tendency(dyn, mesh, partit, ustar, etaA, HA, dU, tend)
        ! The right-hand side of the U substep on the OWNED elements (plan Technical Details §2):
        !     tend = -f e_z x U* - g H^A grad(eta^A) + F + V(dU)
        ! -f e_z x U* = f*(U*_2, -U*_1), compute_vel_rhs's (v*ff, -u*ff); -g grad(eta^A) = Fx, Fy
        ! through gradient_sca exactly as in compute_vel_rhs and se_bt_forcing. So with U* = s^n
        ! = s^{n-1}, eta^A = eta^n, H^A = H^n and dU = 0 the r-term inside F cancels and tend =
        ! G/tau to rounding (test_ssh_se C7: the sign/factor consistency of F and the operator).
        ! V(dU) only when se_visc_ah > 0. Reads eta^A at the element's vertices, dU at eDim
        ! neighbours (se_visc).
        type(t_dyn),    intent(in)  :: dyn
        type(t_mesh),   intent(in)  :: mesh
        type(t_partit), intent(in)  :: partit
        real(kind=WP),  intent(in)  :: ustar(:,:)   ! (2, >= nElemO) U*, the AB3 transport
        real(kind=WP),  intent(in)  :: etaA(:)      ! (nNodL) eta^A, the AM4 elevation
        real(kind=WP),  intent(in)  :: HA(:)        ! (>= nElemO) H^A = H_e(eta^A)
        real(kind=WP),  intent(in)  :: dU(:,:)      ! (2, >= nElemL) U^m - U^n
        real(kind=WP),  intent(out) :: tend(:,:)    ! (2, >= nElemO)
        integer       :: e, elnodes(MAX_NV)
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: f, Fx, Fy, pre(MAX_NV)

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        if (dyn%se_visc_ah > 0.0_WP) then
            call se_visc(dyn, mesh, partit, dU, tend)
        else
            do e = 1, nElemO
                tend(:, e) = 0.0_WP
            end do
        end if
        do e = 1, nElemO
            elnodes = mesh%elem2D_nodes(:, e)
            pre = -(g*etaA(elnodes))
            Fx  = sum(mesh%gradient_sca(1:4, e)*pre)
            Fy  = sum(mesh%gradient_sca(5:8, e)*pre)
            f   = mesh%coriolis(e)
            tend(1, e) =  f*ustar(2, e) + HA(e)*Fx + dyn%se_force(1, e) + tend(1, e)
            tend(2, e) = -f*ustar(1, e) + HA(e)*Fy + dyn%se_force(2, e) + tend(2, e)
        end do
    end subroutine se_tendency

    !===========================================================================
    subroutine se_subcycle(dyn, mesh, partit, dt)
        ! The M barotropic substeps m = 0..M-1 of one baroclinic step tau = dt, after
        ! se_bt_forcing (plan Technical Details §2), dtbt = tau/M:
        !     U*        = (3/2+beta) U^m - (1/2+2 beta) U^{m-1} + beta U^{m-2}     owned elements
        !     <<U>>    += U*/M                                                   owned elements
        !     eta^{m+1} = eta^m + I(U*)*dtbt/areasvol            owned nodes, then exchange_nod
        !     eta^A     = delta eta^{m+1} + (1-delta-gamma-zeta) eta^m + gamma eta^{m-1}
        !                 + zeta eta^{m-2}                                       owned + halo nodes
        !     H^A       = H_e(eta^A)                                             owned elements
        !     U^{m+1}   = U^m + dtbt*se_tendency(U*, eta^A, H^A, U^m - U^n)      owned elements
        ! eta^0 = eta_n (= eta^n), U^0 = ring level 0. The rings shift BY COPYING, so no ring
        ! offset becomes restart state. With the viscosity on, exchange_elem(U^{m+1}) follows:
        ! V reads U at eDim neighbours. Without it nothing reads U off the owned elements (I reads
        ! owned elements only), the exchange is skipped and the eDim entries of the U ring stay
        ! stale (and unused). Per substep: one 2-D exchange_nod and, with the viscosity, one 2-D
        ! exchange_elem; no reductions. Timer TMR_OCE_SE_BT.
        ! On exit: eta^M in se_etam (eta^{M-1}, eta^{M-2} in se_eta(:,1), se_eta(:,2); all on
        ! owned + halo nodes), U^M in ring level 0 (U^{M-1}, U^{M-2} in levels 1, 2), <<U>> in
        ! se_umean (owned elements). eta_n and hbar are untouched: se_trim_finish commits them.
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: dt       ! tau, the baroclinic step
        integer       :: m, M_sub, n, e
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: rM, dtbt, w0, w1, w2, d0, d1, d2, d3
        logical       :: visc
        real(kind=WP), allocatable :: ustar(:,:), unew(:,:), tend(:,:), dU(:,:)
        real(kind=WP), allocatable :: etaA(:), inflow(:), HA(:)

        call timer_start(TMR_OCE_SE_BT)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        M_sub = dyn%se_M
        rM    = real(M_sub, WP)
        dtbt  = dt/rM
        ! AB3: w0 U^m + w1 U^{m-1} + w2 U^{m-2};  AM4: d3 eta^{m+1} + d0 eta^m + d1 eta^{m-1} + d2 eta^{m-2}
        w0 =  1.5_WP + dyn%se_beta
        w1 = -(0.5_WP + 2.0_WP*dyn%se_beta)
        w2 =  dyn%se_beta
        d3 =  dyn%se_delta
        d0 =  1.0_WP - dyn%se_delta - dyn%se_gamma - dyn%se_zeta
        d1 =  dyn%se_gamma
        d2 =  dyn%se_zeta
        visc = (dyn%se_visc_ah > 0.0_WP)

        allocate(ustar(2, nElemL), unew(2, nElemL), tend(2, nElemL), dU(2, nElemL))
        allocate(etaA(nNodL), inflow(nNodL), HA(nElemL))
        ustar = 0.0_WP; unew = 0.0_WP; tend = 0.0_WP; dU = 0.0_WP
        etaA  = 0.0_WP; inflow = 0.0_WP; HA = 0.0_WP

        do n = 1, nNodL
            dyn%se_etam(n) = dyn%eta_n(n)
        end do
        dyn%se_umean = 0.0_WP

        do m = 0, M_sub - 1
            ! AB3 transport and the running mean
            do e = 1, nElemO
                ustar(1, e) = w0*dyn%se_ubt(1, e, 0) + w1*dyn%se_ubt(1, e, 1) + w2*dyn%se_ubt(1, e, 2)
                ustar(2, e) = w0*dyn%se_ubt(2, e, 0) + w1*dyn%se_ubt(2, e, 1) + w2*dyn%se_ubt(2, e, 2)
                dyn%se_umean(1, e) = dyn%se_umean(1, e) + ustar(1, e)/rM
                dyn%se_umean(2, e) = dyn%se_umean(2, e) + ustar(2, e)/rM
            end do

            ! elevation (closed basin, linfs: no freshwater term -- Task 3.1)
            call se_inflow(mesh, partit, ustar, inflow)
            do n = 1, nNodO
                dyn%se_etanew(n) = dyn%se_etam(n) + inflow(n)*dtbt/mesh%areasvol(n)
            end do
            if (is_multirank(partit)) call exchange_nod(dyn%se_etanew, partit)

            ! AM4 elevation and its depth, on the levels held on owned + halo nodes
            do n = 1, nNodL
                etaA(n) = d3*dyn%se_etanew(n) + d0*dyn%se_etam(n) + d1*dyn%se_eta(n, 1) &
                        + d2*dyn%se_eta(n, 2)
            end do
            call se_depth_at(dyn, mesh, partit, etaA, HA)

            ! transport
            if (visc) then
                do e = 1, nElemL
                    dU(1, e) = dyn%se_ubt(1, e, 0) - dyn%se_ubtn(1, e)
                    dU(2, e) = dyn%se_ubt(2, e, 0) - dyn%se_ubtn(2, e)
                end do
            end if
            call se_tendency(dyn, mesh, partit, ustar, etaA, HA, dU, tend)
            do e = 1, nElemO
                unew(1, e) = dyn%se_ubt(1, e, 0) + dtbt*tend(1, e)
                unew(2, e) = dyn%se_ubt(2, e, 0) + dtbt*tend(2, e)
            end do

            ! shift the rings by copying
            do n = 1, nNodL
                dyn%se_eta(n, 2) = dyn%se_eta(n, 1)
                dyn%se_eta(n, 1) = dyn%se_etam(n)
                dyn%se_etam(n)   = dyn%se_etanew(n)
            end do
            do e = 1, nElemL
                dyn%se_ubt(:, e, 2) = dyn%se_ubt(:, e, 1)
                dyn%se_ubt(:, e, 1) = dyn%se_ubt(:, e, 0)
            end do
            do e = 1, nElemO
                dyn%se_ubt(:, e, 0) = unew(:, e)
            end do
            if (visc) then
                if (is_multirank(partit)) call exchange_elem(dyn%se_ubt(:, :, 0), partit)
            end if
        end do

        deallocate(ustar, unew, tend, dU, etaA, inflow, HA)
        call timer_stop(TMR_OCE_SE_BT)
    end subroutine se_subcycle

    !===========================================================================
    ! ---- helpers ----

    subroutine get_knob(name, val, is_set)
        ! The environment variable name; is_set = present and non-empty. A value too long for
        ! val comes back as '<too long>', which every parser rejects.
        character(len=*), intent(in)  :: name
        character(len=*), intent(out) :: val
        logical,          intent(out) :: is_set
        integer :: env_len, ios
        val = ''
        call get_environment_variable(name, val, length=env_len, status=ios)
        is_set = (ios == 0 .or. ios == -1) .and. env_len > 0
        if (ios == -1) val = '<too long>'
    end subroutine get_knob

    pure subroutine parse_01(str, v, ok)
        character(len=*), intent(in)  :: str
        logical,          intent(out) :: v, ok
        v  = .false.
        ok = .true.
        select case (trim(adjustl(str)))
        case ('0')
            v = .false.
        case ('1')
            v = .true.
        case default
            ok = .false.
        end select
    end subroutine parse_01

    subroutine parse_int(str, v, ok)
        ! digits only, at most 9 (no sign, no exponent, no overflow)
        character(len=*), intent(in)  :: str
        integer,          intent(out) :: v
        logical,          intent(out) :: ok
        character(len=len(str)) :: s
        integer :: ls, ios
        v  = 0
        s  = adjustl(str)
        ls = len_trim(s)
        ok = (ls >= 1 .and. ls <= 9)
        if (ok) ok = (verify(s(1:ls), '0123456789') == 0)
        if (.not. ok) return
        read(s(1:ls), *, iostat=ios) v
        ok = (ios == 0)
    end subroutine parse_int

    subroutine parse_real(str, v, ok)
        ! the characters of a plain number only (digits, sign, point, exponent letter), then a
        ! list-directed read that must succeed and give a finite value
        character(len=*), intent(in)  :: str
        real(kind=WP),    intent(out) :: v
        logical,          intent(out) :: ok
        character(len=len(str)) :: s
        integer :: ls, ios
        v  = 0.0_WP
        s  = adjustl(str)
        ls = len_trim(s)
        ok = (ls >= 1 .and. ls <= 64)
        if (ok) ok = (verify(s(1:ls), '0123456789+-.eEdD') == 0)
        if (.not. ok) return
        read(s(1:ls), *, iostat=ios) v
        ok = (ios == 0)
        if (ok) ok = ieee_is_finite(v)
    end subroutine parse_real

    subroutine se_die(msg, partit)
        ! Print msg and abort every rank (MPI_Abort on partit's communicator, MPI_COMM_WORLD
        ! without one); error stop where MPI is not running.
        character(len=*), intent(in)           :: msg
        type(t_partit),   intent(in), optional :: partit
        integer :: comm, ierr
        logical :: inited
        write(*,'(a)') 'oce_ssh_se: '//msg
        flush(output_unit)
        call MPI_Initialized(inited, ierr)
        if (inited) then
            comm = MPI_COMM_WORLD
            if (present(partit)) comm = partit%MPI_COMM_FESOM
            call MPI_Abort(comm, 1, ierr)
        end if
        error stop 1
    end subroutine se_die

end module oce_ssh_se
