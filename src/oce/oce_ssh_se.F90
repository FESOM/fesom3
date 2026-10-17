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
    ! IN PLACE (Task 0.3): the knobs, the state in t_dyn, se_init (coefficients, cold start, CFL
    ! guard) and the step_oce guard. The step itself (se_bt_forcing, se_subcycle, se_trim_finish)
    ! is Stage 1 of the plan.
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
    use mod_constants,   only: g
    use mod_mesh,        only: t_mesh, MAX_NV
    use mod_dyn,         only: t_dyn
    use mod_partit,      only: t_partit
    use mod_part_bounds, only: local_dims
    implicit none
    private
    public :: se_env_requested, se_read_env, se_parse_knob, se_init, se_step_guard
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
        ! cold start, the CFL guard, and se_ready.
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
