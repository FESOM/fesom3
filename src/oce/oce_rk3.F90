module oce_rk3
    ! RK3 time stepping of the ocean, an alternative to the AB2 step (step_oce). Plan:
    ! docs/plans/2026-10-28-rk3.md. The three-stage Wicker-Skamarock Runge-Kutta scheme of Sergey
    ! Danilov's quad FV prototype (~/qq_new2 oce_stepRK3.F90, oce_stepRK3_SE.F90) and of his notes
    ! (rk3.tex), self-starting (no AB history):
    !     phi*    = phi^n + tau/3*F(phi^n)
    !     phi**   = phi^n + tau/2*F(phi*)
    !     phi^n+1 = phi^n + tau  *F(phi**)
    ! Stages 1-2 are cheap: tracer advection (high order only, no limiter); momentum advection,
    ! Coriolis, the pressure gradient and -g grad(eta^n). Both advect with the FULL w of the stage
    ! (rk3.tex: every stage diagnoses its own w), so a uniform tracer stays uniform in every stage.
    ! Everything else is done once, in stage 3: viscosity, vertical mixing, diffusion, GM/Redi,
    ! the surface fluxes, wind and drag, the FCT limiter and the w split (plan D7). The barotropic
    ! part is the semi-implicit CG corrector of step_oce or the split-explicit substeps of
    ! oce_ssh_se (FESOM3_SSH_SE=1), fed by an RK3 forcing. The stage driver is step_oce_rk3
    ! (src/step/mod_step_oce_rk3.F90); this module holds the knobs, the setup and the stage
    ! building blocks.
    !
    ! KNOBS (env only, like every run switch of FESOM_F; read only when FESOM3_RK3=1):
    !   FESOM3_RK3            0 (default) = AB2 (step_oce, byte-identical), 1 = RK3
    !   FESOM3_RK3_LO_NSUB    FCT low-order substeps in stage 3, 1, 2 or 3            default 1
    !                         (1 = today's low order with the w split; 2, 3 only for a horizontal
    !                         low-order Courant number near 1)
    !   FESOM3_RK3_CHECK      1 = per-step invariant line RK3_CHECK (0/1)               default 0
    !   FESOM3_RK3_CHECK_TOL  stop when an RK3_CHECK value exceeds it, a real >= 0;    default 0
    !                         0 = report only
    ! Value-checked like the FESOM3_SE_* knobs: anything outside the range stops the run. RK3 is
    ! SILENT when off (no header, no check line, quiet timers, no checkpoint key):
    ! run_conserve_pi.sh and the G0 gate compare fixed output.
    !
    ! GUARDS (rk3_init): linfs only until zstar lands (plan Stage 6), no zlevel; double precision
    ! only; no cavities; alpha = theta = 1 (the CG corrector of stage 3 is update_vel /
    ! compute_ssh_rhs_ale with theta = 1, alpha = 1). step_oce stops if FESOM3_RK3=1 reaches it
    ! from a driver that did not run rk3_init (rk3_step_guard).
    !
    ! The parser (rk3_parse_knob) returns its verdict instead of stopping, so test/test_rk3.F90
    ! checks every rejection; the driver-facing wrappers stop through rk3_die (MPI_Abort, so no
    ! rank is left waiting in a collective).
    use mpi
    use, intrinsic :: iso_fortran_env, only: output_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use mod_precision,   only: WP
    use mod_mesh,        only: t_mesh
    use mod_dyn,         only: t_dyn
    use mod_partit,      only: t_partit
    use mod_part_bounds, only: local_dims
    use mod_param_phys,  only: alpha, theta
    use mod_config,      only: which_ALE
    use oce_ssh_se,      only: se_env_requested
    implicit none
    private
    public :: rk3_env_requested, rk3_read_env, rk3_parse_knob, rk3_init, rk3_step_guard, rk3_die

    ! the knobs rk3_read_env reads (FESOM3_RK3 itself: rk3_env_requested)
    character(len=20), parameter :: RK3_KNOBS(3) = [character(len=20) :: &
        'FESOM3_RK3_LO_NSUB', 'FESOM3_RK3_CHECK', 'FESOM3_RK3_CHECK_TOL']

contains

    !===========================================================================
    logical function rk3_env_requested(partit)
        ! FESOM3_RK3, parsed ONCE and cached: unset or 0 = AB2, 1 = RK3, any other value stops.
        ! The one place that decides whether RK3 is wanted: rk3_init and the step_oce guard both
        ! ask here. partit (optional, as in step_oce) only selects the communicator of the stop.
        type(t_partit), intent(in), optional :: partit
        integer, save :: state = -1           ! -1 = not parsed yet, 0 = off, 1 = on
        type(t_dyn)        :: scratch
        character(len=256) :: val, msg
        logical :: is_set, ok

        if (state < 0) then
            state = 0
            call get_knob('FESOM3_RK3', val, is_set)
            if (is_set) then
                call rk3_parse_knob('FESOM3_RK3', val, scratch, ok, msg)
                if (.not. ok) call rk3_die(trim(msg), partit)
                if (scratch%use_rk3) state = 1
            end if
        end if
        rk3_env_requested = (state == 1)
    end function rk3_env_requested

    !===========================================================================
    subroutine rk3_read_env(dyn, partit)
        ! The FESOM3_RK3_* knobs into dyn; an unset knob keeps its t_dyn default (one LO step,
        ! check off, tolerance 0). Stops on a bad value. Called by rk3_init.
        type(t_dyn),    intent(inout) :: dyn
        type(t_partit), intent(in)    :: partit
        character(len=256) :: val, msg
        logical :: is_set, ok
        integer :: i

        do i = 1, size(RK3_KNOBS)
            call get_knob(trim(RK3_KNOBS(i)), val, is_set)
            if (.not. is_set) cycle
            call rk3_parse_knob(trim(RK3_KNOBS(i)), val, dyn, ok, msg)
            if (.not. ok) call rk3_die(trim(msg), partit)
        end do
    end subroutine rk3_read_env

    !===========================================================================
    subroutine rk3_parse_knob(name, str, dyn, ok, msg)
        ! Parse the text str of the RK3 knob name (FESOM3_RK3 or one of FESOM3_RK3_*) into its
        ! t_dyn field. A bad value leaves the field unchanged and returns ok = .false. with msg =
        ! '<name> must ...: "<value>"'. No output and no stop: the caller decides. Switches
        ! accept exactly '0' or '1'; FESOM3_RK3_LO_NSUB the digits 1, 2 or 3; the tolerance the
        ! characters of a plain number and a finite result >= 0.
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
        case ('FESOM3_RK3')
            need = 'must be 0 or 1'
            call parse_01(str, lv, ok)
            if (ok) dyn%use_rk3 = lv
        case ('FESOM3_RK3_LO_NSUB')
            need = 'must be 1, 2 or 3'
            call parse_int(str, iv, ok)
            if (ok) ok = (iv >= 1 .and. iv <= 3)
            if (ok) dyn%rk3_lo_nsub = iv
        case ('FESOM3_RK3_CHECK')
            need = 'must be 0 or 1'
            call parse_01(str, lv, ok)
            if (ok) dyn%rk3_check = lv
        case ('FESOM3_RK3_CHECK_TOL')
            need = 'must be a real >= 0'
            call parse_real(str, rv, ok)
            if (ok) ok = (rv >= 0.0_WP)
            if (ok) dyn%rk3_check_tol = rv
        case default
            need = 'is not an RK3 knob'
        end select
        msg = ''
        if (.not. ok) msg = name//' '//trim(need)//': "'//trim(adjustl(str))//'"'
    end subroutine rk3_parse_knob

    !===========================================================================
    subroutine rk3_init(dyn, mesh, partit)
        ! Called once by a driver that supports RK3 (fesom_lifecycle_native_mr, fesom_conserve),
        ! after its cold init and right before se_init. A silent no-op unless FESOM3_RK3=1.
        ! Otherwise: the knobs, the guards, the stage arrays and one header line on rank 0.
        ! rk3_H is not set here: a restart rebuilds the geometry after this call, so the step
        ! rebuilds it every time (step_oce_rk3).
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, nl
        character(len=8) :: bt

        if (.not. rk3_env_requested(partit)) return
        if (dyn%rk3_ready) call rk3_die('rk3_init called twice', partit)
        call rk3_read_env(dyn, partit)
        dyn%use_rk3 = .true.
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        nl = mesh%nl

        ! ---- guards ----
#if defined(USE_SINGLE_PRECISION)
        call rk3_die('RK3 needs a double-precision build (this one is USE_SINGLE_PRECISION)', partit)
#endif
        if (trim(which_ALE) /= 'linfs') &
            call rk3_die('RK3 supports which_ALE = linfs, not '//trim(which_ALE), partit)
        if (any(mesh%ulevels_nod2D(1:nNodL) > 1)) &
            call rk3_die('RK3 does not support cavities (ulevels_nod2D > 1)', partit)
        if (alpha /= 1.0_WP .or. theta /= 1.0_WP) &
            call rk3_die('RK3 needs alpha = theta = 1 (the stage-3 CG corrector)', partit)

        ! ---- stage state ----
        allocate(dyn%rk3_us(2, nl-1, nElemF), dyn%rk3_visc(2, nl-1, nElemF), dyn%rk3_H(nElemL))
        dyn%rk3_us = 0.0_WP; dyn%rk3_visc = 0.0_WP; dyn%rk3_H = 0.0_WP
        if (dyn%rk3_check) then
            allocate(dyn%rk3_sbar0(2, nElemL))
            dyn%rk3_sbar0 = 0.0_WP
        end if

        if (partit%mype == 0) then
            bt = 'CG'
            if (se_env_requested(partit)) bt = 'SE'
            write(*,'(a,a,a,i0,a,l1,a,es9.2)') 'oce_rk3: RK3 time stepping ON (Wicker-Skamarock; ' // &
                'barotropic ', trim(bt), ', FCT LO substeps ', dyn%rk3_lo_nsub, ', check ', dyn%rk3_check, &
                ', tol ', dyn%rk3_check_tol
            write(*,'(a)') 'oce_rk3:   stages 1-2: advection (full w), Coriolis, PGF; stage 3: the rest, w split'
        end if
        dyn%rk3_ready = .true.
    end subroutine rk3_init

    !===========================================================================
    subroutine rk3_step_guard(dyn, partit)
        ! step_oce's first-call check. FESOM3_RK3=1 needs a driver that ran rk3_init
        ! (fesom_lifecycle_native_mr, fesom_conserve): any other driver would silently run AB2.
        ! The RK3 step takes the partition as a required argument, so such a driver must also
        ! pass partit to step_oce.
        type(t_dyn),    intent(in)           :: dyn
        type(t_partit), intent(in), optional :: partit
        if (.not. rk3_env_requested(partit)) return
        if (.not. dyn%rk3_ready) &
            call rk3_die('FESOM3_RK3=1 but this driver does not initialise RK3 (rk3_init)', partit)
        if (.not. present(partit)) &
            call rk3_die('FESOM3_RK3=1 but step_oce got no partit (RK3 needs it)', partit)
    end subroutine rk3_step_guard

    !===========================================================================
    ! ---- helpers (the knob helpers of oce_ssh_se are private there) ----

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

    subroutine rk3_die(msg, partit)
        ! Print msg and abort every rank (MPI_Abort on partit's communicator, MPI_COMM_WORLD
        ! without one); error stop where MPI is not running.
        character(len=*), intent(in)           :: msg
        type(t_partit),   intent(in), optional :: partit
        integer :: comm, ierr
        logical :: inited
        write(*,'(a)') 'oce_rk3: '//msg
        flush(output_unit)
        call MPI_Initialized(inited, ierr)
        if (inited) then
            comm = MPI_COMM_WORLD
            if (present(partit)) comm = partit%MPI_COMM_FESOM
            call MPI_Abort(comm, 1, ierr)
        end if
        error stop 1
    end subroutine rk3_die

end module oce_rk3
