program test_rk3
    ! RK3 time stepping (docs/plans/2026-10-28-rk3.md): the knobs, the kernel hooks, the stage
    ! building blocks and the stage-3 tracer solve of src/oce/oce_rk3.F90 and its callers, on the
    ! rotated pi mesh. Each part pins one property exactly (L54).
    !
    !   K   the knob parser (rk3_parse_knob): every valid value is accepted and lands in t_dyn,
    !       every bad one is rejected (ok = .false., the t_dyn field unchanged, the message starts
    !       with the knob name); the t_dyn defaults (RK3 off, one LO step, check off, tol 0); an
    !       unknown name is rejected; FESOM3_RK3 is unset under ctest, so rk3_env_requested is
    !       .false. (the driver-facing stops are the fesom_conserve_rk3_* ctest cases)
    use mpi
    use mod_precision,    only: WP
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init
    use mod_dyn,          only: t_dyn
    use oce_rk3,          only: rk3_env_requested, rk3_parse_knob
    use test_util,        only: setup_test_mesh, check, finish
    implicit none

    character(len=*), parameter :: PI_DIR = '/home/a/a270088/port2/fesom2/tests/data/MESHES/pi'
    type(t_partit) :: partit
    type(t_mesh)   :: mesh
    type(t_dyn)    :: d_last        ! K: accumulates the parsed values (land)

    call par_init(partit)
    call setup_test_mesh(mesh, partit, PI_DIR, cartesian=.false., partition_np1=.true.)

    call k_knobs()
    call finish(partit, 'test_rk3')

contains

    !===========================================================================
    subroutine k_knobs()
        type(t_dyn) :: d
        integer :: mype
        mype = partit%mype

        call check(.not. d%use_rk3,           'K default: RK3 off', mype)
        call check(d%rk3_lo_nsub == 1,        'K default: one LO step', mype)
        call check(.not. d%rk3_check,         'K default: check off', mype)
        call check(d%rk3_check_tol == 0.0_WP, 'K default: tol 0', mype)
        call check(.not. d%rk3_ready,         'K default: not ready', mype)
        call check(.not. rk3_env_requested(partit), 'K FESOM3_RK3 unset under ctest -> off', mype)

        call good('FESOM3_RK3', '1');  call good('FESOM3_RK3', '0'); call good('FESOM3_RK3', ' 1 ')
        call bad('FESOM3_RK3', '2');   call bad('FESOM3_RK3', 'yes'); call bad('FESOM3_RK3', '01')
        call bad('FESOM3_RK3', '');    call bad('FESOM3_RK3', '<too long>')
        call good('FESOM3_RK3_LO_NSUB', '1'); call good('FESOM3_RK3_LO_NSUB', '2')
        call good('FESOM3_RK3_LO_NSUB', '3')
        call bad('FESOM3_RK3_LO_NSUB', '0');  call bad('FESOM3_RK3_LO_NSUB', '4')
        call bad('FESOM3_RK3_LO_NSUB', '-1'); call bad('FESOM3_RK3_LO_NSUB', '2.0')
        call bad('FESOM3_RK3_LO_NSUB', 'two'); call bad('FESOM3_RK3_LO_NSUB', '')
        call bad('FESOM3_RK3_LO_NSUB', '1 2')
        call good('FESOM3_RK3_CHECK', '1'); call good('FESOM3_RK3_CHECK', '0')
        call bad('FESOM3_RK3_CHECK', 'true'); call bad('FESOM3_RK3_CHECK', '-1')
        call good('FESOM3_RK3_CHECK_TOL', '0'); call good('FESOM3_RK3_CHECK_TOL', '1e-10')
        call good('FESOM3_RK3_CHECK_TOL', '2.5D-3')
        call bad('FESOM3_RK3_CHECK_TOL', '-1e-10'); call bad('FESOM3_RK3_CHECK_TOL', 'nan')
        call bad('FESOM3_RK3_CHECK_TOL', 'inf');    call bad('FESOM3_RK3_CHECK_TOL', '1e999')
        call bad('FESOM3_RK3_CHECK_TOL', '1e-10 2'); call bad('FESOM3_RK3_CHECK_TOL', '')
        call bad('FESOM3_RK3_BOGUS', '1')

        ! values land in t_dyn
        call land('FESOM3_RK3', '1');              call check(d_last%use_rk3, 'K FESOM3_RK3=1 -> use_rk3', mype)
        call land('FESOM3_RK3_LO_NSUB', '3');      call check(d_last%rk3_lo_nsub == 3, 'K LO_NSUB=3 lands', mype)
        call land('FESOM3_RK3_CHECK', '1');        call check(d_last%rk3_check, 'K CHECK=1 lands', mype)
        call land('FESOM3_RK3_CHECK_TOL', '1e-9'); call check(d_last%rk3_check_tol == 1.0e-9_WP, &
                                                              'K CHECK_TOL=1e-9 lands', mype)
    end subroutine k_knobs

    subroutine good(name, str)
        character(len=*), intent(in) :: name, str
        type(t_dyn) :: d
        logical :: ok
        character(len=256) :: msg
        call rk3_parse_knob(name, str, d, ok, msg)
        call check(ok .and. len_trim(msg) == 0, 'K accepts '//name//'="'//str//'"', partit%mype)
    end subroutine good

    subroutine bad(name, str)
        ! rejected, the message names the knob, every t_dyn field keeps its default
        character(len=*), intent(in) :: name, str
        type(t_dyn) :: d
        logical :: ok
        character(len=256) :: msg
        call rk3_parse_knob(name, str, d, ok, msg)
        call check(.not. ok, 'K rejects '//name//'="'//str//'"', partit%mype)
        call check(index(msg, name) == 1, 'K message of '//name//'="'//str//'" starts with the name', partit%mype)
        call check(.not. d%use_rk3 .and. d%rk3_lo_nsub == 1 .and. .not. d%rk3_check .and. &
                   d%rk3_check_tol == 0.0_WP, 'K '//name//'="'//str//'" leaves t_dyn unchanged', partit%mype)
    end subroutine bad

    subroutine land(name, str)
        character(len=*), intent(in) :: name, str
        logical :: ok
        character(len=256) :: msg
        call rk3_parse_knob(name, str, d_last, ok, msg)
        call check(ok, 'K parses '//name//'="'//str//'"', partit%mype)
    end subroutine land

end program test_rk3
