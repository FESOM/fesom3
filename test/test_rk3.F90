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
    !   R1  the momentum hooks of compute_vel_rhs (Task 3), on a hashed state (uv, eta_n, pgf, the
    !       AB history; w from vert_vel_ale):
    !       a) uv_in = a copy of dyn%uv gives the same uv_rhs and uv_rhsAB as no argument, bitwise
    !          on the owned elements (momadv 1 and 2, rvo_upwind 0 and 1, use_wsplit off and on);
    !          completeness: uv_in = u2 with dyn%uv = u1 equals the default call with dyn%uv = u2,
    !          bitwise (no kernel of the rhs still reads dyn%uv);
    !       b) no_history with a nonzero AB history equals lfirst with a zero history, bitwise
    !          (uv_rhs and uv_rhsAB), also with uv_in
    use mpi
    use mod_precision,    only: WP, MP
    use mod_constants,    only: density_0
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init
    use mod_dyn,          only: t_dyn
    use mod_part_bounds,  only: local_dims
    use mod_param_phys,   only: rvo_upwind, Fer_GM, N2smth_h
    use oce_rk3,          only: rk3_env_requested, rk3_parse_knob
    use oce_dyn_velrhs,   only: compute_vel_rhs
    use oce_ale,          only: vert_vel_ale
    use oce_pressure_bv,  only: pressure_bv
    use test_util,        only: setup_test_mesh, check, finish, sum_all, hval
    implicit none

    character(len=*), parameter :: PI_DIR = '/home/a/a270088/port2/fesom2/tests/data/MESHES/pi'
    real(kind=WP),    parameter :: dt = 86400.0_WP / real(48, WP)   ! the pi/CORE2 timestep
    type(t_partit) :: partit
    type(t_mesh)   :: mesh
    type(t_dyn)    :: d_last        ! K: accumulates the parsed values (land)
    integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, nl

    call par_init(partit)
    call setup_test_mesh(mesh, partit, PI_DIR, cartesian=.false., partition_np1=.true.)
    call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
    nl = mesh%nl
    call setup_thickness()

    call k_knobs()
    call r1_momentum_hooks()
    call finish(partit, 'test_rk3')

contains

    !===========================================================================
    ! ---- state ----

    subroutine setup_thickness()
        ! the linfs full-cell thicknesses and depths of the drivers (fesom_conserve), cold
        ! elevation (hbar = hbar_old = 0), hnode_new = hnode
        integer :: n, e, nz, nzmin, nzmax
        real(kind=MP) :: zbar_srf, zbar_bot
        allocate(mesh%hnode(nl-1, nNodL)); mesh%hnode = 0.0_MP
        do n = 1, nNodL
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n)-1
                mesh%hnode(nz, n) = mesh%zbar(nz) - mesh%zbar(nz+1)
            end do
        end do
        allocate(mesh%helem(nl-1, nElemF)); mesh%helem = 0.0_MP
        allocate(mesh%zbar_e_bot(nElemF)); mesh%zbar_e_bot = 0.0_MP
        do e = 1, nElemF
            if (mesh%nlevels(e) <= 0) cycle
            do nz = mesh%ulevels(e), mesh%nlevels(e)-1
                mesh%helem(nz, e) = mesh%zbar(nz) - mesh%zbar(nz+1)
            end do
            mesh%zbar_e_bot(e) = mesh%zbar(mesh%nlevels(e))
        end do
        allocate(mesh%zbar_3d_n(nl, nNodL), mesh%Z_3d_n(nl-1, nNodL))
        mesh%zbar_3d_n = 0.0_MP; mesh%Z_3d_n = 0.0_MP
        do n = 1, nNodL
            nzmin = mesh%ulevels_nod2D(n); nzmax = mesh%nlevels_nod2D(n)
            zbar_srf = mesh%zbar(nzmin); zbar_bot = mesh%zbar(nzmax)
            mesh%zbar_3d_n(1:nzmin-1, n)       = mesh%zbar(1:nzmin-1)
            mesh%zbar_3d_n(nzmin, n)           = zbar_srf
            mesh%zbar_3d_n(nzmin+1:nzmax-1, n) = mesh%zbar(nzmin+1:nzmax-1)
            mesh%zbar_3d_n(nzmax, n)           = zbar_bot
            mesh%Z_3d_n(1:nzmin-1, n)          = mesh%Z(1:nzmin-1)
            mesh%Z_3d_n(nzmin, n)              = mesh%zbar_3d_n(nzmin,n) + (mesh%zbar_3d_n(nzmin+1,n)-zbar_srf)/2
            mesh%Z_3d_n(nzmin+1:nzmax-2, n)    = mesh%Z(nzmin+1:nzmax-2)
            mesh%Z_3d_n(nzmax-1, n)            = mesh%zbar_3d_n(nzmax-1,n) + (zbar_bot-mesh%zbar_3d_n(nzmax-1,n))/2
        end do
        allocate(mesh%hbar(nNodL), mesh%hbar_old(nNodL), mesh%dhe(nElemF), mesh%hnode_new(nl-1, nNodL))
        mesh%hbar = 0.0_MP; mesh%hbar_old = 0.0_MP; mesh%dhe = 0.0_MP
        mesh%hnode_new = mesh%hnode
    end subroutine setup_thickness

    subroutine new_dyn(d, momadv)
        ! a zero dynamics state at the extents the drivers allocate (fesom_conserve)
        type(t_dyn), intent(inout) :: d
        integer,     intent(in)    :: momadv
        allocate(d%uv(2, nl-1, nElemF), d%uv_rhs(2, nl-1, nElemF), d%uv_rhsAB(1, 2, nl-1, nElemF))
        allocate(d%uvnode(2, nl-1, nNodL), d%eta_n(nNodL), d%d_eta(nNodL))
        allocate(d%ssh_rhs(nNodL), d%ssh_rhs_old(nNodL))
        allocate(d%w(nl, nNodL), d%w_e(nl, nNodL), d%w_i(nl, nNodL), d%cfl_z(nl, nNodL))
        allocate(d%work%pgf_x(nl-1, nElemF), d%work%pgf_y(nl-1, nElemF))
        allocate(d%work%u_c(nl-1, nElemF), d%work%v_c(nl-1, nElemF))
        allocate(d%work%uvnode_rhs(2, nl-1, nNodL), d%work%vorticity(nl-1, nNodL))
        allocate(d%work%Kv(nl, nNodL), d%work%Av(nl, nElemF))
        d%uv = 0.0_WP; d%uv_rhs = 0.0_WP; d%uv_rhsAB = 0.0_WP; d%uvnode = 0.0_WP
        d%eta_n = 0.0_WP; d%d_eta = 0.0_WP; d%ssh_rhs = 0.0_WP; d%ssh_rhs_old = 0.0_WP
        d%w = 0.0_WP; d%w_e = 0.0_WP; d%w_i = 0.0_WP; d%cfl_z = 0.0_WP
        d%work%pgf_x = 0.0_WP; d%work%pgf_y = 0.0_WP; d%work%u_c = 0.0_WP; d%work%v_c = 0.0_WP
        d%work%uvnode_rhs = 0.0_WP; d%work%vorticity = 0.0_WP; d%work%Kv = 0.0_WP; d%work%Av = 0.0_WP
        d%AB_order   = 2
        d%momadv_opt = momadv
        d%opt_visc   = 7
        d%visc_gamma0 = 0.003_WP; d%visc_gamma1 = 0.1_WP; d%visc_gamma2 = 0.285_WP
    end subroutine new_dyn

    subroutine hashed_uv(u, seed)
        ! a velocity on every held element and wet layer, hashed from the global element id
        ! (every copy of an element holds the same value)
        real(kind=WP), intent(inout) :: u(:,:,:)
        integer,       intent(in)    :: seed
        integer :: e, nz, c
        u = 0.0_WP
        do e = 1, size(u, 3)
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                do c = 1, 2
                    u(c, nz, e) = 0.2_WP*hval(partit%myList_elem2D(e), 10*nz + c + seed)
                end do
            end do
        end do
    end subroutine hashed_uv

    subroutine hashed_state(d, seed)
        ! uv, eta_n, the PGF and the AB history hashed from the global ids
        type(t_dyn), intent(inout) :: d
        integer,     intent(in)    :: seed
        integer :: n, e, nz
        call hashed_uv(d%uv, seed)
        do n = 1, nNodL
            d%eta_n(n) = 0.1_WP*hval(partit%myList_nod2D(n), 7 + seed)
        end do
        do e = 1, nElemF
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                d%work%pgf_x(nz, e) = 1.0e-5_WP*hval(partit%myList_elem2D(e), 3*nz + 1 + seed)
                d%work%pgf_y(nz, e) = 1.0e-5_WP*hval(partit%myList_elem2D(e), 3*nz + 2 + seed)
                d%uv_rhsAB(1, 1, nz, e) = 1.0e-4_WP*hval(partit%myList_elem2D(e), 5*nz + 3 + seed)
                d%uv_rhsAB(1, 2, nz, e) = 1.0e-4_WP*hval(partit%myList_elem2D(e), 5*nz + 4 + seed)
            end do
        end do
    end subroutine hashed_state

    logical function same_elem(a, b)
        ! bitwise equal on the owned elements' wet layers
        real(kind=WP), intent(in) :: a(:,:,:), b(:,:,:)
        integer :: e, nz
        same_elem = .true.
        do e = 1, nElemO
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                if (any(a(:, nz, e) /= b(:, nz, e))) same_elem = .false.
            end do
        end do
    end function same_elem

    logical function same_ab(a, b)
        real(kind=WP), intent(in) :: a(:,:,:,:), b(:,:,:,:)
        same_ab = same_elem(a(1,:,:,:), b(1,:,:,:))
    end function same_ab

    logical function same_nod(a, b)
        ! bitwise equal on every owned + halo node, all levels
        real(kind=WP), intent(in) :: a(:,:), b(:,:)
        same_nod = all(a(:, 1:nNodL) == b(:, 1:nNodL))
    end function same_nod

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

    !===========================================================================
    subroutine r1_momentum_hooks()
        type(t_dyn) :: d
        real(kind=WP), allocatable :: hist(:,:,:,:), r0(:,:,:), ab0(:,:,:,:), u1(:,:,:), u2(:,:,:)
        integer :: momadv, iup, isp, mype
        character(len=40) :: tag
        real(kind=WP) :: rvo_save
        mype = partit%mype
        rvo_save = rvo_upwind
        allocate(u1(2, nl-1, nElemF), u2(2, nl-1, nElemF))
        call hashed_uv(u2, 77)
        do momadv = 1, 2
            do iup = 0, 1
                do isp = 0, 1
                    write(tag, '(a,i0,a,i0,a,i0)') 'momadv=', momadv, ' rvo=', iup, ' wsplit=', isp
                    call new_dyn(d, momadv)
                    rvo_upwind = real(iup, WP)
                    d%use_wsplit = (isp == 1)
                    d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
                    call hashed_state(d, 11)
                    call vert_vel_ale(d, mesh, dt, partit)
                    hist = d%uv_rhsAB
                    u1   = d%uv
                    call compute_vel_rhs(d, mesh, dt, .false., partit)
                    r0 = d%uv_rhs; ab0 = d%uv_rhsAB
                    ! a) transparency: uv_in = a copy of uv
                    d%uv_rhsAB = hist
                    call compute_vel_rhs(d, mesh, dt, .false., partit, uv_in=u1)
                    call check(same_elem(d%uv_rhs, r0) .and. same_ab(d%uv_rhsAB, ab0), &
                               'R1a uv_in = uv is transparent, '//trim(tag), mype)
                    ! a) completeness: uv_in = u2 with uv = u1 == the default with uv = u2
                    d%uv_rhsAB = hist
                    call compute_vel_rhs(d, mesh, dt, .false., partit, uv_in=u2)
                    r0 = d%uv_rhs; ab0 = d%uv_rhsAB
                    d%uv = u2; d%uv_rhsAB = hist
                    call compute_vel_rhs(d, mesh, dt, .false., partit)
                    call check(same_elem(d%uv_rhs, r0) .and. same_ab(d%uv_rhsAB, ab0), &
                               'R1a uv_in = u2 replaces uv everywhere, '//trim(tag), mype)
                    call check(.not. same_elem(u1, u2), 'R1a u1 /= u2 (non-vacuous)', mype)
                    ! b) no_history (nonzero history) == lfirst (zero history), also with uv_in
                    d%uv = u1; d%uv_rhsAB = hist
                    call compute_vel_rhs(d, mesh, dt, .false., partit, no_history=.true.)
                    r0 = d%uv_rhs; ab0 = d%uv_rhsAB
                    d%uv_rhsAB = 0.0_WP
                    call compute_vel_rhs(d, mesh, dt, .true., partit)
                    call check(same_elem(d%uv_rhs, r0) .and. same_ab(d%uv_rhsAB, ab0), &
                               'R1b no_history == lfirst with a zero history, '//trim(tag), mype)
                    d%uv_rhsAB = hist
                    call compute_vel_rhs(d, mesh, dt, .false., partit, uv_in=u2, no_history=.true.)
                    r0 = d%uv_rhs; ab0 = d%uv_rhsAB
                    d%uv = u2; d%uv_rhsAB = 0.0_WP
                    call compute_vel_rhs(d, mesh, dt, .true., partit)
                    call check(same_elem(d%uv_rhs, r0) .and. same_ab(d%uv_rhsAB, ab0), &
                               'R1b no_history + uv_in == lfirst on uv_in, '//trim(tag), mype)
                    d%uv_rhsAB = hist
                    call compute_vel_rhs(d, mesh, dt, .false., partit)
                    call check(.not. same_elem(d%uv_rhs, r0), 'R1b the AB history matters (non-vacuous), '// &
                               trim(tag), mype)
                    call free_dyn(d)
                end do
            end do
        end do
        rvo_upwind = rvo_save
    end subroutine r1_momentum_hooks

    subroutine free_dyn(d)
        type(t_dyn), intent(inout) :: d
        type(t_dyn) :: fresh
        d = fresh
    end subroutine free_dyn

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
