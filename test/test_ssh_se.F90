program test_ssh_se
    ! Split-explicit barotropic solver (docs/plans/2026-10-17-split-explicit.md), Task 0.3: the
    ! knobs, the CFL guard and se_init of src/oce/oce_ssh_se.F90 on the rotated pi mesh.
    !
    !   C1  the SM coefficients at chi = 0.1 equal the literal numbers worked out by hand from
    !       the notes (beta = 0.281105, zeta = -0.003689497, gamma = 0.0320866, delta =
    !       0.724707606; relative 1e-9), which catches a typo in a constant; at chi = 0 the
    !       constant terms (zeta = 0.00976186, gamma = 0.083445, delta = 0.60296872). The AB3
    !       and AM4 weights sum to 1.
    !   C2  knob parsing (se_parse_knob): every bad value is rejected (ok = .false., the t_dyn
    !       field unchanged, the message starts with the knob name); se_read_env and
    !       se_env_requested stop on exactly these. Good values land in t_dyn.
    !   C3  the CFL guard: the substep limit is finite, equals the MIN over the ranks of an
    !       independent per-rank recomputation, and limit and M_min are identical on every rank
    !       (MIN == MAX over the ranks); tau/M_min <= limit < tau/(M_min - 1); pi has M_min > 1,
    !       so the decision for M = 1 is a real test: SE_CFL_DIE, SE_CFL_FORCED with
    !       FESOM3_SE_M_FORCE, SE_CFL_OK for M_min, SE_CFL_DIE for M_min - 1.
    !   C4  se_init under the ctest environment FESOM3_SSH_SE=1 FESOM3_SE_M=40: the knobs from
    !       the environment (M = 40; chi, visc_ah, check, force at their defaults), the
    !       coefficients in t_dyn, the state at the extents of the plan's Technical Details §1,
    !       the cold start (eta ring == eta_n bitwise; the three U levels bitwise equal and
    !       == sum_k helem*uv on owned + eDim elements; s_old == 0) and se_ready. eta_n and uv
    !       are hashed from the global ids, so every copy of a node or element holds the same
    !       value.
    ! FESOM3_MESH_DIR (default pi), FESOM3_TEST_ROTATE=1 reads it rotated (setup_test_mesh).
    use mpi
    use mod_precision,    only: WP, MP, MPI_WP
    use mod_constants,    only: g
    use mod_mesh,         only: t_mesh, MAX_NV
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init
    use mod_dyn,          only: t_dyn
    use mod_part_bounds,  only: local_dims
    use oce_ssh_se,       only: se_env_requested, se_parse_knob, se_init, se_sm_coeffs, &
                                se_cfl_dtbt_limit, se_cfl_m_min, se_cfl_decision, &
                                SE_CFL_OK, SE_CFL_FORCED, SE_CFL_DIE
    use test_util,        only: setup_test_mesh, check, finish, sum_all, hval
    implicit none

    character(len=*), parameter :: PI_DIR = '/home/a/a270088/port2/fesom2/tests/data/MESHES/pi'
    real(kind=WP),    parameter :: dt = 86400.0_WP / real(48, WP)   ! the pi/CORE2 timestep
    type(t_partit) :: partit
    type(t_mesh)   :: mesh
    integer :: M_min = 0

    call par_init(partit)
    call setup_test_mesh(mesh, partit, PI_DIR, cartesian=.false., partition_np1=.true.)

    call c1_coefficients()
    call c2_parsing()
    call c3_cfl_guard()
    call c4_se_init()
    call finish(partit, 'test_ssh_se')

contains

    logical function near(a, b)
        ! |a - b| <= 1e-9 |b|: the literal numbers are exact decimals of the notes' constants
        real(kind=WP), intent(in) :: a, b
        near = abs(a - b) <= max(1.0e-9_WP*abs(b), 4.0_WP*epsilon(1.0_WP))
    end function near

    subroutine c1_coefficients()
        real(kind=WP) :: b, z, gm, d
        call se_sm_coeffs(0.1_WP, b, z, gm, d)
        call check(near(b,  0.281105_WP),     'C1 chi=0.1 beta  = 0.281105',     partit%mype)
        call check(near(z, -0.003689497_WP),  'C1 chi=0.1 zeta  = -0.003689497', partit%mype)
        call check(near(gm, 0.0320866_WP),    'C1 chi=0.1 gamma = 0.0320866',    partit%mype)
        call check(near(d,  0.724707606_WP),  'C1 chi=0.1 delta = 0.724707606',  partit%mype)
        call check(abs((1.5_WP + b) - (0.5_WP + 2.0_WP*b) + b - 1.0_WP) <= 4.0_WP*epsilon(1.0_WP), &
                   'C1 AB3 weights sum to 1', partit%mype)
        call check(abs(d + (1.0_WP - d - gm - z) + gm + z - 1.0_WP) <= 4.0_WP*epsilon(1.0_WP), &
                   'C1 AM4 weights sum to 1', partit%mype)
        call se_sm_coeffs(0.0_WP, b, z, gm, d)
        call check(near(z, 0.00976186_WP) .and. near(gm, 0.083445_WP) .and. near(d, 0.60296872_WP), &
                   'C1 chi=0: zeta = 0.00976186, gamma = 0.083445, delta = 0.60296872', partit%mype)
    end subroutine c1_coefficients

    subroutine c2_parsing()
        type(t_dyn) :: d
        ! rejected values, per knob
        call bad('FESOM3_SSH_SE', '2');      call bad('FESOM3_SSH_SE', 'yes')
        call bad('FESOM3_SSH_SE', '01');     call bad('FESOM3_SSH_SE', '-1')
        call bad('FESOM3_SSH_SE', ' ');      call bad('FESOM3_SSH_SE', '')
        call bad('FESOM3_SE_M', '0');        call bad('FESOM3_SE_M', '-3')
        call bad('FESOM3_SE_M', '2.5');      call bad('FESOM3_SE_M', '1e2')
        call bad('FESOM3_SE_M', 'abc');      call bad('FESOM3_SE_M', '40 50')
        call bad('FESOM3_SE_M', '1234567890'); call bad('FESOM3_SE_M', '')
        call bad('FESOM3_SE_M_FORCE', '2');  call bad('FESOM3_SE_M_FORCE', 'T')
        call bad('FESOM3_SE_CHI', '0.26');   call bad('FESOM3_SE_CHI', '-0.01')
        call bad('FESOM3_SE_CHI', '1');      call bad('FESOM3_SE_CHI', 'nan')
        call bad('FESOM3_SE_CHI', 'inf');    call bad('FESOM3_SE_CHI', '0.1x')
        call bad('FESOM3_SE_CHI', '0.1 0.2'); call bad('FESOM3_SE_CHI', '0.1,0.2')
        call bad('FESOM3_SE_CHI', '')
        call bad('FESOM3_SE_VISC_AH', '-1'); call bad('FESOM3_SE_VISC_AH', '-1e-9')
        call bad('FESOM3_SE_VISC_AH', 'NaN'); call bad('FESOM3_SE_VISC_AH', 'Infinity')
        call bad('FESOM3_SE_CHECK', '2');    call bad('FESOM3_SE_CHECK', 'on')
        call bad('FESOM3_SE_BOGUS', '1')
        ! accepted values land in their field
        call good('FESOM3_SSH_SE', '1', d);       call check(d%use_ssh_se, 'C2 FESOM3_SSH_SE=1', partit%mype)
        call good('FESOM3_SSH_SE', ' 0 ', d);     call check(.not. d%use_ssh_se, 'C2 FESOM3_SSH_SE=" 0 "', partit%mype)
        call good('FESOM3_SE_M', '1', d);         call check(d%se_M == 1, 'C2 FESOM3_SE_M=1', partit%mype)
        call good('FESOM3_SE_M', ' 75', d);       call check(d%se_M == 75, 'C2 FESOM3_SE_M=" 75"', partit%mype)
        call good('FESOM3_SE_M_FORCE', '1', d);   call check(d%se_m_force, 'C2 FESOM3_SE_M_FORCE=1', partit%mype)
        call good('FESOM3_SE_CHI', '0', d);       call check(d%se_chi == 0.0_WP, 'C2 FESOM3_SE_CHI=0', partit%mype)
        call good('FESOM3_SE_CHI', '0.25', d);    call check(d%se_chi == 0.25_WP, 'C2 FESOM3_SE_CHI=0.25', partit%mype)
        call good('FESOM3_SE_CHI', '5.0e-2', d);  call check(d%se_chi == 0.05_WP, 'C2 FESOM3_SE_CHI=5.0e-2', partit%mype)
        call good('FESOM3_SE_VISC_AH', '0', d);   call check(d%se_visc_ah == 0.0_WP, 'C2 FESOM3_SE_VISC_AH=0', partit%mype)
        call good('FESOM3_SE_VISC_AH', '0.02', d); call check(d%se_visc_ah == 0.02_WP, 'C2 FESOM3_SE_VISC_AH=0.02', partit%mype)
        call good('FESOM3_SE_CHECK', '1', d);     call check(d%se_check, 'C2 FESOM3_SE_CHECK=1', partit%mype)
    end subroutine c2_parsing

    subroutine bad(name, str)
        character(len=*), intent(in) :: name, str
        type(t_dyn) :: d, d0
        logical :: ok
        character(len=256) :: msg
        call se_parse_knob(name, str, d, ok, msg)
        call check(.not. ok, 'C2 '//name//'="'//str//'" rejected', partit%mype)
        call check(index(msg, name) == 1, 'C2 '//name//'="'//str//'" message: '//trim(msg), partit%mype)
        call check((d%use_ssh_se .eqv. d0%use_ssh_se) .and. d%se_M == d0%se_M .and. &
                   d%se_chi == d0%se_chi .and. d%se_visc_ah == d0%se_visc_ah .and. &
                   (d%se_check .eqv. d0%se_check) .and. (d%se_m_force .eqv. d0%se_m_force), &
                   'C2 '//name//'="'//str//'" leaves t_dyn unchanged', partit%mype)
    end subroutine bad

    subroutine good(name, str, d)
        character(len=*), intent(in)    :: name, str
        type(t_dyn),      intent(inout) :: d
        logical :: ok
        character(len=256) :: msg
        call se_parse_knob(name, str, d, ok, msg)
        call check(ok .and. len_trim(msg) == 0, 'C2 '//name//'="'//str//'" accepted', partit%mype)
    end subroutine good

    subroutine c3_cfl_guard()
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, ierr, mm(2)
        real(kind=WP) :: lim, mine, H, ll(2)
        lim   = se_cfl_dtbt_limit(mesh, partit)
        M_min = se_cfl_m_min(dt, lim)
        call check(lim > 0.0_WP .and. lim < huge(1.0_WP), 'C3 the substep limit is finite', partit%mype)
        ! independent recomputation on this rank's owned surface nodes, then MIN over the ranks
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        mine = huge(1.0_WP)
        do n = 1, nNodO
            if (mesh%nlevels_nod2D(n) < 2) cycle
            H = -mesh%zbar(mesh%nlevels_nod2D(n))            ! pi: no cavity, zbar(1) = 0
            mine = min(mine, real(0.5_WP*mesh%mesh_resolution(n)/sqrt(g*H), WP))
        end do
        call MPI_Allreduce(MPI_IN_PLACE, mine, 1, MPI_WP, MPI_MIN, partit%MPI_COMM_FESOM, ierr)
        call check(mine == lim, 'C3 limit == MIN over the ranks of 0.5*res/sqrt(g*H)', partit%mype)
        ! the same on every rank
        ll = [lim, -lim]
        call MPI_Allreduce(MPI_IN_PLACE, ll, 2, MPI_WP, MPI_MAX, partit%MPI_COMM_FESOM, ierr)
        call check(ll(1) == lim .and. -ll(2) == lim, 'C3 limit identical on every rank', partit%mype)
        mm = [M_min, -M_min]
        call MPI_Allreduce(MPI_IN_PLACE, mm, 2, MPI_INTEGER, MPI_MAX, partit%MPI_COMM_FESOM, ierr)
        call check(mm(1) == M_min .and. -mm(2) == M_min, 'C3 M_min identical on every rank', partit%mype)
        ! M_min is the fewest admissible substeps
        call check(dt/real(M_min, WP) <= lim, 'C3 tau/M_min <= limit', partit%mype)
        call check(M_min > 1, 'C3 pi M_min > 1 (M = 1 is below the guard)', partit%mype)
        if (M_min > 1) call check(dt/real(M_min - 1, WP) > lim, 'C3 tau/(M_min-1) > limit', partit%mype)
        ! the decision
        call check(se_cfl_decision(1, M_min, .false.) == SE_CFL_DIE,    'C3 M = 1: the guard fires', partit%mype)
        call check(se_cfl_decision(1, M_min, .true.)  == SE_CFL_FORCED, 'C3 M = 1 + FORCE: warn and run', partit%mype)
        call check(se_cfl_decision(M_min - 1, M_min, .false.) == SE_CFL_DIE, 'C3 M = M_min-1 fires', partit%mype)
        call check(se_cfl_decision(M_min, M_min, .false.) == SE_CFL_OK,  'C3 M = M_min passes', partit%mype)
        call check(se_cfl_decision(50, M_min, .false.) == SE_CFL_OK, &
                   'C3 M = 50 (the default) passes on pi', partit%mype)
        call check(se_cfl_m_min(dt, huge(1.0_WP)) == 1, 'C3 no limit: M_min = 1', partit%mype)
        if (partit%mype == 0) write(*,'(a,f0.1,a,f0.3,a,i0)') 'test_ssh_se: pi, tau = ', dt, &
            ' s: substep limit = ', lim, ' s, M_min = ', M_min
    end subroutine c3_cfl_guard

    subroutine c4_se_init()
        type(t_dyn) :: dyn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, nl, c, nbad
        real(kind=WP) :: b, z, gm, d, ub(2), err
        nl = mesh%nl
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call check(se_env_requested(partit), 'C4 FESOM3_SSH_SE=1 in the test environment', partit%mype)

        ! a cold state as the drivers build it: full-cell helem, uv and eta_n from global ids
        allocate(mesh%helem(nl-1, nElemF)); mesh%helem = 0.0_MP
        allocate(dyn%uv(2, nl-1, nElemF), dyn%eta_n(nNodL))
        dyn%uv = 0.0_WP
        do e = 1, nElemF
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                mesh%helem(nz, e) = mesh%zbar(nz) - mesh%zbar(nz+1)
                do c = 1, 2
                    dyn%uv(c, nz, e) = 0.2_WP*hval(partit%myList_elem2D(e), 10*nz + c)
                end do
            end do
        end do
        do n = 1, nNodL
            dyn%eta_n(n) = 0.5_WP*hval(partit%myList_nod2D(n), 0)
        end do
        dyn%AB_order = 2

        call se_init(dyn, mesh, partit, dt)

        call check(dyn%use_ssh_se .and. dyn%se_ready, 'C4 use_ssh_se and se_ready', partit%mype)
        call check(dyn%se_M == 40, 'C4 FESOM3_SE_M=40 from the environment', partit%mype)
        call check(dyn%se_chi == 0.1_WP .and. dyn%se_visc_ah == 0.01_WP .and. .not. dyn%se_check &
                   .and. .not. dyn%se_m_force, 'C4 unset knobs keep the defaults', partit%mype)
        call se_sm_coeffs(0.1_WP, b, z, gm, d)
        call check(dyn%se_beta == b .and. dyn%se_zeta == z .and. dyn%se_gamma == gm .and. &
                   dyn%se_delta == d, 'C4 the coefficients in t_dyn', partit%mype)
        ! extents (Technical Details §1)
        call check(allocated(dyn%se_eta) .and. allocated(dyn%se_ubt) .and. allocated(dyn%se_sbar_old) &
                   .and. allocated(dyn%se_ubtn) .and. allocated(dyn%se_etam) .and. allocated(dyn%se_etanew) &
                   .and. allocated(dyn%se_umean) .and. allocated(dyn%se_force) .and. allocated(dyn%se_H) &
                   .and. allocated(dyn%se_c) .and. allocated(dyn%se_kappa), 'C4 the state is allocated', &
                   partit%mype)
        if (.not. (allocated(dyn%se_eta) .and. allocated(dyn%se_ubt))) return
        call check(all(shape(dyn%se_eta) == [nNodL, 2]) .and. size(dyn%se_etam) == nNodL .and. &
                   size(dyn%se_etanew) == nNodL, 'C4 node extents nNodL', partit%mype)
        call check(all(shape(dyn%se_ubt) == [2, nElemL, 3]) .and. lbound(dyn%se_ubt, 3) == 0 .and. &
                   all(shape(dyn%se_sbar_old) == [2, nElemL]) .and. all(shape(dyn%se_ubtn) == [2, nElemL]) &
                   .and. all(shape(dyn%se_umean) == [2, nElemL]) .and. all(shape(dyn%se_force) == [2, nElemL]) &
                   .and. size(dyn%se_H) == nElemL, 'C4 element extents nElemL = owned + eDim', partit%mype)
        call check(all(shape(dyn%se_c) == [MAX_NV, nElemO]) .and. all(shape(dyn%se_kappa) == [MAX_NV, nElemO]), &
                   'C4 se_c / se_kappa extents (MAX_NV, nElemO)', partit%mype)
        ! cold start
        nbad = count(dyn%se_eta(:, 1) /= dyn%eta_n(1:nNodL)) + count(dyn%se_eta(:, 2) /= dyn%eta_n(1:nNodL))
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C4 eta ring == eta_n (bitwise, nNodL)', partit%mype)
        nbad = count(dyn%se_ubt(:, :, 1) /= dyn%se_ubt(:, :, 0)) + count(dyn%se_ubt(:, :, 2) /= dyn%se_ubt(:, :, 0))
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C4 the three U ring levels are bitwise equal', partit%mype)
        err = 0.0_WP
        do e = 1, nElemL
            ub = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                ub = ub + mesh%helem(nz, e)*dyn%uv(:, nz, e)
            end do
            err = max(err, maxval(abs(dyn%se_ubt(:, e, 0) - ub))/max(maxval(abs(ub)), 1.0_WP))
        end do
        call check(err <= 1.0e-13_WP, 'C4 U ring == sum_k helem*uv (owned + eDim)', partit%mype)
        call check(all(dyn%se_sbar_old == 0.0_WP), 'C4 s_old == 0', partit%mype)
        call check(maxval(abs(dyn%se_ubt(:, :, 0))) > 0.0_WP, 'C4 the U ring is non-trivial', partit%mype)
    end subroutine c4_se_init

end program test_ssh_se
