program test_ssh_se
    ! Split-explicit barotropic solver (docs/plans/2026-10-17-split-explicit.md), Tasks 0.3-1.4: the
    ! knobs, the CFL guard, se_init, the barotropic forcing, the substeps, the trim and the
    ! invariant suite of src/oce/oce_ssh_se.F90 on the rotated pi mesh.
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
    !   C5  se_elem_depth and the PRESSURE part of the forcing (Task 1.1): H^n == sum_k helem on
    !       owned + eDim elements (bitwise, positive); with uv = 0, pgf = 0, momentum advection off
    !       and a non-trivial eta_n, the real compute_vel_rhs puts only -g grad(eta) into UV_rhs,
    !       and se_bt_forcing's g H^n grad(eta^n) must remove it: max|F| <= 1e-12 * max|H^n Fx|.
    !   C6  the CORIOLIS part of the forcing and the lfirst weight (Task 1.1): momentum advection
    !       off, eta_n = 0, pgf = 0, helem constant. Step 1 (lfirst, cold uv_rhsAB, s_old = 0):
    !       the real compute_vel_rhs on uv^{n-3/2}, then F = 0 to rounding; the same UV_rhs taken
    !       as a non-lfirst step leaves exactly (0.5+eps) f e_z x s^n (the lfirst weight a2 = 1
    !       matters). s_old == sum_k helem*uv^{n-3/2} afterwards (bitwise). Step 2 (not lfirst):
    !       uv^{n-1/2}, compute_vel_rhs blends ab2*C(uv^{n-1/2}) + ab1*C(uv^{n-3/2}), and F = 0 to
    !       rounding again. se_ubtn == ring level 0 (bitwise, owned + eDim).
    !       Tolerances are relative to the largest Coriolis term max|f s|; the measured residuals
    !       are printed.
    !   C7  operator / r-term consistency (Task 1.2): the real compute_vel_rhs twice on the SAME uv
    !       (so s^n = s^{n-1}) with a non-trivial eta_n and pgf, se_bt_forcing after each; then
    !       se_tendency with U* = s^n, eta^A = eta^n, H^A = H^n, dU = 0 equals G/tau = sum_k
    !       helem*UV_rhs/tau to 1e-12 of the largest term. A sign or factor mismatch between F
    !       and the operator (Coriolis, pressure) fails this.
    !   C8  the viscosity (Task 1.2): kappa >= 0, kappa = 0 exactly on boundary slots and on a
    !       triangle's slot 4, > 0 on every interior slot; kappa = A_h*l/(d*A_e) against an
    !       independent recomputation from edge_len (owned edges) and the elem_center distance;
    !       sum_e A_e*V_e = 0 to 1e-12 of sum_e A_e*|V_e| for a hashed halo-valid dU (each
    !       element counted once: on the rank owning its first vertex); V of a uniform velocity is
    !       exactly 0 over a flat bottom and 0 to rounding over the real one.
    !   C9  se_subcycle (Task 1.2):
    !       a) M = 1 on a hashed state: the rings shift by copying (eta^{-2/M} <- eta^{-1/M} <-
    !          eta_n, U levels 2 <- 1 <- 0; bitwise), and <<U>> = the AB3 transport U*, eta^1 =
    !          eta^0 + dtbt*I(U*)/areasvol, U^1 = U^0 + dtbt*se_tendency against the operators
    !          called directly (to 1e-13: GNU may contract a + b*c into an FMA differently here);
    !       b) rest stays at rest: eta = 0, U = 0, F = 0 -> every ring level and <<U>> exactly 0
    !          after M = 40 substeps;
    !       c) a hashed state (eta, the three U levels, F), M = 40, viscosity on: the volume
    !          sum_v areasvol*(eta^M - eta^0) = 0 (closed basin, linfs) to 1e-12 of
    !          sum_v areasvol*|eta^M - eta^0|; the eta-compatibility eta^M - eta^0 =
    !          tau*I(<<U>>)/areasvol on owned nodes to 1e-12 of max(|eta^0|, |eta^M - eta^0|); at
    !          npes > 1 the eta levels and U^M are halo-valid (stale-halo probes = 0);
    !       d) dtbt convergence: a gravity wave from a Gaussian bump (1 m, e-folding 1500 km, at
    !          the node with global id 1000; U = 0, F = 0), chi = 0, viscosity on, one baroclinic
    !          step with M = 16, 32, 64
    !          against an M = 128 reference: the observed orders log2(e(M)/e(2M)) of max|eta|
    !          and max|U| are >= 2 (a swapped ring level or a misplaced weight drops it to 1).
    !       Residuals and orders are printed.
    !   C10 se_trim_finish (Task 1.3) on a hashed predictor state (uv, UV_rhs) with a hashed <<U>>
    !       and eta^M: sum_k helem^n uv = <<U>> to 1e-13 of sum_k helem|u**|; the correction is
    !       depth-uniform, uv_k - u**_k = -(sum_j helem u**_j - <<U>>)/H^n for every layer (to 1e-13
    !       of max|u**|); the layers below nlevels untouched; hbar_old = old hbar, hbar = eta_n =
    !       eta^M on owned + halo nodes and ssh_rhs_old = I(<<U>>) on owned nodes (bitwise); the
    !       U ring untouched; SE_CHECK off computes nothing; at npes > 1 uv is synced over the
    !       shared elements and halo-valid.
    !   C11 FESOM3_SE_CHECK (Task 1.4) after a full SE step (M = 40 substeps + se_trim_finish, which
    !       prints the SE_CHECK line): eta-compatibility, trim and mean-sea-level change at rounding
    !       (1e-12 of their scales), copy spreads 0, max|eta| and max|U| equal to the direct maxima,
    !       no non-finite value; the line is blank-separated key=value tokens; a uv perturbed by
    !       delta on one element shows as trim = helem*delta (the other checks unchanged), and a
    !       NaN there is counted.
    ! FESOM3_MESH_DIR (default pi), FESOM3_TEST_ROTATE=1 reads it rotated (setup_test_mesh).
    use mpi
    use mod_precision,    only: WP, MP, MPI_WP
    use mod_constants,    only: g
    use mod_mesh,         only: t_mesh, MAX_NV
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init
    use mod_dyn,          only: t_dyn
    use mod_part_bounds,  only: local_dims
    use mod_config,       only: ab_epsilon
    use mod_constants,    only: r_earth
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use, intrinsic :: ieee_exceptions, only: ieee_invalid, ieee_support_halting, ieee_get_halting_mode, &
                                             ieee_set_halting_mode, ieee_set_flag
    use mod_halo,         only: stale_halo_max_nod, stale_halo_max_elem, shared_copy_max_elem
    use mod_mesh_areas,   only: elem_center
    use mod_mesh_rotate,  only: trim_cyclic
    use oce_dyn_velrhs,   only: compute_vel_rhs
    use oce_ssh_se,       only: se_env_requested, se_parse_knob, se_init, se_sm_coeffs, &
                                se_cfl_dtbt_limit, se_cfl_m_min, se_cfl_decision, &
                                SE_CFL_OK, SE_CFL_FORCED, SE_CFL_DIE, se_bt_forcing, &
                                se_elem_depth, se_subcycle, se_inflow, se_visc, se_tendency, &
                                se_trim_finish, se_check_values, se_check_line, SE_NCHK, &
                                SE_CHK_ETACOMP, SE_CHK_TRIM, SE_CHK_MSL, SE_CHK_SPREADU, &
                                SE_CHK_SPREADUM, SE_CHK_MAXETA, SE_CHK_MAXU, SE_CHK_NBAD
    use test_util,        only: setup_test_mesh, check, finish, sum_all, max_all, hval
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
    call c5_forcing_pressure()
    call c6_forcing_coriolis()
    call c7_consistency()
    call c8_visc()
    call c9_subcycle()
    call c10_trim_finish()
    call c11_check()
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

    subroutine new_dyn(dyn)
        ! a zero dynamics state at the extents the drivers allocate (fesom_conserve), AB2,
        ! momentum advection off (momadv_opt = 0: neither branch of compute_vel_rhs runs).
        ! se_init comes after the caller has set uv and eta_n (the cold start reads them).
        type(t_dyn), intent(inout) :: dyn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, nl
        nl = mesh%nl
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        if (.not. allocated(mesh%helem)) error stop 'test_ssh_se: helem is set up by C4'
        allocate(dyn%uv(2, nl-1, nElemF), dyn%uv_rhs(2, nl-1, nElemF))
        allocate(dyn%uv_rhsAB(1, 2, nl-1, nElemF), dyn%eta_n(nNodL), dyn%ssh_rhs_old(nNodL))
        allocate(dyn%work%pgf_x(nl-1, nElemF), dyn%work%pgf_y(nl-1, nElemF))
        dyn%uv = 0.0_WP; dyn%uv_rhs = 0.0_WP; dyn%uv_rhsAB = 0.0_WP; dyn%eta_n = 0.0_WP
        dyn%ssh_rhs_old = 0.0_WP
        dyn%work%pgf_x = 0.0_WP; dyn%work%pgf_y = 0.0_WP
        dyn%AB_order   = 2
        dyn%momadv_opt = 0
    end subroutine new_dyn

    subroutine set_uv(dyn, seed)
        ! uv on every held element and wet layer, hashed from the global element id
        type(t_dyn), intent(inout) :: dyn
        integer,     intent(in)    :: seed
        integer :: e, nz, c
        do e = 1, size(dyn%uv, 3)
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                do c = 1, 2
                    dyn%uv(c, nz, e) = 0.2_WP*hval(partit%myList_elem2D(e), 10*nz + c + seed)
                end do
            end do
        end do
    end subroutine set_uv

    subroutine c5_forcing_pressure()
        type(t_dyn) :: dyn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, nbad
        integer :: elnodes(MAX_NV)
        real(kind=WP) :: H, pre(MAX_NV), Fx, Fy, res, scale
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(dyn)
        do n = 1, nNodL
            dyn%eta_n(n) = 0.5_WP*hval(partit%myList_nod2D(n), 3)
        end do
        call se_init(dyn, mesh, partit, dt)
        call compute_vel_rhs(dyn, mesh, dt, .false., partit)
        call se_bt_forcing(dyn, mesh, partit, dt, .false.)

        ! H^n on owned + eDim
        nbad = 0
        do e = 1, nElemL
            H = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                H = H + mesh%helem(nz, e)
            end do
            if (dyn%se_H(e) /= H .or. .not. (H > 0.0_WP)) nbad = nbad + 1
        end do
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C5 se_H == sum_k helem > 0 (bitwise, owned + eDim)', partit%mype)

        ! the pressure part cancels
        res = 0.0_WP; scale = 0.0_WP
        do e = 1, nElemO
            elnodes = mesh%elem2D_nodes(:, e)
            pre   = -(g*dyn%eta_n(elnodes))
            Fx    = sum(mesh%gradient_sca(1:4, e)*pre)
            Fy    = sum(mesh%gradient_sca(5:8, e)*pre)
            scale = max(scale, dyn%se_H(e)*max(abs(Fx), abs(Fy)))
            res   = max(res, maxval(abs(dyn%se_force(:, e))))
        end do
        call max_all(res, partit); call max_all(scale, partit)
        call check(scale > 0.0_WP, 'C5 the pressure term is non-trivial', partit%mype)
        call check(res <= 1.0e-12_WP*scale, 'C5 F = 0 with only -g grad(eta) in UV_rhs', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3)') &
            'test_ssh_se: C5 pressure part: max|F| = ', res, ' m2/s2, max|g H grad eta| = ', scale, &
            ', ratio ', res/scale
    end subroutine c5_forcing_pressure

    subroutine c6_forcing_coriolis()
        type(t_dyn) :: dyn, dwrong
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e, nz, nbad
        real(kind=WP) :: sn(2), f, res1, resw, fw, res2, scale1, scale2, aw
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(dyn)
        call set_uv(dyn, 0)                              ! uv^{n-3/2}
        call se_init(dyn, mesh, partit, dt)              ! s_old = 0
        call check(all(dyn%se_sbar_old == 0.0_WP), 'C6 s_old == 0 after se_init', partit%mype)

        ! ---- step 1, lfirst ----
        call compute_vel_rhs(dyn, mesh, dt, .true., partit)
        dwrong = dyn
        call se_bt_forcing(dwrong, mesh, partit, dt, .false.)   ! a2 = 1.5+eps on an lfirst UV_rhs
        call se_bt_forcing(dyn,    mesh, partit, dt, .true.)
        aw = 0.5_WP + ab_epsilon                         ! (1.5+eps) - 1
        res1 = 0.0_WP; resw = 0.0_WP; fw = 0.0_WP; scale1 = 0.0_WP; nbad = 0
        do e = 1, nElemO
            sn = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                sn(1) = sn(1) + mesh%helem(nz, e)*dyn%uv(1, nz, e)
                sn(2) = sn(2) + mesh%helem(nz, e)*dyn%uv(2, nz, e)
            end do
            f = mesh%coriolis(e)
            scale1 = max(scale1, abs(f)*maxval(abs(sn)))
            res1 = max(res1, maxval(abs(dyn%se_force(:, e))))
            ! the wrong weight leaves (a2 - 1) f e_z x s^n = aw*f*(-sn(2), sn(1))
            resw = max(resw, abs(dwrong%se_force(1, e) + aw*f*sn(2)), &
                             abs(dwrong%se_force(2, e) - aw*f*sn(1)))
            fw   = max(fw, maxval(abs(dwrong%se_force(:, e))))
            if (any(dyn%se_sbar_old(:, e) /= sn)) nbad = nbad + 1
        end do
        call max_all(res1, partit); call max_all(resw, partit); call max_all(fw, partit)
        call max_all(scale1, partit); call sum_all(nbad, partit)
        call check(scale1 > 0.0_WP, 'C6 the Coriolis term is non-trivial', partit%mype)
        call check(res1 <= 1.0e-12_WP*scale1, 'C6 lfirst: F = 0 (a2 = 1)', partit%mype)
        call check(resw <= 1.0e-12_WP*scale1 .and. fw >= 0.5_WP*aw*scale1, &
                   'C6 lfirst UV_rhs with a2 = 1.5+eps: F = (0.5+eps) f e_z x s^n', partit%mype)
        call check(nbad == 0, 'C6 s_old == sum_k helem*uv after the step (bitwise)', partit%mype)

        ! ---- step 2 ----
        call set_uv(dyn, 5)                              ! uv^{n-1/2}
        call compute_vel_rhs(dyn, mesh, dt, .false., partit)
        call se_bt_forcing(dyn, mesh, partit, dt, .false.)
        res2 = 0.0_WP; scale2 = 0.0_WP
        do e = 1, nElemO
            sn = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                sn(1) = sn(1) + mesh%helem(nz, e)*dyn%uv(1, nz, e)
                sn(2) = sn(2) + mesh%helem(nz, e)*dyn%uv(2, nz, e)
            end do
            scale2 = max(scale2, abs(mesh%coriolis(e))*maxval(abs(sn)))
            res2 = max(res2, maxval(abs(dyn%se_force(:, e))))
        end do
        call max_all(res2, partit); call max_all(scale2, partit)
        call check(res2 <= 1.0e-12_WP*max(scale1, scale2), 'C6 step 2 (AB2 blend): F = 0', partit%mype)
        nbad = count(dyn%se_ubtn(:, 1:nElemL) /= dyn%se_ubt(:, 1:nElemL, 0))
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C6 se_ubtn == U ring level 0 (bitwise, owned + eDim)', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3,a,es10.3)') &
            'test_ssh_se: C6 Coriolis part: lfirst max|F| = ', res1, ', AB2 step max|F| = ', res2, &
            ' m2/s2, max|f s| = ', max(scale1, scale2), '; wrong-weight residual ', resw
    end subroutine c6_forcing_coriolis

    subroutine c7_consistency()
        type(t_dyn) :: dyn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz
        integer :: elnodes(MAX_NV)
        real(kind=WP), allocatable :: ustar(:,:), dU(:,:), tend(:,:), HA(:)
        real(kind=WP) :: gb(2), res, scale, pre(MAX_NV), Fx, Fy     ! gb: G (g is gravity)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(dyn)
        call set_uv(dyn, 0)
        do n = 1, nNodL
            dyn%eta_n(n) = 0.5_WP*hval(partit%myList_nod2D(n), 3)
        end do
        do e = 1, nElemF
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                dyn%work%pgf_x(nz, e) = 1.0e-5_WP*hval(partit%myList_elem2D(e), 10*nz + 7)
                dyn%work%pgf_y(nz, e) = 1.0e-5_WP*hval(partit%myList_elem2D(e), 10*nz + 8)
            end do
        end do
        call se_init(dyn, mesh, partit, dt)
        call compute_vel_rhs(dyn, mesh, dt, .true., partit)       ! s_old <- s(uv)
        call se_bt_forcing(dyn, mesh, partit, dt, .true.)
        call compute_vel_rhs(dyn, mesh, dt, .false., partit)      ! the same uv: s^n = s^{n-1}
        call se_bt_forcing(dyn, mesh, partit, dt, .false.)

        allocate(ustar(2, nElemL), dU(2, nElemL), tend(2, nElemL), HA(nElemL))
        ustar = dyn%se_sbar_old                                   ! = s^n
        dU    = 0.0_WP
        HA    = dyn%se_H
        call se_tendency(dyn, mesh, partit, ustar, dyn%eta_n, HA, dU, tend)
        res = 0.0_WP; scale = 0.0_WP
        do e = 1, nElemO
            gb = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                gb = gb + mesh%helem(nz, e)*dyn%uv_rhs(:, nz, e)
            end do
            elnodes = mesh%elem2D_nodes(:, e)
            pre = -(g*dyn%eta_n(elnodes))
            Fx  = sum(mesh%gradient_sca(1:4, e)*pre)
            Fy  = sum(mesh%gradient_sca(5:8, e)*pre)
            scale = max(scale, maxval(abs(gb))/dt, abs(mesh%coriolis(e))*maxval(abs(ustar(:, e))), &
                        dyn%se_H(e)*max(abs(Fx), abs(Fy)))
            res = max(res, maxval(abs(tend(:, e) - gb/dt)))
        end do
        call max_all(res, partit); call max_all(scale, partit)
        call check(res <= 1.0e-12_WP*scale, 'C7 se_tendency(U* = s^n = s_old, eta^A = eta^n, H^A = H^n, '// &
                   'dU = 0) == G/tau', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3)') &
            'test_ssh_se: C7 consistency: max|tend - G/tau| = ', res, ' m2/s2, largest term ', scale, &
            ', ratio ', res/scale
    end subroutine c7_consistency

    subroutine c8_visc()
        type(t_dyn) :: dyn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e, k, nb, ed, c, nbad
        integer :: nbad_zero, nbad_pos
        real(kind=WP), allocatable :: dU(:,:), V(:,:)
        real(kind=WP) :: cxe, cye, cxn, cyn, b1, b2, cosm, d, expect, err, ssum(2), sabs(2)
        real(kind=WP) :: vmax, vscale, u0(2)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(dyn)
        call se_init(dyn, mesh, partit, dt)
        call se_elem_depth(dyn, mesh, partit)
        call check(dyn%se_visc_ah > 0.0_WP, 'C8 the viscosity is on by default', partit%mype)

        ! ---- kappa: structure and value ----
        nbad_zero = 0; nbad_pos = 0; nbad = 0; err = 0.0_WP
        do e = 1, nElemO
            do k = 1, MAX_NV
                nb = mesh%elem_neighbors(k, e)
                if (k > mesh%elem2D_nnodes(e) .or. nb == e) then
                    if (dyn%se_kappa(k, e) /= 0.0_WP) nbad_zero = nbad_zero + 1
                    cycle
                end if
                if (.not. (dyn%se_kappa(k, e) > 0.0_WP)) nbad_pos = nbad_pos + 1
                ed = mesh%elem_edges(k, e)
                if (nb > nElemO .or. ed > nEdgeO) cycle
                call elem_center(mesh, e,  cxe, cye)
                call elem_center(mesh, nb, cxn, cyn)
                cosm = 0.5_WP*(mesh%elem_cos(e) + mesh%elem_cos(nb))
                b1 = cxn - cxe; call trim_cyclic(b1)
                b2 = cyn - cye
                d  = sqrt((b1*cosm*r_earth)**2 + (b2*r_earth)**2)
                expect = dyn%se_visc_ah*mesh%edge_len(ed)**2/(d*mesh%elem_area(e))
                err = max(err, abs(dyn%se_kappa(k, e) - expect)/expect)
                nbad = nbad + 1
            end do
        end do
        call sum_all(nbad_zero, partit); call sum_all(nbad_pos, partit); call max_all(err, partit)
        call sum_all(nbad, partit)
        call check(nbad_zero == 0, 'C8 kappa == 0 on boundary slots and triangle slot 4', partit%mype)
        call check(nbad_pos == 0, 'C8 kappa > 0 on every interior slot', partit%mype)
        call check(nbad > 0 .and. err <= 1.0e-12_WP, &
                   'C8 kappa = visc_ah*edge_len^2/(d*A_e) (owned edges, owned neighbours)', partit%mype)

        ! ---- sum_e A_e V_e = 0 for a hashed dU ----
        allocate(dU(2, nElemL), V(2, nElemL))
        do e = 1, nElemL
            do c = 1, 2
                dU(c, e) = dyn%se_H(e)*0.5_WP*hval(partit%myList_elem2D(e), 40 + c)
            end do
        end do
        V = 0.0_WP
        call se_visc(dyn, mesh, partit, dU, V)
        ssum = 0.0_WP; sabs = 0.0_WP
        do e = 1, nElemO
            if (mesh%elem2D_nodes(1, e) > nNodO) cycle        ! counted on its first vertex's owner
            ssum = ssum + mesh%elem_area(e)*V(:, e)
            sabs = sabs + mesh%elem_area(e)*abs(V(:, e))
        end do
        call sum_all(ssum, partit); call sum_all(sabs, partit)
        call check(all(sabs > 0.0_WP), 'C8 V of a hashed dU is non-trivial', partit%mype)
        call check(all(abs(ssum) <= 1.0e-12_WP*sabs), 'C8 sum_e A_e*V_e = 0', partit%mype)
        if (partit%mype == 0) write(*,'(a,2es10.3,a,2es10.3)') &
            'test_ssh_se: C8 viscosity: sum_e A_e V_e = ', ssum, ' of sum_e A_e |V_e| = ', sabs
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3)') &
            'test_ssh_se: C8 kappa vs edge_len/centroid recomputation: max rel err ', err, &
            ', max kappa ', maxval(dyn%se_kappa)

        ! ---- uniform velocity ----
        u0 = [0.3_WP, -0.2_WP]
        do e = 1, nElemL
            dU(:, e) = dyn%se_H(e)*u0
        end do
        call se_visc(dyn, mesh, partit, dU, V)
        vmax = maxval(abs(V(:, 1:nElemO))); vscale = maxval(dyn%se_kappa)*maxval(dyn%se_H)*maxval(abs(u0))
        call max_all(vmax, partit); call max_all(vscale, partit)
        call check(vmax <= 1.0e-12_WP*vscale, 'C8 V(uniform velocity, real bottom) = 0 to rounding', partit%mype)
        dyn%se_H = 4000.0_WP
        do e = 1, nElemL
            dU(:, e) = dyn%se_H(e)*u0
        end do
        call se_visc(dyn, mesh, partit, dU, V)
        nbad = count(V(:, 1:nElemO) /= 0.0_WP)
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C8 V(uniform velocity, flat bottom) == 0 exactly', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3)') &
            'test_ssh_se: C8 uniform velocity, real bottom: max|V| = ', vmax, ' of kappa*H*u ', vscale
    end subroutine c8_visc

    subroutine c9_subcycle()
        type(t_dyn) :: d0, d, dref
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, c, nbad, i, Mk(4)
        real(kind=WP), allocatable :: ustar(:,:), tend(:,:), dU(:,:), HA(:), infl(:), eta1(:)
        real(kind=WP), allocatable :: etaM(:,:), uM(:,:,:), tmp(:)
        real(kind=WP) :: w0, w1, w2, x, y, z, ev(3), eu(3), pe(2), pu(2), vsum, vabs
        real(kind=WP) :: res, scale, stale, p0(3)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        allocate(ustar(2, nElemL), tend(2, nElemL), dU(2, nElemL), HA(nElemL), infl(nNodL), eta1(nNodL))

        ! ---- a hashed state: eta^n, the eta ring, the three U levels, F; then copy U^n ----
        call new_dyn(d0)
        do n = 1, nNodL
            d0%eta_n(n) = 0.1_WP*hval(partit%myList_nod2D(n), 11)
        end do
        call se_init(d0, mesh, partit, dt)
        call se_elem_depth(d0, mesh, partit)
        do n = 1, nNodL
            d0%se_eta(n, 1) = 0.1_WP*hval(partit%myList_nod2D(n), 12)
            d0%se_eta(n, 2) = 0.1_WP*hval(partit%myList_nod2D(n), 13)
        end do
        do e = 1, nElemL
            do c = 1, 2
                d0%se_ubt(c, e, 0) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 20 + c)
                d0%se_ubt(c, e, 1) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 22 + c)
                d0%se_ubt(c, e, 2) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 24 + c)
                d0%se_force(c, e)  = 1.0e-3_WP*hval(partit%myList_elem2D(e), 26 + c)
            end do
            d0%se_ubtn(:, e) = d0%se_ubt(:, e, 0)
        end do

        ! ---- a) M = 1: the ring shifts, <<U>>, and one substep against the operators ----
        d = d0
        d%se_M = 1
        call se_subcycle(d, mesh, partit, dt)
        w0 = 1.5_WP + d0%se_beta; w1 = -(0.5_WP + 2.0_WP*d0%se_beta); w2 = d0%se_beta
        do e = 1, nElemO
            ustar(:, e) = w0*d0%se_ubt(:, e, 0) + w1*d0%se_ubt(:, e, 1) + w2*d0%se_ubt(:, e, 2)
        end do
        call se_inflow(mesh, partit, ustar, infl)
        eta1 = 0.0_WP
        do n = 1, nNodO
            eta1(n) = d0%eta_n(n) + infl(n)*dt/mesh%areasvol(n)
        end do
        nbad = count(d%se_eta(:, 2) /= d0%se_eta(:, 1)) + count(d%se_eta(:, 1) /= d0%eta_n(1:nNodL)) &
             + count(d%se_etanew /= d%se_etam)
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C9a eta ring shifted by copy (bitwise)', partit%mype)
        res = maxval(abs(d%se_etam(1:nNodO) - eta1(1:nNodO)))
        scale = max(maxval(abs(eta1(1:nNodO))), maxval(abs(d0%eta_n(1:nNodO))))
        call max_all(res, partit); call max_all(scale, partit)
        call check(res <= 1.0e-13_WP*scale, 'C9a eta^1 = eta^0 + dtbt*I(U*)/areasvol', partit%mype)
        nbad = count(d%se_ubt(:, 1:nElemL, 2) /= d0%se_ubt(:, 1:nElemL, 1)) &
             + count(d%se_ubt(:, 1:nElemL, 1) /= d0%se_ubt(:, 1:nElemL, 0))
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C9a U ring shifted by copy (bitwise)', partit%mype)
        res = maxval(abs(d%se_umean(:, 1:nElemO) - ustar(:, 1:nElemO)))
        scale = maxval(abs(d0%se_ubt(:, 1:nElemO, :)))
        call max_all(res, partit); call max_all(scale, partit)
        call check(res <= 1.0e-13_WP*scale, 'C9a <<U>> = U* for M = 1', partit%mype)
        ! U^1 from the operators: eta^A on owned + halo with eta^1 exchanged as se_subcycle does
        eta1(1:nNodL) = d%se_etam(1:nNodL)
        do n = 1, nNodL
            infl(n) = d0%se_delta*eta1(n) + (1.0_WP - d0%se_delta - d0%se_gamma - d0%se_zeta)*d0%eta_n(n) &
                    + d0%se_gamma*d0%se_eta(n, 1) + d0%se_zeta*d0%se_eta(n, 2)
        end do
        HA = d0%se_H
        dU = 0.0_WP
        call se_tendency(d0, mesh, partit, ustar, infl, HA, dU, tend)
        res = 0.0_WP; scale = 0.0_WP
        do e = 1, nElemO
            res   = max(res, maxval(abs(d%se_ubt(:, e, 0) - (d0%se_ubt(:, e, 0) + dt*tend(:, e)))))
            scale = max(scale, maxval(abs(d0%se_ubt(:, e, 0))), dt*maxval(abs(tend(:, e))))
        end do
        call max_all(res, partit); call max_all(scale, partit)
        call check(res <= 1.0e-13_WP*scale, 'C9a U^1 = U^0 + dtbt*se_tendency', partit%mype)

        ! ---- b) rest stays at rest ----
        d = d0
        d%eta_n = 0.0_WP; d%se_eta = 0.0_WP; d%se_ubt = 0.0_WP; d%se_ubtn = 0.0_WP; d%se_force = 0.0_WP
        d%se_M = 40
        call se_subcycle(d, mesh, partit, dt)
        nbad = count(d%se_eta /= 0.0_WP) + count(d%se_etam /= 0.0_WP) + count(d%se_ubt(:, 1:nElemO, :) /= 0.0_WP) &
             + count(d%se_umean(:, 1:nElemO) /= 0.0_WP)
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C9b rest stays at rest (bitwise, M = 40)', partit%mype)

        ! ---- c) volume and eta-compatibility, M = 40 ----
        d = d0
        d%se_M = 40
        call se_subcycle(d, mesh, partit, dt)
        vsum = 0.0_WP; vabs = 0.0_WP
        do n = 1, nNodO
            vsum = vsum + mesh%areasvol(n)*(d%se_etam(n) - d0%eta_n(n))
            vabs = vabs + mesh%areasvol(n)*abs(d%se_etam(n) - d0%eta_n(n))
        end do
        call sum_all(vsum, partit); call sum_all(vabs, partit)
        call check(vabs > 0.0_WP .and. abs(vsum) <= 1.0e-12_WP*vabs, &
                   'C9c sum_v areasvol*(eta^M - eta^0) = 0 (closed basin)', partit%mype)
        call se_inflow(mesh, partit, d%se_umean, infl)
        res = 0.0_WP; scale = 0.0_WP
        do n = 1, nNodO
            res   = max(res, abs((d%se_etam(n) - d0%eta_n(n)) - dt*infl(n)/mesh%areasvol(n)))
            scale = max(scale, abs(d0%eta_n(n)), abs(d%se_etam(n) - d0%eta_n(n)))
        end do
        call max_all(res, partit); call max_all(scale, partit)
        call check(res <= 1.0e-12_WP*scale, 'C9c eta^M - eta^0 = tau*I(<<U>>)/areasvol', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3,a,es10.3)') &
            'test_ssh_se: C9c M = 40: sum areasvol*d(eta) = ', vsum, ' m3 of ', vabs, &
            '; eta-compatibility max residual ', res, ' m of ', scale
        if (partit%npes > 1) then
            allocate(tmp(nElemL))
            stale = max(stale_halo_max_nod(d%se_etam, partit), stale_halo_max_nod(d%se_eta(:, 1), partit), &
                        stale_halo_max_nod(d%se_eta(:, 2), partit))
            do c = 1, 2
                tmp = d%se_ubt(c, 1:nElemL, 0)
                stale = max(stale, stale_halo_max_elem(tmp, partit))
            end do
            call max_all(stale, partit)
            call check(stale == 0.0_WP, 'C9c eta levels and U^M halo-valid after the substeps', partit%mype)
            deallocate(tmp)
        end if

        ! ---- d) dtbt convergence: a smooth gravity wave, chi = 0 ----
        call new_dyn(dref)
        ! the bump centre: the position of the node with global id 1000 (MAX over the ranks
        ! holding a copy; every copy has the same coordinates)
        p0 = -huge(1.0_WP)
        do n = 1, nNodL
            if (partit%myList_nod2D(n) /= 1000) cycle
            p0(1) = cos(mesh%coord_nod2D(2, n))*cos(mesh%coord_nod2D(1, n))
            p0(2) = cos(mesh%coord_nod2D(2, n))*sin(mesh%coord_nod2D(1, n))
            p0(3) = sin(mesh%coord_nod2D(2, n))
        end do
        do c = 1, 3
            call max_all(p0(c), partit)
        end do
        do n = 1, nNodL
            x = cos(mesh%coord_nod2D(2, n))*cos(mesh%coord_nod2D(1, n)) - p0(1)
            y = cos(mesh%coord_nod2D(2, n))*sin(mesh%coord_nod2D(1, n)) - p0(2)
            z = sin(mesh%coord_nod2D(2, n)) - p0(3)
            dref%eta_n(n) = exp(-(x*x + y*y + z*z)*(r_earth/1.5e6_WP)**2)     ! chord distance
        end do
        call se_init(dref, mesh, partit, dt)                    ! eta ring = eta_n, U = 0, F = 0
        call se_elem_depth(dref, mesh, partit)
        dref%se_chi = 0.0_WP
        call se_sm_coeffs(0.0_WP, dref%se_beta, dref%se_zeta, dref%se_gamma, dref%se_delta)
        Mk = [16, 32, 64, 128]
        allocate(etaM(nNodO, 4), uM(2, nElemO, 4))
        do i = 1, 4
            d = dref
            d%se_M = Mk(i)
            call se_subcycle(d, mesh, partit, dt)
            etaM(:, i) = d%se_etam(1:nNodO)
            uM(:, :, i) = d%se_ubt(:, 1:nElemO, 0)
        end do
        do i = 1, 3
            ev(i) = maxval(abs(etaM(:, i) - etaM(:, 4)))
            eu(i) = maxval(abs(uM(:, :, i) - uM(:, :, 4)))
            call max_all(ev(i), partit); call max_all(eu(i), partit)
        end do
        res = maxval(abs(etaM(:, 4) - dref%eta_n(1:nNodO)))
        call max_all(res, partit)
        pe = log(ev(1:2)/ev(2:3))/log(2.0_WP)
        pu = log(eu(1:2)/eu(2:3))/log(2.0_WP)
        call check(res > 1.0e-2_WP, 'C9d the wave moves eta by > 1 cm in one step', partit%mype)
        call check(all(pe >= 2.0_WP) .and. all(pu >= 2.0_WP), 'C9d dtbt convergence order >= 2 (eta, U)', &
                   partit%mype)
        if (partit%mype == 0) then
            write(*,'(a,es10.3,a)') 'test_ssh_se: C9d gravity wave, chi = 0: max|eta^M - eta^0| = ', res, ' m'
            write(*,'(a,3es10.3,a,2f6.2)') 'test_ssh_se: C9d   eta errors vs M = 128 (M = 16, 32, 64): ', &
                ev, '; orders ', pe
            write(*,'(a,3es10.3,a,2f6.2)') 'test_ssh_se: C9d   U   errors vs M = 128 (M = 16, 32, 64): ', &
                eu, '; orders ', pu
        end if
        deallocate(etaM, uM)
    end subroutine c9_subcycle

    subroutine hashed_step_state(d0)
        ! a hashed SE state at the end of a predictor: uv = u^{n-1/2} and UV_rhs on every held
        ! element (wet layers), eta^n, H^n, the eta and U rings, F and U^n; hbar = eta^n. Every
        ! copy of a node or element holds the same value (hashed from the global id).
        type(t_dyn), intent(inout) :: d0
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, c
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(d0)
        call set_uv(d0, 0)
        do e = 1, nElemF
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                do c = 1, 2
                    d0%uv_rhs(c, nz, e) = 0.01_WP*hval(partit%myList_elem2D(e), 10*nz + c + 50)
                end do
            end do
        end do
        do n = 1, nNodL
            d0%eta_n(n) = 0.1_WP*hval(partit%myList_nod2D(n), 11)
        end do
        call se_init(d0, mesh, partit, dt)
        call se_elem_depth(d0, mesh, partit)
        do n = 1, nNodL
            d0%se_eta(n, 1) = 0.1_WP*hval(partit%myList_nod2D(n), 12)
            d0%se_eta(n, 2) = 0.1_WP*hval(partit%myList_nod2D(n), 13)
        end do
        do e = 1, nElemL
            do c = 1, 2
                d0%se_ubt(c, e, 0) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 20 + c)
                d0%se_ubt(c, e, 1) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 22 + c)
                d0%se_ubt(c, e, 2) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 24 + c)
                d0%se_force(c, e)  = 1.0e-3_WP*hval(partit%myList_elem2D(e), 26 + c)
            end do
            d0%se_ubtn(:, e) = d0%se_ubt(:, e, 0)
        end do
        if (.not. allocated(mesh%hbar)) allocate(mesh%hbar(nNodL), mesh%hbar_old(nNodL))
        mesh%hbar(1:nNodL)     = d0%eta_n(1:nNodL)
        mesh%hbar_old(1:nNodL) = 0.0_WP
    end subroutine hashed_step_state

    subroutine c10_trim_finish()
        ! Task 1.3: se_trim_finish on the hashed state with a hashed <<U>> and eta^M (no substeps:
        ! the trim and the commit only), FESOM3_SE_CHECK off.
        type(t_dyn) :: d0, d
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, c, nbad
        real(kind=WP), allocatable :: wf(:), infl(:), hbar0(:), tmp(:)
        real(kind=WP) :: s(2), sa(2), ustar2(2), cexp(2), res, scale, resu, scaleu, cmax, stale
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call hashed_step_state(d0)
        do e = 1, nElemL
            do c = 1, 2
                d0%se_umean(c, e) = d0%se_H(e)*0.05_WP*hval(partit%myList_elem2D(e), 33 + c)
            end do
        end do
        do n = 1, nNodL
            d0%se_etam(n)  = 0.2_WP*hval(partit%myList_nod2D(n), 36)
            mesh%hbar(n)   = 0.3_WP*hval(partit%myList_nod2D(n), 37)      ! hbar /= eta_n on purpose
        end do
        hbar0 = mesh%hbar(1:nNodL)
        allocate(wf(nNodL), infl(nNodL)); wf = 0.0_WP
        call check(.not. d0%se_check, 'C10 FESOM3_SE_CHECK is off in the test environment', partit%mype)
        d = d0
        call se_trim_finish(d, mesh, partit, dt, 1, wf)

        ! (a) the trim: sum_k helem^n uv = <<U>>; (b) the correction is depth-uniform and equals
        ! (sum_k helem u** - <<U>>)/H^n; (c) the layers below nlevels are untouched
        res = 0.0_WP; scale = 0.0_WP; resu = 0.0_WP; scaleu = 0.0_WP; cmax = 0.0_WP; nbad = 0
        do e = 1, nElemO
            s = 0.0_WP; sa = 0.0_WP; ustar2 = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                s  = s  + mesh%helem(nz, e)*d%uv(:, nz, e)
                sa = sa + mesh%helem(nz, e)*abs(d0%uv(:, nz, e) + d0%uv_rhs(:, nz, e))
                ustar2 = ustar2 + mesh%helem(nz, e)*(d0%uv(:, nz, e) + d0%uv_rhs(:, nz, e))
            end do
            res   = max(res, maxval(abs(s - d0%se_umean(:, e))))
            scale = max(scale, maxval(sa), maxval(abs(d0%se_umean(:, e))))
            cexp  = (ustar2 - d0%se_umean(:, e))/d0%se_H(e)
            cmax  = max(cmax, maxval(abs(cexp)))
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                resu   = max(resu, maxval(abs((d%uv(:, nz, e) - (d0%uv(:, nz, e) + d0%uv_rhs(:, nz, e))) + cexp)))
                scaleu = max(scaleu, maxval(abs(d0%uv(:, nz, e) + d0%uv_rhs(:, nz, e))))
            end do
            do nz = mesh%nlevels(e), mesh%nl - 1
                if (any(d%uv(:, nz, e) /= d0%uv(:, nz, e))) nbad = nbad + 1
            end do
        end do
        call max_all(res, partit); call max_all(scale, partit); call max_all(resu, partit)
        call max_all(scaleu, partit); call max_all(cmax, partit); call sum_all(nbad, partit)
        call check(res <= 1.0e-13_WP*scale, 'C10 trim: sum_k helem^n uv = <<U>>', partit%mype)
        call check(cmax > 1.0e-2_WP*scaleu, 'C10 the trim correction is non-trivial', partit%mype)
        call check(resu <= 1.0e-13_WP*(scaleu + cmax), &
                   'C10 the correction is depth-uniform: uv_k = u**_k - (sum helem u** - <<U>>)/H', partit%mype)
        call check(nbad == 0, 'C10 the layers below nlevels are untouched (bitwise)', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,es10.3)') &
            'test_ssh_se: C10 trim: max|sum helem uv - <<U>>| = ', res, ' m2/s of ', scale, &
            '; depth-uniform correction residual ', resu, ' m/s of ', scaleu, ', max|correction| ', cmax

        ! (d) the commit on owned + halo nodes, (e) ssh_rhs_old = I(<<U>>) on owned nodes, (f) the
        ! U ring and <<U>> untouched, nothing of SE_CHECK computed
        nbad = count(mesh%hbar_old(1:nNodL) /= hbar0) + count(mesh%hbar(1:nNodL) /= d0%se_etam) &
             + count(d%eta_n(1:nNodL) /= d0%se_etam)
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C10 hbar_old = hbar, hbar = eta_n = eta^M on owned + halo (bitwise)', partit%mype)
        call se_inflow(mesh, partit, d0%se_umean, infl)
        nbad = count(d%ssh_rhs_old(1:nNodO) /= infl(1:nNodO))
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C10 ssh_rhs_old = I(<<U>>) on owned nodes (bitwise)', partit%mype)
        nbad = count(d%se_ubt(:, 1:nElemL, :) /= d0%se_ubt(:, 1:nElemL, :)) &
             + count(d%se_umean(:, 1:nElemO) /= d0%se_umean(:, 1:nElemO))
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C10 the U ring and <<U>> are untouched', partit%mype)
        call check(.not. d%se_msl0_set, 'C10 SE_CHECK off: nothing computed', partit%mype)

        ! (g) npes > 1: uv synced over the shared elements and halo-valid, as after update_vel
        if (partit%npes > 1) then
            stale = shared_copy_max_elem(d%uv, partit)
            allocate(tmp(nElemL))
            do c = 1, 2
                do nz = 1, 2
                    tmp = d%uv(c, nz, 1:nElemL)
                    stale = max(stale, stale_halo_max_elem(tmp, partit))
                end do
            end do
            call max_all(stale, partit)
            call check(stale == 0.0_WP, 'C10 uv: shared copies equal and halo valid after the trim', partit%mype)
            deallocate(tmp)
        end if
    end subroutine c10_trim_finish

    subroutine c11_check()
        ! Task 1.4: FESOM3_SE_CHECK on a full SE step (M = 40 substeps + the trim) of the hashed
        ! state; then a perturbed uv on one element and a NaN.
        type(t_dyn) :: d0, d, dp
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, k0
        real(kind=WP), allocatable :: wf(:)
        real(kind=WP) :: vals(SE_NCHK), v2(SE_NCHK), sa(2), etascale, trimscale, mxe, mxu
        real(kind=WP) :: expect, delta
        character(len=256) :: line
        logical :: halt
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call hashed_step_state(d0)
        allocate(wf(nNodL)); wf = 0.0_WP
        d = d0
        d%se_M = 40
        d%se_check = .true.
        call se_subcycle(d, mesh, partit, dt)
        call se_trim_finish(d, mesh, partit, dt, 7, wf)          ! prints the SE_CHECK line (rank 0)
        call check(d%se_msl0_set, 'C11 SE_CHECK on: the MSL reference is set by the first check', partit%mype)
        call se_check_values(d, mesh, partit, dt, d0%eta_n, vals)

        ! scales and the direct maxima
        etascale = 0.0_WP; mxe = 0.0_WP
        do n = 1, nNodO
            etascale = max(etascale, abs(d0%eta_n(n)), abs(d%eta_n(n) - d0%eta_n(n)))
            mxe      = max(mxe, abs(d%eta_n(n)))
        end do
        trimscale = 0.0_WP; mxu = 0.0_WP
        do e = 1, nElemO
            sa = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                sa = sa + mesh%helem(nz, e)*abs(d%uv(:, nz, e))
            end do
            trimscale = max(trimscale, maxval(sa))
            mxu = max(mxu, maxval(abs(d%se_ubt(:, e, 0))))
        end do
        call max_all(etascale, partit); call max_all(mxe, partit)
        call max_all(trimscale, partit); call max_all(mxu, partit)
        call check(vals(SE_CHK_ETACOMP) <= 1.0e-12_WP*etascale, 'C11 eta-compatibility at rounding', partit%mype)
        call check(vals(SE_CHK_TRIM) <= 1.0e-12_WP*trimscale, 'C11 trim at rounding', partit%mype)
        call check(abs(vals(SE_CHK_MSL)) <= 1.0e-12_WP*etascale, 'C11 mean sea level unchanged (closed basin)', &
                   partit%mype)
        call check(vals(SE_CHK_SPREADU) == 0.0_WP .and. vals(SE_CHK_SPREADUM) == 0.0_WP, &
                   'C11 copy spreads of U^M and <<U>> are 0', partit%mype)
        call check(vals(SE_CHK_MAXETA) == mxe .and. vals(SE_CHK_MAXU) == mxu .and. mxe > 0.0_WP, &
                   'C11 max|eta|, max|U| = the direct maxima', partit%mype)
        call check(vals(SE_CHK_NBAD) == 0.0_WP, 'C11 no non-finite value', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,2es10.3)') &
            'test_ssh_se: C11 valid state: etacomp ', vals(SE_CHK_ETACOMP), ' m of ', etascale, &
            '; trim ', vals(SE_CHK_TRIM), ' m2/s of ', trimscale, '; msl ', vals(SE_CHK_MSL), &
            ' m; spreads ', vals(SE_CHK_SPREADU), vals(SE_CHK_SPREADUM)

        ! the line format
        line = se_check_line(7, vals)
        call check(index(line, 'SE_CHECK step=7 etacomp=') == 1 .and. index(line, ' trim=') > 0 .and. &
                   index(line, ' msl=') > 0 .and. index(line, ' spreadU=') > 0 .and. &
                   index(line, ' spreadUm=') > 0 .and. index(line, ' maxeta=') > 0 .and. &
                   index(line, ' maxU=') > 0 .and. index(trim(line), '  ') == 0 .and. &
                   index(trim(line), '= ') == 0, 'C11 the SE_CHECK line: key=value tokens, '// &
                   'single blanks: '//trim(line), partit%mype)

        ! a perturbed uv on one element: rank 0's first element, its top layer
        delta = 1.0e-3_WP
        expect = 0.0_WP
        dp = d
        if (partit%mype == 0) then
            k0 = mesh%ulevels(1)
            dp%uv(1, k0, 1) = dp%uv(1, k0, 1) + delta
            expect = real(mesh%helem(k0, 1), WP)*delta
        end if
        call max_all(expect, partit)
        call se_check_values(dp, mesh, partit, dt, d0%eta_n, v2)
        call check(abs(v2(SE_CHK_TRIM) - expect) <= 1.0e-6_WP*expect .and. &
                   v2(SE_CHK_TRIM) > 1.0e6_WP*vals(SE_CHK_TRIM), &
                   'C11 a perturbed uv on one element shows in the trim check (= helem*delta)', partit%mype)
        call check(v2(SE_CHK_ETACOMP) == vals(SE_CHK_ETACOMP) .and. v2(SE_CHK_NBAD) == 0.0_WP, &
                   'C11 the perturbation leaves the other checks unchanged', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a)') &
            'test_ssh_se: C11 perturbed uv: trim ', v2(SE_CHK_TRIM), ' m2/s (helem*delta = ', expect, ')'
        ! a NaN on one element. The Intel build traps invalid operations (-fpe0), and the max and
        ! sum over a NaN raise one: halting off around the check (a NaN reaches SE_CHECK in a run
        ! only without that trap, e.g. the GNU build; with it the producing operation stops first)
        halt = .false.
        if (ieee_support_halting(ieee_invalid)) then
            call ieee_get_halting_mode(ieee_invalid, halt)
            call ieee_set_halting_mode(ieee_invalid, .false.)
        end if
        if (partit%mype == 0) dp%uv(2, mesh%ulevels(1), 1) = ieee_value(1.0_WP, ieee_quiet_nan)
        call se_check_values(dp, mesh, partit, dt, d0%eta_n, v2)
        call ieee_set_flag(ieee_invalid, .false.)
        if (ieee_support_halting(ieee_invalid)) call ieee_set_halting_mode(ieee_invalid, halt)
        call check(v2(SE_CHK_NBAD) == 1.0_WP, 'C11 a NaN in uv is counted (NBAD = 1)', partit%mype)
    end subroutine c11_check

end program test_ssh_se
