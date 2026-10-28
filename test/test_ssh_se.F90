program test_ssh_se
    ! Split-explicit barotropic solver (docs/plans/completed/2026-10-17-split-explicit.md), Tasks 0.3-1.4: the
    ! knobs, the CFL guard, se_init, the barotropic forcing, the substeps, the trim and the
    ! invariant suite of src/oce/oce_ssh_se.F90 on the rotated pi mesh.
    !
    !   C1  the SM coefficients at chi = 0.1 equal the literal numbers worked out by hand from
    !       the notes (beta = 0.281105, zeta = -0.003689497, gamma = 0.0320866, delta =
    !       0.724707606; relative 1e-9), which catches a typo in a constant; at chi = 0 the
    !       constant terms (zeta = 0.00976186, gamma = 0.083445, delta = 0.60296872); at
    !       chi = 0.25 zeta = -0.0238665325, gamma = -0.044951, delta = 0.907315935. The
    !       AB3 and AM4 weights sum to 1.
    !   C2  knob parsing (se_parse_knob): every bad value is rejected (ok = .false., the t_dyn
    !       field unchanged, the message starts with the knob name); se_read_env and
    !       se_env_requested stop on exactly these. Good values land in t_dyn. The defaults of
    !       the flow-dependent viscosity (off, gamma0 10, gamma1 2750), of the bottom drag
    !       (FESOM3_SE_BOTTDRAG on; 0/1 only) and se_knob_conflict (a set FESOM3_SE_VISC_AH with
    !       FLOW=1, a set GAMMA0/1 without it).
    !   C3  the CFL guard (plan Task 0.4; runs after C4, the operator reads helem):
    !       a) a_crit(chi) = 1.853 / 1.579 / 1.329 at chi = 0 / 0.1 / 0.25 to 1e-3 (the numpy root
    !          finder of plan Task 5.1); se_cfl_stable flips between 0.999 and 1.001 a_crit; the
    !          spectral radius at 1.1 a_crit equals numpy's 2.186123 / 1.879877 / 1.700745 to
    !          1e-5, at 0.9 a_crit it is < 1;
    !       b) A (se_cfl_apply_a) is self-adjoint in the areasvol inner product, (u, A w)_W =
    !          (A u, w)_W to 1e-12 of ||u||_W ||A w||_W, and (u, A u)_W > 0 for hashed u, w (so
    !          the power-iteration value is a lower bound of omega_max);
    !       c) omega_max (se_cfl_omega_max): finite, > 0, converged (last change < 1e-6), >= the
    !          Rayleigh quotient sqrt of the hashed vectors, identical on every rank (MIN == MAX)
    !          like M_min; np > 1 against the np 1 value (file FESOM3_TEST_SE_P4REF//'.c3', written
    !          by np 1): omega_max to 1e-12 relative, M_min equal;
    !       d) at the default chi = 0.1, M_min = ceiling(tau*omega_max/(s*a_crit)):
    !          tau/M_min <= s*a_crit/omega_max < tau/(M_min - 1), M_min > 1 on pi, and the
    !          decision: SE_CFL_DIE for M = 1 and M_min - 1, SE_CFL_FORCED with
    !          FESOM3_SE_M_FORCE, SE_CFL_OK for M_min, 40 and 50;
    !       e) EMPIRICAL, the real se_subcycle (F = 0, U = 0, eta = the power-iteration vector,
    !          viscosity on, one call per baroclinic step) for chi = 0 / 0.1 / 0.25: at M = M_min
    !          for 300 baroclinic steps and at omega_max*dtbt = 0.97 a_crit for 3000 substeps the
    !          norm ||eta||_W does not grow (max over the last 10 calls <= the initial); at
    !          omega_max*dtbt = 1.03 a_crit and 1.1 a_crit it grows by > 1e6, at 1.1 a_crit with a
    !          rate per substep within 10 % of se_cfl_rho.
    !   C4  se_init under the ctest environment FESOM3_SSH_SE=1 FESOM3_SE_M=40: the knobs from
    !       the environment (M = 40; chi, visc_ah, bottom drag, check, force at their defaults),
    !       the coefficients in t_dyn, the state at the extents of the plan's Technical Details §1,
    !       the cold start (eta ring == eta_n bitwise; the three U levels bitwise equal and
    !       == sum_k helem*uv on owned + eDim elements; s_old == 0; the drag rate 0 until the
    !       first forcing) and se_ready. eta_n and uv are hashed from the global ids, so every
    !       copy of a node or element holds the same value.
    !   C5  se_elem_depth and the PRESSURE part of the forcing (Task 1.1): H^n == sum_k helem on
    !       owned + eDim elements (bitwise, positive); with uv = 0, pgf = 0, momentum advection off
    !       and a non-trivial eta_n, the real compute_vel_rhs puts only -g grad(eta) into UV_rhs,
    !       and se_bt_forcing's g H^n grad(eta^n) must remove it: max|F| <= 1e-12 * max|H^n Fx|.
    !   C6  the CORIOLIS part of the forcing, the lfirst weight (Task 1.1) and the drag rate
    !       (Task 5.4): momentum advection off, eta_n = 0, pgf = 0, helem constant. Step 1 (lfirst,
    !       cold uv_rhsAB, s_old = 0): the real compute_vel_rhs on uv^{n-3/2}, then F = 0 to
    !       rounding; the same UV_rhs taken as a non-lfirst step leaves exactly (0.5+eps) f e_z x
    !       s^n (the lfirst weight a2 = 1 matters). s_old == sum_k helem*uv^{n-3/2} afterwards
    !       (bitwise). The drag rate kdrag = C_d |uv(:, nlevels-1)|/H^n against an independent
    !       recomputation (1e-14), > 0 where the bottom velocity is not 0, finite; with
    !       FESOM3_SE_BOTTDRAG off exactly 0 and F bitwise the same (the drag is not in F).
    !       Step 2 (not lfirst): uv^{n-1/2}, compute_vel_rhs blends ab2*C(uv^{n-1/2}) +
    !       ab1*C(uv^{n-3/2}), and F = 0 to rounding again. se_ubtn == ring level 0 (bitwise,
    !       owned + eDim). Tolerances are relative to the largest Coriolis term max|f s|; the
    !       measured residuals are printed. (Plan Task 5.4 had replaced the stored sums by the
    !       first-substep Coriolis f e_z x U*^0; Task 5.5 restored them, and these checks.)
    !   C7  operator / r-term consistency (Task 1.2): the real compute_vel_rhs twice on the SAME uv
    !       (so s^n = s^{n-1}) with a non-trivial eta_n and pgf, se_bt_forcing after each (the
    !       drag on: kdrag > 0); then se_tendency with U* = s^n, eta^A = eta^n, H^A = H^n, dU = 0
    !       and U^m = U^n (so D = 0) equals G/tau = sum_k helem*UV_rhs/tau to 1e-12 of the largest
    !       term. A sign or factor mismatch between F and the operator (Coriolis, pressure) fails
    !       this.
    !   C12 the barotropic bottom drag D (Task 5.4) through se_tendency (U* = 0, eta = 0, no
    !       viscosity, F = 0, kdrag from se_bt_forcing on a hashed uv): with U^m = U^n the
    !       tendency is bitwise that of FESOM3_SE_BOTTDRAG off; with U^m = U^n + X it changes by
    !       -kdrag*X (1e-14), by twice that for 2X (linear) and by the sum for X + Y (additive);
    !       max dtbt*kdrag (pi's hashed uv; CORE2 shelf: <= 4.5e-3 per m/s of bottom speed) is
    !       printed. Then a full M = 40 substep cycle with the drag on and off: both finite, they
    !       differ (the drag acts), and the eta-compatibility of the drag run holds (1e-12).
    !   R1  the restart registration (Tasks 5.4, 5.5): with SE on the checkpoint carries exactly
    !       10 SE stores (se_eta1, se_eta2, se_ubt{0,1,2}_{u,v}, se_sbar_old_{u,v}), R%se is set;
    !       with SE off none, and the other stores are the same.
    !   C8  the viscosity (Task 1.2): kappa >= 0, kappa = 0 exactly on boundary slots and on a
    !       triangle's slot 4, > 0 on every interior slot; kappa = A_h*l/(d*A_e) against an
    !       independent recomputation from edge_len (owned edges) and the elem_center distance;
    !       sum_e A_e*V_e = 0 to 1e-12 of sum_e A_e*|V_e| for a hashed halo-valid dU (each
    !       element counted once: on the rank owning its first vertex); V of a uniform velocity is
    !       exactly 0 over a flat bottom and 0 to rounding over the real one.
    !   C8f the flow-dependent viscosity W (se_visc_flow, FESOM3_SE_VISC_FLOW=1) on a hashed
    !       halo-valid transport (velocity differences ~5 cm/s, so both branches of
    !       max(gamma0, gamma1*|dU|/H) occur): zlen = sqrt(A_e + A_nb) on interior slots (1e-15),
    !       0 on boundary slots and a triangle's slot 4, bitwise the same from both sides of an
    !       edge; sum_e A_e*W_e = 0 to 1e-12 of sum_e A_e*|W_e| (each element once); W against an
    !       independent edge scatter in the form of the Zenodo code (owned edges; elements whose
    !       edges are all owned) to 1e-13 of the largest flux sum, for gamma1 = 2750 and 0; W of
    !       a uniform transport == 0 exactly; gamma1 = 0: W(2U) == 2 W(U) bitwise and W(U + U2) =
    !       W(U) + W(U2) to 1e-13 (linear, coefficient sqrt(gamma0*zlen) by the scatter check);
    !       gamma0 = 0: W(4U) = 8 W(U) to 1e-13 (the flow term alone, ~|dU|^(3/2)); at npes > 1
    !       the copy spread of W over the shared elements is 0 for all three settings (LESSONS
    !       L71). Then one substep (M = 1) with W from U^0 = U^n: W(U^0) - W(U^n) == 0, so U^1 on
    !       the owned elements equals the run without viscosity bitwise.
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
    !   P5  P2 with the flow-dependent viscosity: NK = 6 SE steps through the real kernels with
    !       FESOM3_SE_VISC_FLOW on and the shared-element sync on: the copy spreads of U (3
    !       levels), <<U>> and uv are 0 after every step, the state is finite, and U^M differs
    !       from the linear viscosity's (the switch acts).
    !   Z1-Z6 zstar (Task 3.1): the depth law, the freshwater term and the zstar checks through the
    !       real ALE tail; their own block at the end of this file (run last).
    ! FESOM3_MESH_DIR (default pi), FESOM3_TEST_ROTATE=1 reads it rotated (setup_test_mesh).
    !   S1  RK3 (docs/plans/2026-10-28-rk3.md, Task 13): se_bt_forcing_rk3 with momentum advection
    !       off, pgf = 0, no viscosity, no wind and no bottom velocity (so no drag): R1 from the real
    !       compute_vel_rhs over tau/3 (no_history) on a hashed uv and eta_n; F = 0 to 1e-12 of
    !       max(H|grad p|, |f s|): the pressure and the Coriolis of s^n = sum_k helem*u^n are
    !       removed exactly, without AB2 weights
    !   S2  the same with eta = 0, a hashed bottom velocity, viscosity increment and wind stress:
    !       F = sum_k helem*visc/tau + tau_s/rho_0 - C_d|u_b|u_b to 1e-12 relative, kdrag =
    !       C_d|u_b|/H (1e-14), se_ubtn == ring level 0
    !   S3  np 2: the copies of F and kdrag on the shared elements agree bitwise
    use mpi
    use mod_precision,    only: WP, MP, MPI_WP
    use mod_constants,    only: g
    use mod_mesh,         only: t_mesh, MAX_NV
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init
    use mod_dyn,          only: t_dyn
    use mod_part_bounds,  only: local_dims
    use mod_config,       only: ab_epsilon
    use mod_param_phys,   only: C_d
    use mod_constants,    only: r_earth
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use, intrinsic :: ieee_exceptions, only: ieee_invalid, ieee_support_halting, ieee_get_halting_mode, &
                                             ieee_set_halting_mode, ieee_set_flag
    use mod_halo,         only: stale_halo_max_nod, stale_halo_max_elem, shared_copy_max_elem
    use mod_mesh_areas,   only: elem_center
    use mod_mesh_rotate,  only: trim_cyclic
    use oce_dyn_velrhs,   only: compute_vel_rhs
    use mod_constants,    only: density_0
    use oce_ssh_se,       only: se_bt_forcing_rk3
    use oce_dyn_visc,     only: viscosity_filter
    use oce_ssh_se,       only: se_env_requested, se_parse_knob, se_init, se_sm_coeffs, &
                                se_cfl_m_min, se_cfl_decision, se_cfl_stable, se_cfl_rho, &
                                se_cfl_acrit, se_cfl_apply_a, se_cfl_omega_max, SE_CFL_SAFETY, &
                                SE_CFL_OK, SE_CFL_FORCED, SE_CFL_DIE, se_bt_forcing, &
                                se_elem_depth, se_subcycle, se_inflow, se_visc, se_tendency, &
                                se_visc_flow, se_knob_conflict, &
                                se_trim_finish, se_check_values, se_check_line, SE_NCHK, &
                                SE_CHK_ETACOMP, SE_CHK_TRIM, SE_CHK_MSL, SE_CHK_SPREADU, &
                                SE_CHK_SPREADUM, SE_CHK_MAXETA, SE_CHK_MAXU, SE_CHK_NBAD
    use test_util,        only: setup_test_mesh, check, finish, sum_all, max_all, hval
    ! zstar block (Z1-Z6)
    use mod_config,       only: which_ALE
    use oce_ale,          only: vert_vel_ale, update_thickness_ale
    use oce_ssh_se,       only: se_depth_at, se_check_tail_values, se_check_tail_line, SE_NTAIL, &
                                SE_TAIL_LAYSUM, SE_TAIL_W1, SE_TAIL_HLAW, SE_TAIL_NBAD, SE_CHK_FW
    ! restart registry (R1)
    use mod_io_restart,   only: t_restart, restart_register_state
    use mod_tracer,       only: t_tracer
    use mod_ice,          only: t_ice
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
    call c4_se_init()
    call c3_cfl_guard()           ! after C4: the operator reads helem
    call c5_forcing_pressure()
    call c6_forcing_coriolis()
    call s_rk3_forcing()
    call c7_consistency()
    call c12_bottdrag()
    call c8_visc()
    call c8f_visc_flow()
    call c9_subcycle()
    call c10_trim_finish()
    call p_many_ranks()
    call p5_flow()
    call c11_check()
    call z_zstar()                ! it leaves the zstar thickness fields allocated
    call r1_restart_registry()    ! after Z: registers pointers into mesh%hbar / hnode
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
        call se_sm_coeffs(0.25_WP, b, z, gm, d)
        call check(near(b, 0.281105_WP) .and. near(z, -0.0238665325_WP) .and. near(gm, -0.044951_WP) &
                   .and. near(d, 0.907315935_WP), &
                   'C1 chi=0.25: zeta = -0.0238665325, gamma = -0.044951, delta = 0.907315935', partit%mype)
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
        call bad('FESOM3_SE_VISC_FLOW', '2'); call bad('FESOM3_SE_VISC_FLOW', 'yes')
        call bad('FESOM3_SE_VISC_FLOW', '')
        call bad('FESOM3_SE_VISC_GAMMA0', '-1'); call bad('FESOM3_SE_VISC_GAMMA0', 'nan')
        call bad('FESOM3_SE_VISC_GAMMA0', '10 20'); call bad('FESOM3_SE_VISC_GAMMA0', '')
        call bad('FESOM3_SE_VISC_GAMMA1', '-2750'); call bad('FESOM3_SE_VISC_GAMMA1', 'Inf')
        call bad('FESOM3_SE_VISC_GAMMA1', '2750x'); call bad('FESOM3_SE_VISC_GAMMA1', '2750,0')
        call bad('FESOM3_SE_BOTTDRAG', '2');  call bad('FESOM3_SE_BOTTDRAG', 'on')
        call bad('FESOM3_SE_BOTTDRAG', '-1'); call bad('FESOM3_SE_BOTTDRAG', '0.5')
        call bad('FESOM3_SE_BOTTDRAG', '')
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
        call check(.not. d%se_visc_flow .and. d%se_visc_g0 == 10.0_WP .and. d%se_visc_g1 == 2750.0_WP, &
                   'C2 defaults: linear viscosity, gamma0 10, gamma1 2750', partit%mype)
        call good('FESOM3_SE_VISC_FLOW', '1', d); call check(d%se_visc_flow, 'C2 FESOM3_SE_VISC_FLOW=1', partit%mype)
        call good('FESOM3_SE_VISC_FLOW', '0', d); call check(.not. d%se_visc_flow, 'C2 FESOM3_SE_VISC_FLOW=0', partit%mype)
        call good('FESOM3_SE_VISC_GAMMA0', '0', d)
        call check(d%se_visc_g0 == 0.0_WP, 'C2 FESOM3_SE_VISC_GAMMA0=0', partit%mype)
        call good('FESOM3_SE_VISC_GAMMA0', '12.5', d)
        call check(d%se_visc_g0 == 12.5_WP, 'C2 FESOM3_SE_VISC_GAMMA0=12.5', partit%mype)
        call good('FESOM3_SE_VISC_GAMMA1', '2.75e3', d)
        call check(d%se_visc_g1 == 2750.0_WP, 'C2 FESOM3_SE_VISC_GAMMA1=2.75e3', partit%mype)
        call good('FESOM3_SE_VISC_GAMMA1', '0', d)
        call check(d%se_visc_g1 == 0.0_WP, 'C2 FESOM3_SE_VISC_GAMMA1=0', partit%mype)
        call check(d%se_bottdrag, 'C2 default: barotropic bottom drag on', partit%mype)
        call good('FESOM3_SE_BOTTDRAG', '0', d);  call check(.not. d%se_bottdrag, 'C2 FESOM3_SE_BOTTDRAG=0', partit%mype)
        call good('FESOM3_SE_BOTTDRAG', ' 1', d); call check(d%se_bottdrag, 'C2 FESOM3_SE_BOTTDRAG=" 1"', partit%mype)
        ! the knobs that have no effect (flow, VISC_AH set, a GAMMA set)
        call check(len_trim(se_knob_conflict(.false., .false., .false.)) == 0 .and. &
                   len_trim(se_knob_conflict(.false., .true., .false.)) == 0 .and. &
                   len_trim(se_knob_conflict(.true., .false., .true.)) == 0 .and. &
                   len_trim(se_knob_conflict(.true., .false., .false.)) == 0, &
                   'C2 se_knob_conflict: no conflict for the meaningful combinations', partit%mype)
        call check(index(se_knob_conflict(.true., .true., .false.), 'FESOM3_SE_VISC_AH ') == 1, &
                   'C2 se_knob_conflict: FESOM3_SE_VISC_AH with FLOW=1', partit%mype)
        call check(index(se_knob_conflict(.false., .false., .true.), 'FESOM3_SE_VISC_GAMMA0/1 ') == 1 .and. &
                   index(se_knob_conflict(.false., .true., .true.), 'FESOM3_SE_VISC_GAMMA0/1 ') == 1, &
                   'C2 se_knob_conflict: GAMMA0/1 without FLOW=1', partit%mype)
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
                   (d%se_check .eqv. d0%se_check) .and. (d%se_m_force .eqv. d0%se_m_force) .and. &
                   (d%se_visc_flow .eqv. d0%se_visc_flow) .and. d%se_visc_g0 == d0%se_visc_g0 .and. &
                   d%se_visc_g1 == d0%se_visc_g1, &
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
        type(t_dyn) :: dbase
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, ic, ierr, mm(2), Mc, u_, ios
        integer :: hdr(2), hdr_ref(2), mref
        real(kind=WP), parameter :: CHIS(3)  = [0.0_WP, 0.1_WP, 0.25_WP]
        real(kind=WP), parameter :: ACREF(3) = [1.853_WP, 1.579_WP, 1.329_WP]       ! plan Task 5.1
        real(kind=WP), parameter :: RHO11(3) = [2.186123_WP, 1.879877_WP, 1.700745_WP] ! numpy roots
        real(kind=WP) :: ac(3), rho, omega, relchg, acd, lim, ll(2), sm(7), vmax, oref
        real(kind=WP) :: rl(4), re(4), rate(4)
        real(kind=WP), allocatable :: v(:), u(:), w(:), Au(:), Aw(:), Uw(:,:), infl(:)
        character(len=8)   :: cs
        character(len=512) :: path
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)

        ! ---- a) a_crit and the spectral radius of the scalar recurrence ----
        do ic = 1, 3
            write(cs, '(f4.2)') CHIS(ic)
            ac(ic) = se_cfl_acrit(CHIS(ic))
            call check(abs(ac(ic) - ACREF(ic)) <= 1.0e-3_WP, 'C3a chi = '//trim(cs)//': a_crit to 1e-3', partit%mype)
            call check(se_cfl_stable(0.999_WP*ac(ic), CHIS(ic), 1.0_WP) .and. &
                       .not. se_cfl_stable(1.001_WP*ac(ic), CHIS(ic), 1.0_WP), &
                       'C3a chi = '//trim(cs)//': stable at 0.999 a_crit, not at 1.001 a_crit', partit%mype)
            rho = se_cfl_rho(1.1_WP*ac(ic), CHIS(ic))
            call check(abs(rho - RHO11(ic)) <= 1.0e-5_WP, 'C3a chi = '//trim(cs)//': radius at 1.1 a_crit = numpy', &
                       partit%mype)
            call check(se_cfl_rho(0.9_WP*ac(ic), CHIS(ic)) < 1.0_WP, 'C3a chi = '//trim(cs)//': radius at 0.9 a_crit < 1', &
                       partit%mype)
            if (partit%mype == 0) write(*,'(a,a,a,f10.7,a,f9.6,a,f9.6)') 'test_ssh_se: C3a chi = ', trim(cs), &
                ': a_crit = ', ac(ic), ', radius at 0.9 a_crit ', se_cfl_rho(0.9_WP*ac(ic), CHIS(ic)), &
                ', at 1.1 a_crit ', rho
        end do

        ! ---- b) A is W-self-adjoint and >= 0 ----
        call new_dyn(dbase)
        call se_init(dbase, mesh, partit, dt)
        call se_elem_depth(dbase, mesh, partit)
        allocate(v(nNodL), u(nNodL), w(nNodL), Au(nNodL), Aw(nNodL), Uw(2, nElemL), infl(nNodL))
        v = 0.0_WP; Au = 0.0_WP; Aw = 0.0_WP; Uw = 0.0_WP; infl = 0.0_WP
        do n = 1, nNodL
            u(n) = hval(partit%myList_nod2D(n), 41)
            w(n) = hval(partit%myList_nod2D(n), 42)
        end do
        call se_cfl_apply_a(mesh, partit, dbase%se_H, u, Uw, infl, Au)
        call se_cfl_apply_a(mesh, partit, dbase%se_H, w, Uw, infl, Aw)
        sm = 0.0_WP
        do n = 1, nNodO
            sm(1) = sm(1) + mesh%areasvol(n)*u(n)*Aw(n)
            sm(2) = sm(2) + mesh%areasvol(n)*Au(n)*w(n)
            sm(3) = sm(3) + mesh%areasvol(n)*u(n)*Au(n)
            sm(4) = sm(4) + mesh%areasvol(n)*w(n)*Aw(n)
            sm(5) = sm(5) + mesh%areasvol(n)*u(n)*u(n)
            sm(6) = sm(6) + mesh%areasvol(n)*w(n)*w(n)
            sm(7) = sm(7) + mesh%areasvol(n)*Aw(n)*Aw(n)
        end do
        call sum_all(sm, partit)
        call check(abs(sm(1) - sm(2)) <= 1.0e-12_WP*sqrt(sm(5)*sm(7)), &
                   'C3b (u, A w)_W = (A u, w)_W (A self-adjoint)', partit%mype)
        call check(sm(3) > 0.0_WP .and. sm(4) > 0.0_WP, 'C3b (u, A u)_W > 0', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a)') 'test_ssh_se: C3b |(u,Aw) - (Au,w)| = ', &
            abs(sm(1) - sm(2))/sqrt(sm(5)*sm(7)), ' of ||u|| ||Aw||'

        ! ---- c) omega_max ----
        call se_cfl_omega_max(mesh, partit, omega, relchg, v)
        call check(omega > 0.0_WP .and. omega < huge(1.0_WP), 'C3c omega_max finite and > 0', partit%mype)
        call check(relchg < 1.0e-6_WP, 'C3c power iteration converged (last change < 1e-6)', partit%mype)
        call check(omega**2 >= max(sm(3)/sm(5), sm(4)/sm(6)), 'C3c omega_max^2 >= Rayleigh quotients', partit%mype)
        ll = [omega, -omega]
        call MPI_Allreduce(MPI_IN_PLACE, ll, 2, MPI_WP, MPI_MAX, partit%MPI_COMM_FESOM, ierr)
        call check(ll(1) == omega .and. -ll(2) == omega, 'C3c omega_max identical on every rank', partit%mype)

        ! ---- d) M_min and the decision (chi = 0.1, the se_init default) ----
        call check(dbase%se_chi == 0.1_WP, 'C3d chi = 0.1 (the default)', partit%mype)
        acd   = se_cfl_acrit(dbase%se_chi)
        lim   = SE_CFL_SAFETY*acd/omega
        M_min = se_cfl_m_min(dt, lim)
        mm = [M_min, -M_min]
        call MPI_Allreduce(MPI_IN_PLACE, mm, 2, MPI_INTEGER, MPI_MAX, partit%MPI_COMM_FESOM, ierr)
        call check(mm(1) == M_min .and. -mm(2) == M_min, 'C3d M_min identical on every rank', partit%mype)
        call check(M_min == ceiling(dt*omega/(SE_CFL_SAFETY*acd)) .or. &
                   abs(dt*omega/(SE_CFL_SAFETY*acd) - nint(dt*omega/(SE_CFL_SAFETY*acd))) < 1.0e-9_WP, &
                   'C3d M_min = ceiling(tau*omega_max/(s*a_crit))', partit%mype)
        call check(dt/real(M_min, WP) <= lim, 'C3d tau/M_min <= s*a_crit/omega_max', partit%mype)
        call check(M_min > 1, 'C3d pi M_min > 1 (M = 1 is below the guard)', partit%mype)
        if (M_min > 1) call check(dt/real(M_min - 1, WP) > lim, 'C3d tau/(M_min-1) > s*a_crit/omega_max', &
                                  partit%mype)
        call check(se_cfl_decision(1, M_min, .false.) == SE_CFL_DIE,    'C3d M = 1: the guard fires', partit%mype)
        call check(se_cfl_decision(1, M_min, .true.)  == SE_CFL_FORCED, 'C3d M = 1 + FORCE: warn and run', partit%mype)
        call check(se_cfl_decision(M_min - 1, M_min, .false.) == SE_CFL_DIE, 'C3d M = M_min-1 fires', partit%mype)
        call check(se_cfl_decision(M_min, M_min, .false.) == SE_CFL_OK,  'C3d M = M_min passes', partit%mype)
        call check(se_cfl_decision(40, M_min, .false.) == SE_CFL_OK, &
                   'C3d M = 40 (the ctest environment) passes on pi', partit%mype)
        call check(se_cfl_decision(50, M_min, .false.) == SE_CFL_OK, &
                   'C3d M = 50 (the default) passes on pi', partit%mype)
        call check(se_cfl_m_min(dt, huge(1.0_WP)) == 1, 'C3d no limit: M_min = 1', partit%mype)
        if (partit%mype == 0) write(*,'(a,f0.1,a,es14.7,a,es8.1,a,f6.4,a,f4.2,a,f0.3,a,i0)') &
            'test_ssh_se: C3 pi, tau = ', dt, ' s: omega_max = ', omega, ' 1/s (last change ', relchg, &
            '), a_crit(0.1) = ', acd, ', s = ', SE_CFL_SAFETY, ' -> dtbt <= ', lim, ' s, M_min = ', M_min

        ! ---- c) np > 1 against np 1 (the file next to the P4 reference) ----
        path = ''
        call get_environment_variable('FESOM3_TEST_SE_P4REF', path, status=ios)
        if (ios /= 0 .or. len_trim(path) == 0) then
            if (partit%mype == 0) write(*,'(a)') 'test_ssh_se: C3c np 1 reference skipped (FESOM3_TEST_SE_P4REF unset)'
        else if (partit%mype == 0) then
            path = trim(path)//'.c3'
            hdr  = [mesh%nod2D, mesh%elem2D]
            if (partit%npes == 1) then
                open(newunit=u_, file=trim(path), access='stream', form='unformatted', status='replace', &
                     action='write', iostat=ios)
                if (ios == 0) write(u_, iostat=ios) hdr, omega, M_min
                if (ios == 0) close(u_)
                call check(ios == 0, 'C3c np 1 writes the reference '//trim(path), partit%mype)
            else
                hdr_ref = -1; oref = -1.0_WP; mref = -1
                open(newunit=u_, file=trim(path), access='stream', form='unformatted', status='old', &
                     action='read', iostat=ios)
                if (ios == 0) read(u_, iostat=ios) hdr_ref, oref, mref
                if (ios == 0) close(u_)
                call check(ios == 0 .and. all(hdr_ref == hdr), 'C3c the np 1 reference '//trim(path)// &
                           ' exists and matches', partit%mype)
                call check(abs(omega - oref) <= 1.0e-12_WP*oref, 'C3c omega_max == np 1 to 1e-12', partit%mype)
                call check(M_min == mref, 'C3c M_min == np 1', partit%mype)
                write(*,'(a,i0,a,es10.3,a,i0,a,i0)') 'test_ssh_se: C3c np ', partit%npes, &
                    ' vs np 1: omega_max relative difference ', abs(omega - oref)/oref, ', M_min ', M_min, &
                    ' vs ', mref
            end if
        end if

        ! ---- e) the real substeps from the power-iteration vector ----
        vmax = maxval(abs(v(1:nNodO)))
        call max_all(vmax, partit)
        v = v/vmax
        do ic = 1, 3
            write(cs, '(f4.2)') CHIS(ic)
            Mc = se_cfl_m_min(dt, SE_CFL_SAFETY*ac(ic)/omega)
            call c3_run(dbase, v, dt/real(Mc, WP), Mc, 300*Mc, CHIS(ic), rl(1), re(1), rate(1))
            call c3_run(dbase, v, 0.97_WP*ac(ic)/omega, 10, 3000, CHIS(ic), rl(2), re(2), rate(2))
            call c3_run(dbase, v, 1.03_WP*ac(ic)/omega, 4, 3000, CHIS(ic), rl(3), re(3), rate(3))
            call c3_run(dbase, v, 1.1_WP*ac(ic)/omega, 4, 3000, CHIS(ic), rl(4), re(4), rate(4))
            rho = se_cfl_rho(1.1_WP*ac(ic), CHIS(ic))
            call check(rl(1) <= 1.0_WP, 'C3e chi = '//trim(cs)//': M = M_min does not grow (300 steps)', partit%mype)
            call check(rl(2) <= 1.0_WP, 'C3e chi = '//trim(cs)//': 0.97 a_crit does not grow', partit%mype)
            call check(re(3) > 1.0e6_WP, 'C3e chi = '//trim(cs)//': 1.03 a_crit grows', partit%mype)
            call check(re(4) > 1.0e6_WP, 'C3e chi = '//trim(cs)//': 1.1 a_crit grows', partit%mype)
            call check(abs(rate(4)/rho - 1.0_WP) <= 0.1_WP, &
                       'C3e chi = '//trim(cs)//': growth at 1.1 a_crit = se_cfl_rho within 10 %', partit%mype)
            if (partit%mype == 0) then
                write(*,'(a,a,a,i0,a,f6.4,a,es9.2,a,es9.2,a,f7.4)') 'test_ssh_se: C3e chi = ', trim(cs), &
                    ': M_min = ', Mc, ' (omega*dtbt/a_crit = ', omega*dt/real(Mc, WP)/ac(ic), &
                    '): ||eta|| last/0 = ', rl(1), '; 0.97 a_crit: ', rl(2), ', rate ', rate(2)
                write(*,'(a,a,a,es9.2,a,f7.4,a,es9.2,a,f7.4,a,f7.4)') 'test_ssh_se: C3e chi = ', trim(cs), &
                    ': 1.03 a_crit: end/0 = ', re(3), ', rate ', rate(3), '; 1.1 a_crit: end/0 = ', re(4), &
                    ', rate ', rate(4), ' (se_cfl_rho ', rho
            end if
        end do
        deallocate(v, u, w, Au, Aw, Uw, infl)
    end subroutine c3_cfl_guard

    subroutine c3_run(dbase, v, dtbt, ch, nsub, chi, rlast, rend, rate)
        ! C3e: the real se_subcycle from eta = v (all three ring levels), U = 0, F = 0, chi, at the
        ! substep dtbt, ch substeps per call; between the calls eta_n <- eta^M and U^n <- U^M, as a
        ! baroclinic step leaves them. At most nsub substeps, or until ||eta||_W exceeds 1e12 of
        ! the start. rlast = max ||eta||_W over the last 10 calls / the start, rend = the last /
        ! the start, rate = the growth per substep over the last 3 calls.
        type(t_dyn),   intent(in)  :: dbase
        real(kind=WP), intent(in)  :: v(:), dtbt, chi
        integer,       intent(in)  :: ch, nsub
        real(kind=WP), intent(out) :: rlast, rend, rate
        type(t_dyn) :: d
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, k, kmax, j
        real(kind=WP), allocatable :: hist(:)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        d = dbase
        d%se_chi = chi
        call se_sm_coeffs(chi, d%se_beta, d%se_zeta, d%se_gamma, d%se_delta)
        d%se_force = 0.0_WP; d%se_ubt = 0.0_WP; d%se_ubtn = 0.0_WP
        do n = 1, nNodL
            d%eta_n(n) = v(n); d%se_eta(n, 1) = v(n); d%se_eta(n, 2) = v(n)
        end do
        d%se_M = ch
        kmax = max(1, nsub/ch)
        allocate(hist(0:kmax))
        hist = 0.0_WP
        hist(0) = c3_wnorm(d%eta_n)
        do k = 1, kmax
            call se_subcycle(d, mesh, partit, real(ch, WP)*dtbt)
            d%eta_n(1:nNodL) = d%se_etam(1:nNodL)
            d%se_ubtn = d%se_ubt(:, :, 0)
            hist(k) = c3_wnorm(d%se_etam)
            if (.not. (hist(k) <= 1.0e12_WP*hist(0))) exit
        end do
        k = min(k, kmax)
        rlast = maxval(hist(max(1, k-9):k))/hist(0)
        rend  = hist(k)/hist(0)
        j     = min(3, k)
        rate  = (hist(k)/hist(k-j))**(1.0_WP/real(ch*j, WP))
        deallocate(hist)
    end subroutine c3_run

    real(kind=WP) function c3_wnorm(x)
        ! ||x||_W = sqrt(sum areasvol*x^2) over the owned nodes, reduced
        real(kind=WP), intent(in) :: x(:)
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, i
        real(kind=WP) :: s
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        s = 0.0_WP
        do i = 1, nNodO
            s = s + mesh%areasvol(i)*x(i)*x(i)
        end do
        call sum_all(s, partit)
        c3_wnorm = sqrt(s)
    end function c3_wnorm

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
                   .and. .not. dyn%se_m_force .and. dyn%se_bottdrag .and. .not. dyn%se_visc_flow, &
                   'C4 unset knobs keep the defaults', partit%mype)
        call se_sm_coeffs(0.1_WP, b, z, gm, d)
        call check(dyn%se_beta == b .and. dyn%se_zeta == z .and. dyn%se_gamma == gm .and. &
                   dyn%se_delta == d, 'C4 the coefficients in t_dyn', partit%mype)
        ! extents (Technical Details §1)
        call check(allocated(dyn%se_eta) .and. allocated(dyn%se_ubt) .and. allocated(dyn%se_sbar_old) &
                   .and. allocated(dyn%se_kdrag) .and. allocated(dyn%se_ubtn) .and. allocated(dyn%se_etam) &
                   .and. allocated(dyn%se_etanew) &
                   .and. allocated(dyn%se_umean) .and. allocated(dyn%se_force) .and. allocated(dyn%se_H) &
                   .and. allocated(dyn%se_c) .and. allocated(dyn%se_kappa), 'C4 the state is allocated', &
                   partit%mype)
        if (.not. (allocated(dyn%se_eta) .and. allocated(dyn%se_ubt))) return
        call check(all(shape(dyn%se_eta) == [nNodL, 2]) .and. size(dyn%se_etam) == nNodL .and. &
                   size(dyn%se_etanew) == nNodL, 'C4 node extents nNodL', partit%mype)
        call check(all(shape(dyn%se_ubt) == [2, nElemL, 3]) .and. lbound(dyn%se_ubt, 3) == 0 .and. &
                   all(shape(dyn%se_sbar_old) == [2, nElemL]) .and. size(dyn%se_kdrag) == nElemL &
                   .and. all(shape(dyn%se_ubtn) == [2, nElemL]) &
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
        call check(all(dyn%se_kdrag == 0.0_WP), 'C4 the drag rate is 0 until the first forcing', partit%mype)
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

    subroutine s_rk3_forcing()
        ! S1-S3 (header)
        type(t_dyn) :: dyn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, nb, nl, c, elnodes(MAX_NV)
        real(kind=WP) :: res, scale, pre(MAX_NV), Fx, Fy, sn(2), ex(2), ub(2), fr, kd, rk
        real(kind=WP), allocatable :: stress(:,:)
        nl = mesh%nl
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        allocate(stress(2, nElemF))
        ! ---- S1: pressure and Coriolis removed exactly ----
        call new_dyn(dyn)
        call set_uv(dyn, 401)
        do e = 1, nElemF
            dyn%uv(:, mesh%nlevels(e) - 1, e) = 0.0_WP          ! no bottom velocity: no drag
        end do
        do n = 1, nNodL
            dyn%eta_n(n) = 0.5_WP*hval(partit%myList_nod2D(n), 41)
        end do
        call se_init(dyn, mesh, partit, dt)
        allocate(dyn%rk3_visc(2, nl-1, nElemF)); dyn%rk3_visc = 0.0_WP
        stress = 0.0_WP
        call compute_vel_rhs(dyn, mesh, dt/3.0_WP, .true., partit, no_history=.true.)
        call se_bt_forcing_rk3(dyn, mesh, partit, dt, stress)
        res = 0.0_WP; scale = 0.0_WP
        do e = 1, nElemO
            elnodes = mesh%elem2D_nodes(:, e)
            pre = -(g*dyn%eta_n(elnodes))
            Fx  = sum(mesh%gradient_sca(1:4, e)*pre)
            Fy  = sum(mesh%gradient_sca(5:8, e)*pre)
            sn  = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                sn = sn + mesh%helem(nz, e)*dyn%uv(:, nz, e)
            end do
            scale = max(scale, dyn%se_H(e)*max(abs(Fx), abs(Fy)), abs(mesh%coriolis(e))*maxval(abs(sn)))
            res   = max(res, maxval(abs(dyn%se_force(:, e))))
        end do
        call max_all(res, partit); call max_all(scale, partit)
        call check(scale > 0.0_WP, 'S1 pressure and Coriolis are non-trivial', partit%mype)
        call check(res <= 1.0e-12_WP*scale, 'S1 RK3 forcing: F = 0 (pressure and Coriolis removed)', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3)') 'test_ssh_se: S1 max|F| = ', res, ', scale ', scale
        ! ---- S2: viscosity, wind and drag alone ----
        call set_uv(dyn, 403)
        dyn%eta_n = 0.0_WP
        do e = 1, nElemF
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                do c = 1, 2
                    dyn%rk3_visc(c, nz, e) = 1.0e-4_WP*hval(partit%myList_elem2D(e), 20*nz + c)
                end do
            end do
            stress(1, e) = 0.1_WP*hval(partit%myList_elem2D(e), 5)
            stress(2, e) = 0.1_WP*hval(partit%myList_elem2D(e), 6)
        end do
        call compute_vel_rhs(dyn, mesh, dt/3.0_WP, .true., partit, no_history=.true.)
        call se_bt_forcing_rk3(dyn, mesh, partit, dt, stress)
        res = 0.0_WP; scale = 0.0_WP; kd = 0.0_WP
        do e = 1, nElemO
            ex = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                ex = ex + mesh%helem(nz, e)*dyn%rk3_visc(:, nz, e)
            end do
            nb = mesh%nlevels(e) - 1
            ub = dyn%uv(:, nb, e)
            fr = C_d*sqrt(ub(1)**2 + ub(2)**2)
            ex = ex/dt + stress(:, e)/density_0 - fr*ub
            scale = max(scale, maxval(abs(ex)))
            res   = max(res, maxval(abs(dyn%se_force(:, e) - ex)))
            rk    = fr/dyn%se_H(e)
            kd    = max(kd, abs(dyn%se_kdrag(e) - rk)/max(rk, tiny(1.0_WP)))
        end do
        call max_all(res, partit); call max_all(scale, partit); call max_all(kd, partit)
        call check(scale > 0.0_WP, 'S2 viscosity, wind and drag are non-trivial', partit%mype)
        call check(res <= 1.0e-12_WP*scale, 'S2 RK3 forcing: F = visc/tau + wind - drag', partit%mype)
        call check(kd <= 1.0e-14_WP, 'S2 kdrag = C_d|u_b|/H', partit%mype)
        call check(all(dyn%se_ubtn(:, 1:nElemL) == dyn%se_ubt(:, 1:nElemL, 0)), 'S2 se_ubtn == ring level 0', &
                   partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3)') 'test_ssh_se: S2 max|F - expected| = ', res, &
            ', scale ', scale
        ! ---- S3: shared copies ----
        call check(shared_copy_max_elem(dyn%se_force, partit) == 0.0_WP, 'S3 F copies agree on shared elements', &
                   partit%mype)
        call check(shared_copy_max_elem(dyn%se_kdrag, partit) == 0.0_WP, 'S3 kdrag copies agree', partit%mype)
    end subroutine s_rk3_forcing

    subroutine c6_forcing_coriolis()
        type(t_dyn) :: dyn, dwrong, doff
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e, nz, nbad, nb, nfoff
        real(kind=WP) :: sn(2), f, res1, resw, fw, res2, scale1, scale2, aw, kd, kerr, kmax, koff
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(dyn)
        call set_uv(dyn, 0)                              ! uv^{n-3/2}
        call se_init(dyn, mesh, partit, dt)              ! s_old = 0
        call check(all(dyn%se_sbar_old == 0.0_WP), 'C6 s_old == 0 after se_init', partit%mype)

        ! ---- step 1, lfirst ----
        call compute_vel_rhs(dyn, mesh, dt, .true., partit)
        dwrong = dyn
        doff   = dyn
        doff%se_bottdrag = .false.
        call se_bt_forcing(dwrong, mesh, partit, dt, .false.)   ! a2 = 1.5+eps on an lfirst UV_rhs
        call se_bt_forcing(doff,   mesh, partit, dt, .true.)    ! the drag off
        call se_bt_forcing(dyn,    mesh, partit, dt, .true.)
        aw = 0.5_WP + ab_epsilon                         ! (1.5+eps) - 1
        res1 = 0.0_WP; resw = 0.0_WP; fw = 0.0_WP; scale1 = 0.0_WP; nbad = 0
        kerr = 0.0_WP; kmax = 0.0_WP
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
            ! the drag rate (Task 5.4) against an independent recomputation
            nb = mesh%nlevels(e) - 1
            kd = C_d*sqrt(dyn%uv(1, nb, e)**2 + dyn%uv(2, nb, e)**2)/dyn%se_H(e)
            kerr = max(kerr, abs(dyn%se_kdrag(e) - kd)/max(kd, tiny(1.0_WP)))
            kmax = max(kmax, dyn%se_kdrag(e))
            if (.not. (dyn%se_kdrag(e) >= 0.0_WP .and. dyn%se_kdrag(e) <= huge(1.0_WP))) nbad = nbad + 1
            if (any(dyn%uv(:, nb, e) /= 0.0_WP) .and. .not. (dyn%se_kdrag(e) > 0.0_WP)) nbad = nbad + 1
        end do
        koff  = maxval(abs(doff%se_kdrag))
        nfoff = count(doff%se_force(:, 1:nElemL) /= dyn%se_force(:, 1:nElemL))
        call max_all(res1, partit); call max_all(resw, partit); call max_all(fw, partit)
        call max_all(scale1, partit); call sum_all(nbad, partit)
        call max_all(kerr, partit); call max_all(kmax, partit); call max_all(koff, partit)
        call sum_all(nfoff, partit)
        call check(scale1 > 0.0_WP, 'C6 the Coriolis term is non-trivial', partit%mype)
        call check(res1 <= 1.0e-12_WP*scale1, 'C6 lfirst: F = 0 (a2 = 1)', partit%mype)
        call check(resw <= 1.0e-12_WP*scale1 .and. fw >= 0.5_WP*aw*scale1, &
                   'C6 lfirst UV_rhs with a2 = 1.5+eps: F = (0.5+eps) f e_z x s^n', partit%mype)
        call check(nbad == 0, 'C6 s_old == sum_k helem*uv after the step (bitwise); kdrag finite, '// &
                   '> 0 where u_bottom /= 0', partit%mype)
        call check(kerr <= 1.0e-14_WP .and. kmax > 0.0_WP, 'C6 kdrag = C_d |u_bottom|/H^n', partit%mype)
        call check(koff == 0.0_WP .and. nfoff == 0, &
                   'C6 FESOM3_SE_BOTTDRAG off: kdrag == 0, F bitwise the same', partit%mype)

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
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,es10.3)') &
            'test_ssh_se: C6 Coriolis part: lfirst max|F| = ', res1, ', AB2 step max|F| = ', res2, &
            ' m2/s2, max|f s| = ', max(scale1, scale2), '; wrong-weight residual ', resw, &
            '; kdrag max ', kmax, ' 1/s, rel. error ', kerr
    end subroutine c6_forcing_coriolis

    subroutine c7_consistency()
        type(t_dyn) :: dyn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz
        integer :: elnodes(MAX_NV)
        real(kind=WP), allocatable :: ustar(:,:), dU(:,:), tend(:,:), HA(:)
        real(kind=WP) :: gb(2), res, scale, pre(MAX_NV), Fx, Fy, kmax     ! gb: G (g is gravity)
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
        kmax = maxval(dyn%se_kdrag(1:nElemO))
        call max_all(kmax, partit)

        allocate(ustar(2, nElemL), dU(2, nElemL), tend(2, nElemL), HA(nElemL))
        ustar = dyn%se_sbar_old                                   ! = s^n
        dU    = 0.0_WP                                            ! and U^m = U^n (ring level 0 = se_ubtn): D = 0
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
        call check(kmax > 0.0_WP, 'C7 the drag is on (kdrag > 0)', partit%mype)
        call check(res <= 1.0e-12_WP*scale, 'C7 se_tendency(U* = s^n = s_old, eta^A = eta^n, H^A = H^n, '// &
                   'dU = 0, U^m = U^n) == G/tau', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3)') &
            'test_ssh_se: C7 consistency: max|tend - G/tau| = ', res, ' m2/s2, largest term ', scale, &
            ', ratio ', res/scale
        deallocate(ustar, dU, tend, HA)
    end subroutine c7_consistency

    subroutine c12_bottdrag()
        ! C12 (header): the barotropic bottom drag D(U^m - U^n) inside se_tendency, and a full substep cycle
        type(t_dyn) :: d, doff, dc, dcoff
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e, c, n, nbad
        real(kind=WP), allocatable :: ustar(:,:), dU(:,:), HA(:), eta0(:), t0(:,:), t1(:,:), t2(:,:), t3(:,:), toff(:,:)
        real(kind=WP), allocatable :: X(:,:), Y(:,:), infl(:)
        real(kind=WP) :: r1, r2, r3, sc, kmax, dmax, umax, res, escale
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        allocate(ustar(2, nElemL), dU(2, nElemL), HA(nElemL), eta0(nNodL), infl(nNodL))
        allocate(t0(2, nElemL), t1(2, nElemL), t2(2, nElemL), t3(2, nElemL), toff(2, nElemL))
        allocate(X(2, nElemL), Y(2, nElemL))
        call new_dyn(d)
        call set_uv(d, 3)
        call se_init(d, mesh, partit, dt)
        d%se_visc_ah = 0.0_WP                            ! D alone
        call se_bt_forcing(d, mesh, partit, dt, .true.)  ! kdrag from the hashed uv; ubtn = ring level 0
        d%se_force = 0.0_WP
        ustar = 0.0_WP; dU = 0.0_WP; eta0 = 0.0_WP
        HA = d%se_H
        do e = 1, nElemL
            do c = 1, 2
                X(c, e) = d%se_H(e)*0.1_WP*hval(partit%myList_elem2D(e), 50 + c)
                Y(c, e) = d%se_H(e)*0.1_WP*hval(partit%myList_elem2D(e), 52 + c)
            end do
        end do
        doff = d
        doff%se_bottdrag = .false.
        call se_tendency(d,    mesh, partit, ustar, eta0, HA, dU, t0)
        call se_tendency(doff, mesh, partit, ustar, eta0, HA, dU, toff)
        nbad = count(t0(:, 1:nElemO) /= toff(:, 1:nElemO))
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C12 U^m = U^n: the tendency is bitwise that of the drag off', partit%mype)
        d%se_ubt(:, :, 0) = d%se_ubtn + X
        call se_tendency(d, mesh, partit, ustar, eta0, HA, dU, t1)
        d%se_ubt(:, :, 0) = d%se_ubtn + 2.0_WP*X
        call se_tendency(d, mesh, partit, ustar, eta0, HA, dU, t2)
        d%se_ubt(:, :, 0) = d%se_ubtn + (X + Y)
        call se_tendency(d, mesh, partit, ustar, eta0, HA, dU, t3)
        d%se_ubt(:, :, 0) = d%se_ubtn
        r1 = 0.0_WP; r2 = 0.0_WP; r3 = 0.0_WP; sc = 0.0_WP; kmax = 0.0_WP
        do e = 1, nElemO
            sc   = max(sc, d%se_kdrag(e)*maxval(abs(X(:, e)) + abs(Y(:, e))))
            kmax = max(kmax, d%se_kdrag(e))
            r1 = max(r1, maxval(abs((t1(:, e) - t0(:, e)) + d%se_kdrag(e)*X(:, e))))
            r2 = max(r2, maxval(abs((t2(:, e) - t0(:, e)) - 2.0_WP*(t1(:, e) - t0(:, e)))))
            r3 = max(r3, maxval(abs((t3(:, e) - t0(:, e)) + d%se_kdrag(e)*(X(:, e) + Y(:, e)))))
        end do
        call max_all(r1, partit); call max_all(r2, partit); call max_all(r3, partit)
        call max_all(sc, partit); call max_all(kmax, partit)
        call check(sc > 0.0_WP .and. r1 <= 1.0e-14_WP*sc, 'C12 D(X) = -kdrag*X', partit%mype)
        call check(r2 <= 1.0e-14_WP*sc, 'C12 D(2X) = 2 D(X) (linear)', partit%mype)
        call check(r3 <= 1.0e-14_WP*sc, 'C12 D(X + Y) = -kdrag*(X + Y) (additive)', partit%mype)

        ! ---- a full M = 40 cycle from a hashed state, the drag on and off ----
        dc = d
        dc%se_visc_ah = 0.01_WP
        do n = 1, nNodL
            dc%eta_n(n) = 0.1_WP*hval(partit%myList_nod2D(n), 54)
        end do
        dc%se_eta(:, 1) = dc%eta_n(1:nNodL); dc%se_eta(:, 2) = dc%eta_n(1:nNodL)
        do e = 1, nElemL
            do c = 1, 2
                dc%se_ubt(c, e, :) = d%se_H(e)*0.05_WP*hval(partit%myList_elem2D(e), 56 + c)
                dc%se_force(c, e)  = 1.0e-3_WP*hval(partit%myList_elem2D(e), 58 + c)
            end do
            dc%se_ubtn(:, e) = dc%se_ubt(:, e, 0)
        end do
        dc%se_M = 40
        dcoff = dc
        dcoff%se_bottdrag = .false.
        call se_subcycle(dc,    mesh, partit, dt)
        call se_subcycle(dcoff, mesh, partit, dt)
        nbad = 0
        do e = 1, nElemO
            if (.not. all(abs(dc%se_ubt(:, e, 0)) <= huge(1.0_WP))) nbad = nbad + 1
        end do
        do n = 1, nNodO
            if (.not. (abs(dc%se_etam(n)) <= huge(1.0_WP))) nbad = nbad + 1
        end do
        dmax = maxval(abs(dc%se_ubt(:, 1:nElemO, 0) - dcoff%se_ubt(:, 1:nElemO, 0)))
        umax = maxval(abs(dcoff%se_ubt(:, 1:nElemO, 0)))
        call se_inflow(mesh, partit, dc%se_umean, infl)
        res = 0.0_WP; escale = 0.0_WP
        do n = 1, nNodO
            res    = max(res, abs(dc%se_etam(n) - (dc%eta_n(n) + dt*infl(n)/mesh%areasvol(n))))
            escale = max(escale, abs(dc%eta_n(n)), abs(dc%se_etam(n) - dc%eta_n(n)))
        end do
        call sum_all(nbad, partit); call max_all(dmax, partit); call max_all(umax, partit)
        call max_all(res, partit); call max_all(escale, partit)
        call check(nbad == 0 .and. dmax > 0.0_WP, 'C12 M = 40 with the drag: finite, differs from the drag off', &
                   partit%mype)
        call check(res <= 1.0e-12_WP*escale, 'C12 M = 40 with the drag: eta^M = eta^0 + tau*I(<<U>>)/areasvol', &
                   partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,3es10.3,a,es10.3,a,es10.3,a,es10.3)') &
            'test_ssh_se: C12 bottom drag: max kdrag ', kmax, ' 1/s (dtbt*kdrag at M = 50: ', kmax*dt/50.0_WP, &
            '); residuals D(X), D(2X), D(X+Y) ', r1, r2, r3, ' of ', sc, ' m2/s2; M = 40: max|U(drag) - U(no drag)| ', &
            dmax, ' of ', umax
        deallocate(ustar, dU, HA, eta0, infl, t0, t1, t2, t3, toff, X, Y)
    end subroutine c12_bottdrag

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

    subroutine c8f_visc_flow()
        ! C8f (header): the flow-dependent viscosity W of se_visc_flow
        type(t_dyn) :: dyn, d0, d, dn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e, k, kb, nb, c, ed, el(2)
        integer :: nbad, nbad_zero, nbad_sym, ncmp, nlo, nhi, ip
        real(kind=WP), allocatable :: U(:,:), U2(:,:), W(:,:), W2(:,:), W3(:,:), acc(:,:), aacc(:,:)
        real(kind=WP) :: ssum(2), sabs(2), res, scale, du(2), hh, nu, x, sp, wmax, zerr, gam1(2)
        logical :: allown
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(dyn)
        call se_init(dyn, mesh, partit, dt)
        call se_elem_depth(dyn, mesh, partit)
        call check(.not. dyn%se_visc_flow, 'C8f the flow-dependent viscosity is off by default', partit%mype)
        dyn%se_visc_flow = .true.

        ! ---- zlen: structure, value, symmetry ----
        nbad_zero = 0; nbad = 0; nbad_sym = 0; zerr = 0.0_WP
        do e = 1, nElemO
            do k = 1, MAX_NV
                nb = mesh%elem_neighbors(k, e)
                if (k > mesh%elem2D_nnodes(e) .or. nb == e) then
                    if (dyn%se_zlen(k, e) /= 0.0_WP) nbad_zero = nbad_zero + 1
                    cycle
                end if
                x = sqrt(mesh%elem_area(e) + mesh%elem_area(nb))
                zerr = max(zerr, abs(dyn%se_zlen(k, e) - x)/x)
                if (nb > nElemO) cycle
                do kb = 1, mesh%elem2D_nnodes(nb)
                    if (mesh%elem_neighbors(kb, nb) /= e) cycle
                    if (dyn%se_zlen(kb, nb) /= dyn%se_zlen(k, e)) nbad_sym = nbad_sym + 1
                    nbad = nbad + 1
                end do
            end do
        end do
        call sum_all(nbad_zero, partit); call sum_all(nbad_sym, partit); call sum_all(nbad, partit)
        call max_all(zerr, partit)
        call check(nbad_zero == 0, 'C8f zlen == 0 on boundary slots and triangle slot 4', partit%mype)
        call check(zerr <= 1.0e-15_WP, 'C8f zlen = sqrt(A_e + A_nb) on interior slots', partit%mype)
        call check(nbad > 0 .and. nbad_sym == 0, 'C8f zlen bitwise the same from both sides of an edge', &
                   partit%mype)

        ! ---- a hashed halo-valid transport, velocities ~5 cm/s ----
        allocate(U(2, nElemL), U2(2, nElemL), W(2, nElemL), W2(2, nElemL), W3(2, nElemL))
        allocate(acc(2, nElemL), aacc(2, nElemL))
        do e = 1, nElemL
            do c = 1, 2
                U(c, e)  = dyn%se_H(e)*0.05_WP*hval(partit%myList_elem2D(e), 70 + c)
                U2(c, e) = dyn%se_H(e)*0.05_WP*hval(partit%myList_elem2D(e), 72 + c)
            end do
        end do
        W = 0.0_WP; W2 = 0.0_WP; W3 = 0.0_WP

        ! both branches of max(gamma0, gamma1*|dU|/H) occur
        nlo = 0; nhi = 0
        do e = 1, nElemO
            do k = 1, mesh%elem2D_nnodes(e)
                nb = mesh%elem_neighbors(k, e)
                if (nb == e) cycle
                du = U(:, e) - U(:, nb)
                hh = 0.5_WP*(dyn%se_H(e) + dyn%se_H(nb))
                if (dyn%se_visc_g1*sqrt(du(1)**2 + du(2)**2)/hh > dyn%se_visc_g0) then
                    nhi = nhi + 1
                else
                    nlo = nlo + 1
                end if
            end do
        end do
        call sum_all(nlo, partit); call sum_all(nhi, partit)
        call check(nlo > 0 .and. nhi > 0, 'C8f both branches of max(gamma0, gamma1*|dU|/H) occur', partit%mype)

        ! ---- sum_e A_e W_e = 0 ----
        call se_visc_flow(dyn, mesh, partit, U, W)
        ssum = 0.0_WP; sabs = 0.0_WP
        do e = 1, nElemO
            if (mesh%elem2D_nodes(1, e) > nNodO) cycle        ! counted on its first vertex's owner
            ssum = ssum + mesh%elem_area(e)*W(:, e)
            sabs = sabs + mesh%elem_area(e)*abs(W(:, e))
        end do
        call sum_all(ssum, partit); call sum_all(sabs, partit)
        call check(all(sabs > 0.0_WP), 'C8f W of a hashed U is non-trivial', partit%mype)
        call check(all(abs(ssum) <= 1.0e-12_WP*sabs), 'C8f sum_e A_e*W_e = 0', partit%mype)
        if (partit%mype == 0) write(*,'(a,2es10.3,a,2es10.3,a,i0,a,i0,a)') &
            'test_ssh_se: C8f flow viscosity: sum_e A_e W_e = ', ssum, ' of sum_e A_e |W_e| = ', sabs, &
            ' (slots: ', nlo, ' at gamma0, ', nhi, ' at gamma1*|dU|/H)'

        ! ---- against the edge scatter of the Zenodo code, gamma1 = 2750 and 0 ----
        gam1 = [dyn%se_visc_g1, 0.0_WP]
        do ip = 1, 2
            dyn%se_visc_g1 = gam1(ip)
            call se_visc_flow(dyn, mesh, partit, U, W)
            acc = 0.0_WP; aacc = 0.0_WP
            do ed = 1, nEdgeO
                el = mesh%edge_tri(:, ed)
                if (el(2) == el(1)) cycle                          ! boundary edge
                hh = 0.5_WP*(dyn%se_H(el(1)) + dyn%se_H(el(2)))
                du = U(:, el(1)) - U(:, el(2))
                nu = sqrt(max(dyn%se_visc_g0, dyn%se_visc_g1*sqrt(du(1)**2 + du(2)**2)/hh) &
                          *sqrt(mesh%elem_area(el(1)) + mesh%elem_area(el(2))))
                acc(:, el(1))  = acc(:, el(1)) - du*nu
                acc(:, el(2))  = acc(:, el(2)) + du*nu
                aacc(:, el(1)) = aacc(:, el(1)) + abs(du*nu)
                aacc(:, el(2)) = aacc(:, el(2)) + abs(du*nu)
            end do
            res = 0.0_WP; scale = 0.0_WP; ncmp = 0
            do e = 1, nElemO
                allown = .true.
                do k = 1, mesh%elem2D_nnodes(e)
                    if (mesh%elem_edges(k, e) > nEdgeO) allown = .false.
                end do
                if (.not. allown) cycle
                ncmp  = ncmp + 1
                res   = max(res, maxval(abs(W(:, e) - acc(:, e)/mesh%elem_area(e))))
                scale = max(scale, maxval(aacc(:, e))/mesh%elem_area(e))
            end do
            call max_all(res, partit); call max_all(scale, partit); call sum_all(ncmp, partit)
            call check(ncmp > 0 .and. res <= 1.0e-13_WP*scale, 'C8f W = the Zenodo edge scatter ('// &
                       trim(merge('gamma1 = 2750', 'gamma1 = 0   ', ip == 1))//')', partit%mype)
            if (partit%mype == 0) write(*,'(a,f6.0,a,es10.3,a,es10.3,a,i0,a)') &
                'test_ssh_se: C8f gamma1 = ', gam1(ip), ': W vs the edge scatter ', res, ' of ', scale, &
                ' (', ncmp, ' elements)'
        end do

        ! ---- gamma1 = 0: linear in U ----
        ! (dyn%se_visc_g1 = 0 from the loop above)
        call se_visc_flow(dyn, mesh, partit, U, W)
        call se_visc_flow(dyn, mesh, partit, 2.0_WP*U, W2)
        nbad = count(W2(:, 1:nElemO) /= 2.0_WP*W(:, 1:nElemO))
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C8f gamma1 = 0: W(2U) == 2 W(U) (bitwise)', partit%mype)
        call se_visc_flow(dyn, mesh, partit, U2, W2)
        call se_visc_flow(dyn, mesh, partit, U + U2, W3)
        res = maxval(abs(W3(:, 1:nElemO) - W(:, 1:nElemO) - W2(:, 1:nElemO)))
        scale = max(maxval(abs(W(:, 1:nElemO))), maxval(abs(W2(:, 1:nElemO))))
        call max_all(res, partit); call max_all(scale, partit)
        call check(res <= 1.0e-13_WP*scale, 'C8f gamma1 = 0: W(U + U2) = W(U) + W(U2)', partit%mype)
        sp = 0.0_WP
        if (partit%npes > 1) sp = shared_copy_max_elem(W, partit)

        ! ---- gamma0 = 0: the flow term alone, W(4U) = 8 W(U) ----
        dyn%se_visc_g0 = 0.0_WP
        dyn%se_visc_g1 = gam1(1)
        call se_visc_flow(dyn, mesh, partit, U, W)
        call se_visc_flow(dyn, mesh, partit, 4.0_WP*U, W2)
        res = maxval(abs(W2(:, 1:nElemO) - 8.0_WP*W(:, 1:nElemO)))
        scale = 8.0_WP*maxval(abs(W(:, 1:nElemO)))
        call max_all(res, partit); call max_all(scale, partit)
        call check(scale > 0.0_WP .and. res <= 1.0e-13_WP*scale, 'C8f gamma0 = 0: W(4U) = 8 W(U)', partit%mype)
        if (partit%npes > 1) sp = max(sp, shared_copy_max_elem(W, partit))

        ! ---- the defaults again: copy spread, uniform transport ----
        dyn%se_visc_g0 = 10.0_WP
        call se_visc_flow(dyn, mesh, partit, U, W)
        if (partit%npes > 1) then
            sp = max(sp, shared_copy_max_elem(W, partit))
            call max_all(sp, partit)
            call check(sp == 0.0_WP, 'C8f copy spread of W over the shared elements is 0', partit%mype)
            if (partit%mype == 0) write(*,'(a,es10.3)') 'test_ssh_se: C8f max copy spread of W ', sp
        end if
        wmax = maxval(abs(W(:, 1:nElemO)))
        call max_all(wmax, partit)
        do e = 1, nElemL
            U(:, e) = [1200.0_WP, -800.0_WP]
        end do
        call se_visc_flow(dyn, mesh, partit, U, W)
        nbad = count(W(:, 1:nElemO) /= 0.0_WP)
        call sum_all(nbad, partit)
        call check(nbad == 0, 'C8f W(uniform transport) == 0 exactly', partit%mype)

        ! ---- one substep from U^0 = U^n: W(U^0) - W(U^n) == 0 ----
        call new_dyn(d0)
        call se_init(d0, mesh, partit, dt)
        call se_elem_depth(d0, mesh, partit)
        do e = 1, nElemL
            do c = 1, 2
                d0%se_ubt(c, e, 0) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 80 + c)
                d0%se_ubt(c, e, 1) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 82 + c)
                d0%se_ubt(c, e, 2) = d0%se_H(e)*0.01_WP*hval(partit%myList_elem2D(e), 84 + c)
                d0%se_force(c, e)  = 1.0e-3_WP*hval(partit%myList_elem2D(e), 86 + c)
            end do
            d0%se_ubtn(:, e) = d0%se_ubt(:, e, 0)
        end do
        d0%se_M = 1
        d = d0
        d%se_visc_flow = .true.
        dn = d0
        dn%se_visc_ah = 0.0_WP
        call se_subcycle(d, mesh, partit, dt)
        call se_subcycle(dn, mesh, partit, dt)
        nbad = count(d%se_ubt(:, 1:nElemO, 0) /= dn%se_ubt(:, 1:nElemO, 0))
        x = maxval(abs(d%se_vn(:, 1:nElemO)))
        call sum_all(nbad, partit); call max_all(x, partit)
        call check(nbad == 0 .and. x > 0.0_WP, 'C8f M = 1 from U^0 = U^n: U^1 equals the run '// &
                   'without viscosity (W(U^0) - W(U^n) == 0, W(U^n) non-trivial)', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a)') 'test_ssh_se: C8f max|W| (defaults, hashed U) ', &
            wmax, ' m2/s2; max|W(U^n)| of the substep test ', x, ' m2/s2'
        deallocate(U, U2, W, W2, W3, acc, aacc)
    end subroutine c8f_visc_flow

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
            d0%se_kdrag(e) = 1.0e-4_WP*abs(hval(partit%myList_elem2D(e), 28))   ! the drag acts in c)
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

    !===========================================================================
    ! ---- Stage 2 (plan Task 2.1): split-explicit on many ranks, P1-P4 ----
    ! The state at the end of a real predictor: uv (0.02 m/s), eta_n (0.1 m) and the baroclinic
    ! pgf hashed from the global ids (every copy of a node or element holds the same value),
    ! then every step the real compute_vel_rhs (momentum advection off) and viscosity_filter
    ! (opt_visc 7, the drivers' coefficients). The filter scatters its edge fluxes into a shared
    ! border element in each rank's own edge order, so the owners' copies of UV_rhs differ by
    ! roundoff, and with them G = sum_k helem*UV_rhs and F: the duplicated state of LESSONS L70.
    ! Plan D2 syncs F once per step (the last statement of se_bt_forcing); every substep operator
    ! is a gather, so U and <<U>> then stay bitwise equal on all copies without a per-substep sync.
    ! P1 and P2 run together: NK = 6 SE steps through the real kernels (compute_vel_rhs,
    ! viscosity_filter, se_bt_forcing, se_subcycle with M = 40, se_trim_finish; step_oce's order
    ! without the implicit vertical viscosity and the tail), the shared-element sync on.
    !   P1  the F sync, every step: a copy of the state runs se_bt_forcing with
    !       partit%sync_shared_elem off; its copy spread of F is > 0 in at least one step at
    !       np 8 (the roundoff exists; the UV_rhs spread is printed; at np 2 printed only: there
    !       the Intel build's UV_rhs roundoff is absorbed in G in every step). The real call (sync on)
    !       leaves the spread exactly 0, and every slot that is not a receive slot of
    !       com_elem2D_shared holds bitwise the unsynced value (the designated copies are kept).
    !       At np 1 both spreads are 0. The bottom-drag rate kdrag (on by default, plan Task 5.4)
    !       is synced with F: its copy spread is 0 after every real call, and it is > 0.
    !   P2  after every step the copy spreads of the three U ring levels, <<U>>, uv and uv_rhsAB
    !       are exactly 0, and the inflow-sum ratios |sum I(X)| / sum|I(X)| over the owned nodes
    !       (reduced; the divsum ratio of FESOM3_STEP_DIAG and fesom_conserve) of X = <<U>>
    !       (= ssh_rhs_old) and X = U^M are <= 1e-14. Every value finite; eta and U non-trivial.
    !   P3  (npes > 1) the same NK steps with the shared-element sync off (F and uv): the copy
    !       spreads of U^M and <<U>> are > 0 at np 8 (printed at np 2), so P2's zeros are the
    !       sync's doing, not a lack of shared elements or of roundoff. The ratios are printed.
    !   P4  partition independence: eta and U^M after P2's NK steps, by global id. With
    !       FESOM3_TEST_SE_P4REF set (ctest), np 1 writes them to that file (fixture se_p4_ref)
    !       and np 2 / np 8 compare: max|eta - eta(np 1)| <= 1e-12 max|eta|, the same for U; the
    !       levels are printed. Without the variable P4 is skipped (one line).
    subroutine p_many_ranks()
        integer, parameter :: NK = 6       ! the SE steps of P1-P4
        type(t_dyn) :: ds, doff, d, dp2
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e, n, nz, k, l
        integer :: nbad, nkeep, ndiff, nrecv, npos
        logical :: sync_env
        logical, allocatable :: is_recv(:)
        real(kind=WP), allocatable :: infl(:), wf(:)
        real(kind=WP) :: x, sp_off, sp_on, sp_rhs, fmax, sp(4), spmax(4), r(2), rmax(2), mxe, mxu, kdmax
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        sync_env = partit%sync_shared_elem
        kdmax = 0.0_WP
        allocate(is_recv(nElemL), infl(nNodL), wf(nNodL))
        wf = 0.0_WP
        is_recv = .false.
        if (allocated(partit%com_elem2D_shared%rlist)) then
            do l = 1, size(partit%com_elem2D_shared%rlist)
                is_recv(partit%com_elem2D_shared%rlist(l)) = .true.
            end do
        end if
        nrecv = count(is_recv)
        call sum_all(nrecv, partit)

        ! ---- P1 + P2: NK SE steps, sync on ----
        call p_state(ds)
        d = ds
        sp_off = 0.0_WP; sp_on = 0.0_WP; sp_rhs = 0.0_WP; fmax = 0.0_WP
        spmax = 0.0_WP; rmax = 0.0_WP; nbad = 0; nkeep = 0; ndiff = 0; npos = 0
        do k = 1, NK
            call compute_vel_rhs(d, mesh, dt, k == 1, partit)
            call viscosity_filter(7, d, mesh, dt, partit)
            x = shared_copy_max_elem(d%uv_rhs, partit)
            call max_all(x, partit)
            sp_rhs = max(sp_rhs, x)
            ! P1: the forcing without the sync (a copy), then the real call
            doff = d
            partit%sync_shared_elem = .false.
            call se_bt_forcing(doff, mesh, partit, dt, k == 1)
            partit%sync_shared_elem = .true.
            call se_bt_forcing(d, mesh, partit, dt, k == 1)
            partit%sync_shared_elem = sync_env
            x = shared_copy_max_elem(doff%se_force, partit)
            call max_all(x, partit)
            sp_off = max(sp_off, x)
            if (x > 0.0_WP) npos = npos + 1
            x = shared_copy_max_elem(d%se_force, partit)
            call max_all(x, partit)
            sp_on = max(sp_on, x)
            x = shared_copy_max_elem(d%se_kdrag, partit)
            call max_all(x, partit)
            sp_on = max(sp_on, x)
            kdmax = max(kdmax, maxval(d%se_kdrag(1:nElemO)))
            do e = 1, nElemO
                fmax = max(fmax, maxval(abs(d%se_force(:, e))))
                if (.not. any(d%se_force(:, e) /= doff%se_force(:, e))) cycle
                if (is_recv(e)) then
                    ndiff = ndiff + 1
                else
                    nkeep = nkeep + 1
                end if
            end do
            call se_subcycle(d, mesh, partit, dt)
            call se_trim_finish(d, mesh, partit, dt, k, wf)
            ! P2: the copy spreads and the inflow-sum ratios
            sp(1) = max(shared_copy_max_elem(d%se_ubt(:, :, 0), partit), &
                        shared_copy_max_elem(d%se_ubt(:, :, 1), partit), &
                        shared_copy_max_elem(d%se_ubt(:, :, 2), partit))
            sp(2) = shared_copy_max_elem(d%se_umean, partit)
            sp(3) = shared_copy_max_elem(d%uv, partit)
            sp(4) = shared_copy_max_elem(d%uv_rhsAB(1, :, :, :), partit)
            do l = 1, 4
                call max_all(sp(l), partit)
            end do
            spmax = max(spmax, sp)
            call se_inflow(mesh, partit, d%se_ubt(:, :, 0), infl)
            r(1) = p_ratio(d%ssh_rhs_old(1:nNodO))
            r(2) = p_ratio(infl(1:nNodO))
            rmax = max(rmax, r)
            do n = 1, nNodO
                if (.not. (abs(d%eta_n(n)) <= huge(1.0_WP))) nbad = nbad + 1
            end do
            do e = 1, nElemO
                if (.not. all(abs(d%se_ubt(:, e, 0)) <= huge(1.0_WP))) nbad = nbad + 1
                do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                    if (.not. all(abs(d%uv(:, nz, e)) <= huge(1.0_WP))) nbad = nbad + 1
                end do
            end do
        end do
        call max_all(fmax, partit); call max_all(kdmax, partit)
        call sum_all(nkeep, partit); call sum_all(ndiff, partit); call sum_all(nbad, partit)

        call check(fmax > 0.0_WP, 'P1 F is non-trivial', partit%mype)
        call check(kdmax > 0.0_WP, 'P1 the bottom drag is on (kdrag > 0)', partit%mype)
        call check(sp_on == 0.0_WP, 'P1 the copy spreads of F and kdrag after the sync are 0 (every step)', &
                   partit%mype)
        call check(nkeep == 0, 'P1 the sync keeps every slot that is not a receive slot (bitwise)', partit%mype)
        if (partit%npes > 1) then
            call check(nrecv > 0, 'P1 the partition has shared elements', partit%mype)
            ! required at np >= 8 only: at pi np 2 (35 receive slots) the Intel build's UV_rhs roundoff
            ! is absorbed in the layer sum G in all NK steps (measured, plan Task 2.1)
            if (partit%npes >= 8) call check(sp_off > 0.0_WP, 'P1 the copy spread of F before the sync '// &
                                             'is > 0 (roundoff of R)', partit%mype)
        else
            call check(sp_off == 0.0_WP .and. nrecv == 0, 'P1 np 1: no shared copies, spread 0', partit%mype)
        end if
        if (partit%mype == 0) write(*,'(a,i0,a,es10.3,a,i0,a,es10.3,a,es10.3,a,es10.3,a,i0,a,i0,a)') &
            'test_ssh_se: P1 ', NK, ' steps: max F copy spread before the sync ', sp_off, ' m2/s2 (> 0 in ', &
            npos, ' steps; max|F| ', fmax, ', max UV_rhs spread ', sp_rhs, ' m/s), after ', sp_on, '; ', &
            ndiff, ' receive slots changed (', nrecv, ' per step)'

        mxe = maxval(abs(d%eta_n(1:nNodO)))
        mxu = maxval(abs(d%se_ubt(:, 1:nElemO, 0)))
        call max_all(mxe, partit); call max_all(mxu, partit)
        call check(nbad == 0 .and. mxe > 0.0_WP .and. mxu > 0.0_WP, 'P2 eta, U, uv finite and non-trivial', &
                   partit%mype)
        call check(all(spmax == 0.0_WP), 'P2 copy spreads of U (3 levels), <<U>>, uv, uv_rhsAB are 0 '// &
                   'after every step', partit%mype)
        call check(all(rmax <= 1.0e-14_WP), 'P2 inflow-sum ratios of <<U>> and U^M <= 1e-14', partit%mype)
        if (partit%mype == 0) write(*,'(a,i0,a,4es10.3,a,2es10.3,a,es10.3,a,es10.3,a)') &
            'test_ssh_se: P2 ', NK, ' SE steps, sync on: max spreads U/<<U>>/uv/uv_rhsAB', spmax, &
            '; max inflow-sum ratio <<U>>, U^M', rmax, '; max|eta| ', mxe, ' m, max|U| ', mxu, ' m2/s'
        dp2 = d

        ! ---- P3: the same steps without the shared-element sync ----
        if (partit%npes > 1) then
            d = ds
            partit%sync_shared_elem = .false.
            do k = 1, NK
                call p_step(d, k)
            end do
            partit%sync_shared_elem = sync_env
            sp(1) = shared_copy_max_elem(d%se_ubt(:, :, 0), partit)
            sp(2) = shared_copy_max_elem(d%se_umean, partit)
            call max_all(sp(1), partit); call max_all(sp(2), partit)
            call se_inflow(mesh, partit, d%se_ubt(:, :, 0), infl)
            r(1) = p_ratio(d%ssh_rhs_old(1:nNodO))
            r(2) = p_ratio(infl(1:nNodO))
            ! required at np >= 8 only (as P1: at pi np 2 the roundoff of F can be absorbed in U)
            if (partit%npes >= 8) call check(sp(1) > 0.0_WP .and. sp(2) > 0.0_WP, 'P3 sync off: the '// &
                                             'copies of U^M and <<U>> drift apart (P2 is not vacuous)', partit%mype)
            if (partit%mype == 0) write(*,'(a,2es10.3,a,2es10.3)') &
                'test_ssh_se: P3 sync off: spreads U^M, <<U>>', sp(1:2), '; inflow-sum ratio <<U>>, U^M', r
        end if

        ! ---- P4: np 1 vs np 2 vs np 8 ----
        call p_partition_independence(dp2, NK)
        deallocate(is_recv, infl, wf)
    end subroutine p_many_ranks

    subroutine p5_flow()
        ! P5 (header): P2's NK SE steps with the flow-dependent viscosity, the shared-element sync on
        integer, parameter :: NK = 6
        type(t_dyn) :: d, dl
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e, n, nz, k, l, nbad
        logical :: sync_env
        real(kind=WP) :: sp(3), spmax(3), dmax, umax
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        sync_env = partit%sync_shared_elem
        partit%sync_shared_elem = .true.
        call p_state(d)
        dl = d
        d%se_visc_flow = .true.
        spmax = 0.0_WP; nbad = 0
        do k = 1, NK
            call p_step(d, k)
            call p_step(dl, k)
            sp(1) = max(shared_copy_max_elem(d%se_ubt(:, :, 0), partit), &
                        shared_copy_max_elem(d%se_ubt(:, :, 1), partit), &
                        shared_copy_max_elem(d%se_ubt(:, :, 2), partit))
            sp(2) = shared_copy_max_elem(d%se_umean, partit)
            sp(3) = shared_copy_max_elem(d%uv, partit)
            do l = 1, 3
                call max_all(sp(l), partit)
            end do
            spmax = max(spmax, sp)
            do n = 1, nNodO
                if (.not. (abs(d%eta_n(n)) <= huge(1.0_WP))) nbad = nbad + 1
            end do
            do e = 1, nElemO
                if (.not. all(abs(d%se_ubt(:, e, 0)) <= huge(1.0_WP))) nbad = nbad + 1
                do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                    if (.not. all(abs(d%uv(:, nz, e)) <= huge(1.0_WP))) nbad = nbad + 1
                end do
            end do
        end do
        partit%sync_shared_elem = sync_env
        dmax = maxval(abs(d%se_ubt(:, 1:nElemO, 0) - dl%se_ubt(:, 1:nElemO, 0)))
        umax = maxval(abs(dl%se_ubt(:, 1:nElemO, 0)))
        call max_all(dmax, partit); call max_all(umax, partit); call sum_all(nbad, partit)
        call check(nbad == 0, 'P5 flow viscosity: eta, U, uv finite', partit%mype)
        call check(all(spmax == 0.0_WP), 'P5 flow viscosity: copy spreads of U (3 levels), <<U>>, uv '// &
                   'are 0 after every step', partit%mype)
        call check(dmax > 0.0_WP, 'P5 flow viscosity: U^M differs from the linear viscosity''s', partit%mype)
        if (partit%mype == 0) write(*,'(a,i0,a,3es10.3,a,es10.3,a,es10.3,a)') 'test_ssh_se: P5 ', NK, &
            ' SE steps, flow viscosity, sync on: max spreads U/<<U>>/uv', spmax, '; max|U^M - U^M(linear)| ', &
            dmax, ' of ', umax, ' m2/s'
    end subroutine p5_flow

    subroutine p_state(d)
        ! the pre-predictor state of the P tests (block header), se_init's cold start, hbar = eta_n
        type(t_dyn), intent(inout) :: d
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, c
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(d)
        do e = 1, nElemF
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                do c = 1, 2
                    d%uv(c, nz, e) = 0.02_WP*hval(partit%myList_elem2D(e), 10*nz + c + 60)
                end do
                d%work%pgf_x(nz, e) = 1.0e-6_WP*hval(partit%myList_elem2D(e), 10*nz + 7)
                d%work%pgf_y(nz, e) = 1.0e-6_WP*hval(partit%myList_elem2D(e), 10*nz + 8)
            end do
        end do
        do n = 1, nNodL
            d%eta_n(n) = 0.1_WP*hval(partit%myList_nod2D(n), 61)
        end do
        allocate(d%work%u_c(mesh%nl-1, nElemF), d%work%v_c(mesh%nl-1, nElemF))
        d%work%u_c = 0.0_WP; d%work%v_c = 0.0_WP
        d%visc_gamma0   = 0.003_WP          ! opt_visc 7 as the drivers set it
        d%visc_gamma1   = 0.1_WP
        d%visc_gamma2   = 0.285_WP
        d%visc_gamma0_h = 0.0_WP
        d%visc_gamma1_h = 0.0_WP
        call se_init(d, mesh, partit, dt)
        if (.not. allocated(mesh%hbar)) allocate(mesh%hbar(nNodL), mesh%hbar_old(nNodL))
        mesh%hbar(1:nNodL)     = d%eta_n(1:nNodL)
        mesh%hbar_old(1:nNodL) = 0.0_WP
    end subroutine p_state

    subroutine p_step(d, k)
        ! SE step k of the P tests through the real kernels, in step_oce's order
        type(t_dyn), intent(inout) :: d
        integer,     intent(in)    :: k
        real(kind=WP), allocatable :: wf(:)
        allocate(wf(size(d%eta_n)))
        wf = 0.0_WP
        call compute_vel_rhs(d, mesh, dt, k == 1, partit)
        call viscosity_filter(7, d, mesh, dt, partit)
        call se_bt_forcing(d, mesh, partit, dt, k == 1)
        call se_subcycle(d, mesh, partit, dt)
        call se_trim_finish(d, mesh, partit, dt, k, wf)
    end subroutine p_step

    real(kind=WP) function p_ratio(x)
        ! the inflow-sum (divsum) ratio |sum x| / sum|x| of owned-node values, reduced
        real(kind=WP), intent(in) :: x(:)
        real(kind=WP) :: s(2)
        s = [sum(x), sum(abs(x))]
        call sum_all(s, partit)
        p_ratio = abs(s(1))/max(s(2), tiny(1.0_WP))
    end function p_ratio

    subroutine p_gather(vals, gids, nglob, out)
        ! out(gid) on rank 0 from every rank's (gids, vals); a gid held by several ranks (a shared
        ! element) takes the value of the last one, the copies being equal. Unheld gids keep huge.
        real(kind=WP), intent(in)  :: vals(:)
        integer,       intent(in)  :: gids(:)
        integer,       intent(in)  :: nglob
        real(kind=WP), intent(out) :: out(nglob)
        integer :: i, ierr, nloc, cnt(partit%npes), displ(partit%npes)
        integer,       allocatable :: gl(:), ga(:)
        real(kind=WP), allocatable :: vl(:), va(:)
        out  = huge(1.0_WP)
        nloc = size(vals)
        gl = gids
        vl = vals
        if (partit%npes == 1) then
            do i = 1, nloc
                out(gl(i)) = vl(i)
            end do
            return
        end if
        call MPI_Gather(nloc, 1, MPI_INTEGER, cnt, 1, MPI_INTEGER, 0, partit%MPI_COMM_FESOM, ierr)
        displ = 0
        if (partit%mype == 0) then
            do i = 2, partit%npes
                displ(i) = displ(i-1) + cnt(i-1)
            end do
            allocate(ga(sum(cnt)), va(sum(cnt)))
        else
            allocate(ga(1), va(1))
        end if
        call MPI_Gatherv(gl, nloc, MPI_INTEGER, ga, cnt, displ, MPI_INTEGER, 0, partit%MPI_COMM_FESOM, ierr)
        call MPI_Gatherv(vl, nloc, MPI_WP, va, cnt, displ, MPI_WP, 0, partit%MPI_COMM_FESOM, ierr)
        if (partit%mype == 0) then
            do i = 1, size(ga)
                out(ga(i)) = va(i)
            end do
        end if
    end subroutine p_gather

    subroutine p_partition_independence(d, nsteps)
        ! P4 (block header): eta (owned nodes) and U^M (owned elements) by global id; np 1 writes
        ! the reference file, npes > 1 compares with it (rank 0)
        type(t_dyn), intent(in) :: d
        integer,     intent(in) :: nsteps
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, u, ios, hdr(4), hdr_ref(4)
        character(len=512) :: path
        real(kind=WP), allocatable :: eta(:), ux(:), uy(:), eta_r(:), ux_r(:), uy_r(:)
        real(kind=WP) :: de, du, se, su
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        path = ''
        call get_environment_variable('FESOM3_TEST_SE_P4REF', path, status=ios)
        if (ios /= 0 .or. len_trim(path) == 0) then
            if (partit%mype == 0) write(*,'(a)') 'test_ssh_se: P4 skipped (FESOM3_TEST_SE_P4REF unset)'
            return
        end if
        allocate(eta(mesh%nod2D), ux(mesh%elem2D), uy(mesh%elem2D))
        call p_gather(d%eta_n(1:nNodO), partit%myList_nod2D(1:nNodO), mesh%nod2D, eta)
        call p_gather(d%se_ubt(1, 1:nElemO, 0), partit%myList_elem2D(1:nElemO), mesh%elem2D, ux)
        call p_gather(d%se_ubt(2, 1:nElemO, 0), partit%myList_elem2D(1:nElemO), mesh%elem2D, uy)
        hdr = [mesh%nod2D, mesh%elem2D, nsteps, d%se_M]
        if (partit%mype /= 0) return
        call check(all(eta < huge(1.0_WP)) .and. all(ux < huge(1.0_WP)) .and. all(uy < huge(1.0_WP)), &
                   'P4 every global node and element gathered', partit%mype)
        if (partit%npes == 1) then
            open(newunit=u, file=trim(path), access='stream', form='unformatted', status='replace', &
                 action='write', iostat=ios)
            if (ios == 0) write(u, iostat=ios) hdr, eta, ux, uy
            if (ios == 0) close(u)
            call check(ios == 0, 'P4 np 1 writes the reference '//trim(path), partit%mype)
            write(*,'(a,a)') 'test_ssh_se: P4 np 1 reference written: ', trim(path)
            return
        end if
        allocate(eta_r(mesh%nod2D), ux_r(mesh%elem2D), uy_r(mesh%elem2D))
        hdr_ref = -1
        open(newunit=u, file=trim(path), access='stream', form='unformatted', status='old', &
             action='read', iostat=ios)
        if (ios == 0) read(u, iostat=ios) hdr_ref
        if (ios == 0 .and. all(hdr_ref == hdr)) read(u, iostat=ios) eta_r, ux_r, uy_r
        if (ios == 0) close(u)
        call check(ios == 0 .and. all(hdr_ref == hdr), 'P4 the np 1 reference '//trim(path)// &
                   ' exists and matches (nod2D, elem2D, steps, M)', partit%mype)
        if (.not. (ios == 0 .and. all(hdr_ref == hdr))) return
        de = maxval(abs(eta - eta_r));  se = maxval(abs(eta_r))
        du = max(maxval(abs(ux - ux_r)), maxval(abs(uy - uy_r)));  su = max(maxval(abs(ux_r)), maxval(abs(uy_r)))
        call check(de <= 1.0e-12_WP*se, 'P4 eta agrees with np 1 to 1e-12 of max|eta|', partit%mype)
        call check(du <= 1.0e-12_WP*su, 'P4 U^M agrees with np 1 to 1e-12 of max|U|', partit%mype)
        write(*,'(a,i0,a,i0,a,es10.3,a,es10.3,a,es10.3,a,es10.3,a)') 'test_ssh_se: P4 np ', partit%npes, &
            ' vs np 1 after ', nsteps, ' SE steps: max|d eta| ', de, ' m (', de/se, ' of max|eta|), max|d U| ', &
            du, ' m2/s (', du/su, ' of max|U|)'
    end subroutine p_partition_independence

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

    !===========================================================================
    ! ==== zstar (split-explicit Task 3.1): Z1-Z6 ===============================
    ! z_zstar sets which_ALE = 'zstar' and restores it and helem on exit; the zstar thickness
    ! fields it allocates (hnode, hnode_new, zbar_3d_n, Z_3d_n) stay, so it runs last.
    !   Z1  the depth-law weights c of se_elem_depth on the cold and on a stretched thickness
    !       state: nlevels(e) == min over the element's vertices of nlevels_nod2D (the vertex
    !       bottom the formula relies on); c == 1 exactly at every vertex with nlevels_nod2D <=
    !       nlevels(e)+1 (the shallowest vertex among them), 0 < c < 1 at the deeper ones, 0 beyond
    !       elem2D_nnodes; some elements have a c < 1 (the law differs from sum_v elem_w*d eta).
    !   Z2  H_e(eta^n) = sum_k helem^n: se_depth_at(eta_n) == se_H == sum_k helem (bitwise).
    !   Z3  DECISIVE: a synthetic elevation change (hashed, |d eta| <= 0.5 m) through the REAL
    !       vert_vel_ale + update_thickness_ale, twice: from the cold state, then from the
    !       stretched one with c recomputed from the stretched zbar_3d_n. sum_k helem^{n+1} =
    !       H_e(eta^{n+1}) on every owned element to 1e-14 of max H; the c = 1 law and the linfs
    !       law (c = 0) miss by more than 1 mm (the check discriminates); the layer sum
    !       sum_k hnode = H0 + eta^{n+1} to 1e-14 of max H; se_check_tail_values reports both.
    !   Z4  a uniform freshwater flux W (1e-6 m/s) with U = 0, eta = 0, F = 0, M = 40: eta^M -
    !       eta^0 = -tau*W on every owned node and sum areasvol*(eta^M - eta^0) = -tau*sum
    !       W*areasvol (1e-12 relative); then se_trim_finish with SE_CHECK: ssh_rhs_old =
    !       I(<<U>>) - W*areasvol (bitwise), the freshwater part of the mean-sea-level change
    !       fw = -tau*W, msl net of it 0 and the eta-compatibility at rounding, the line carries
    !       fw=. Under linfs the same W leaves eta exactly 0.
    !   Z5  one full zstar SE step with a hashed W from a hashed state whose thicknesses carry its
    !       elevation: se_subcycle(W) -> se_trim_finish(W, SE_CHECK) -> vert_vel_ale(W) ->
    !       update_thickness_ale. SE_CHECK: eta-compatibility, trim, msl net of the freshwater at
    !       rounding (1e-12 of their scales), fw /= 0, copy spreads 0; tail: the layer sum and the
    !       depth law to 1e-14 of max H, w(1) = 0 to 1e-12 of max|d eta/tau| + max|W|; at npes > 1
    !       the copies of c and helem on the shared elements agree (spread 0), and so do those of
    !       <<U>> and U^M after the substeps for every M = 1..40 (this caught the division by M in
    !       se_subcycle's running mean: copies differed in the last bit at intermediate substeps
    !       and could round back to equal by M = 40); the SE_CHECK_TAIL line format.
    !   Z6  linfs: se_elem_depth leaves c == 0 (bitwise).
    ! Residuals are printed.

    subroutine z_zstar()
        real(kind=WP), allocatable :: helem0(:,:)
        character(len=20) :: ale0
        ale0 = which_ALE
        helem0 = mesh%helem
        which_ALE = 'zstar'
        call z3_tail_law()           ! with Z1 and Z2
        call z4_uniform_w()          ! with Z6
        call z5_full_step()
        which_ALE = ale0
        mesh%helem = helem0
    end subroutine z_zstar

    subroutine z_thickness_cold()
        ! the cold zstar thickness state of the drivers (fesom_conserve): full-cell hnode,
        ! zbar_3d_n and Z_3d_n on owned + halo nodes, hnode_new = hnode, full-cell helem on every
        ! held element, hbar = hbar_old = 0
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, nl
        nl = mesh%nl
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        if (.not. allocated(mesh%hnode))     allocate(mesh%hnode(nl-1, nNodL))
        if (.not. allocated(mesh%hnode_new)) allocate(mesh%hnode_new(nl-1, nNodL))
        if (.not. allocated(mesh%zbar_3d_n)) allocate(mesh%zbar_3d_n(nl, nNodL))
        if (.not. allocated(mesh%Z_3d_n))    allocate(mesh%Z_3d_n(nl-1, nNodL))
        if (.not. allocated(mesh%hbar))      allocate(mesh%hbar(nNodL), mesh%hbar_old(nNodL))
        mesh%hnode = 0.0_MP; mesh%zbar_3d_n = 0.0_MP; mesh%Z_3d_n = 0.0_MP
        do n = 1, nNodL
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                mesh%hnode(nz, n)  = mesh%zbar(nz) - mesh%zbar(nz+1)
                mesh%Z_3d_n(nz, n) = 0.5_WP*(mesh%zbar(nz) + mesh%zbar(nz+1))
            end do
            mesh%zbar_3d_n(1:mesh%nlevels_nod2D(n), n) = mesh%zbar(1:mesh%nlevels_nod2D(n))
        end do
        mesh%hnode_new = mesh%hnode
        mesh%helem = 0.0_MP
        do e = 1, nElemF
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                mesh%helem(nz, e) = mesh%zbar(nz) - mesh%zbar(nz+1)
            end do
        end do
        mesh%hbar = 0.0_MP; mesh%hbar_old = 0.0_MP
    end subroutine z_thickness_cold

    subroutine z_new_dyn(dyn)
        ! new_dyn plus the vertical-velocity arrays vert_vel_ale writes (no w split, no GM)
        type(t_dyn), intent(inout) :: dyn
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, nl
        nl = mesh%nl
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call new_dyn(dyn)
        allocate(dyn%w(nl, nNodL), dyn%w_e(nl, nNodL), dyn%w_i(nl, nNodL), dyn%cfl_z(nl, nNodL))
        dyn%w = 0.0_WP; dyn%w_e = 0.0_WP; dyn%w_i = 0.0_WP; dyn%cfl_z = 0.0_WP
    end subroutine z_new_dyn

    subroutine z_tail(dyn, eta_from, eta_to)
        ! step_oce's ALE tail on a prescribed elevation change, as se_trim_finish leaves it
        ! (owned + halo): hbar_old = eta_from, hbar = eta_to, then the real vert_vel_ale (no
        ! freshwater) and update_thickness_ale
        type(t_dyn),   intent(inout) :: dyn
        real(kind=WP), intent(in)    :: eta_from(:), eta_to(:)
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        mesh%hbar_old(1:nNodL) = eta_from(1:nNodL)
        mesh%hbar(1:nNodL)     = eta_to(1:nNodL)
        call vert_vel_ale(dyn, mesh, dt, partit)
        call update_thickness_ale(mesh, partit)
    end subroutine z_tail

    real(kind=WP) function z_maxH()
        ! max over all elements of sum_k helem (the scale of the depth-law checks)
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        z_maxH = 0.0_WP
        do e = 1, nElemO
            z_maxH = max(z_maxH, real(sum(mesh%helem(mesh%ulevels(e):mesh%nlevels(e)-1, e)), WP))
        end do
        call max_all(z_maxH, partit)
    end function z_maxH

    subroutine z1_weights(dyn, label)
        ! Z1 on the current thickness state (se_elem_depth has run under zstar)
        type(t_dyn),      intent(in) :: dyn
        character(len=*), intent(in) :: label
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, e, k, nn, nlv, nle
        integer :: nbad_min, nbad_one, nbad_frac, nbad_zero, nfrac
        real(kind=WP) :: cmin
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        nbad_min = 0; nbad_one = 0; nbad_frac = 0; nbad_zero = 0; nfrac = 0; cmin = 1.0_WP
        do e = 1, nElemO
            nn  = mesh%elem2D_nnodes(e)
            nle = mesh%nlevels(e)
            if (nle /= minval(mesh%nlevels_nod2D(mesh%elem2D_nodes(1:nn, e)))) nbad_min = nbad_min + 1
            if (any(dyn%se_c(1:nn, e) < 1.0_WP)) nfrac = nfrac + 1
            do k = 1, MAX_NV
                if (k > nn) then
                    if (dyn%se_c(k, e) /= 0.0_WP) nbad_zero = nbad_zero + 1
                    cycle
                end if
                nlv = mesh%nlevels_nod2D(mesh%elem2D_nodes(k, e))
                if (nlv <= nle + 1) then
                    if (dyn%se_c(k, e) /= 1.0_WP) nbad_one = nbad_one + 1
                else
                    if (.not. (dyn%se_c(k, e) > 0.0_WP .and. dyn%se_c(k, e) < 1.0_WP)) nbad_frac = nbad_frac + 1
                    cmin = min(cmin, dyn%se_c(k, e))
                end if
            end do
        end do
        call sum_all(nbad_min, partit); call sum_all(nbad_one, partit); call sum_all(nbad_frac, partit)
        call sum_all(nbad_zero, partit); call sum_all(nfrac, partit)
        cmin = -cmin; call max_all(cmin, partit); cmin = -cmin
        call check(nbad_min == 0, 'Z1 '//label//': nlevels(e) == min over the vertices of nlevels_nod2D', &
                   partit%mype)
        call check(nbad_one == 0, 'Z1 '//label//': c == 1 at the vertices with nlevels_nod2D <= nlevels(e)+1', &
                   partit%mype)
        call check(nbad_frac == 0, 'Z1 '//label//': 0 < c < 1 at the deeper vertices', partit%mype)
        call check(nbad_zero == 0, 'Z1 '//label//': c == 0 beyond elem2D_nnodes', partit%mype)
        call check(nfrac > 0, 'Z1 '//label//': some elements have a c < 1', partit%mype)
        if (partit%mype == 0) write(*,'(a,i0,a,f8.5)') 'test_ssh_se: Z1 '//label//': elements with a c < 1: ', &
            nfrac, ', min c = ', cmin
    end subroutine z1_weights

    subroutine z3_tail_law()
        ! Z1, Z2 and the decisive Z3 (header of this block)
        type(t_dyn) :: d, dz
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, i, nbad
        integer :: elnodes(MAX_NV)
        real(kind=WP), allocatable :: eta_a(:), eta_b(:), HA(:)
        real(kind=WP) :: maxH, s, res(2), resone(2), reslin(2), lay(2), tv(SE_NTAIL), deta
        character(len=8) :: tag
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call z_thickness_cold()
        allocate(eta_a(nNodL), eta_b(nNodL), HA(nElemL))
        call z_new_dyn(d)                                  ! eta_n = 0 = the cold elevation
        call se_init(d, mesh, partit, dt)
        call z_new_dyn(dz)                                 ! the tail's scratch state (uv = 0)
        maxH = z_maxH()
        eta_a = 0.0_WP
        do i = 1, 2
            ! eta^n = eta_a (d%eta_n); eta^{n+1} = eta_b
            do n = 1, nNodL
                eta_b(n) = eta_a(n) + 0.5_WP*hval(partit%myList_nod2D(n), 60 + i)
            end do
            d%eta_n(1:nNodL) = eta_a
            call se_elem_depth(d, mesh, partit)
            if (i == 1) then
                call z1_weights(d, 'cold')
            else
                call z1_weights(d, 'stretched')
            end if
            ! Z2: H_e(eta^n) == se_H == sum_k helem^n (bitwise)
            call se_depth_at(d, mesh, partit, d%eta_n, HA)
            nbad = 0
            do e = 1, nElemO
                s = 0.0_WP
                do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                    s = s + mesh%helem(nz, e)
                end do
                if (HA(e) /= d%se_H(e) .or. d%se_H(e) /= s) nbad = nbad + 1
            end do
            call sum_all(nbad, partit)
            call check(nbad == 0, 'Z2 H_e(eta^n) == se_H == sum_k helem^n (bitwise)', partit%mype)
            ! Z3: the depth law at eta^{n+1}, then the real tail
            call se_depth_at(d, mesh, partit, eta_b, HA)
            call z_tail(dz, eta_a, eta_b)
            res(i) = 0.0_WP; resone(i) = 0.0_WP; reslin(i) = 0.0_WP
            do e = 1, nElemO
                s = 0.0_WP
                do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                    s = s + mesh%helem(nz, e)
                end do
                elnodes = mesh%elem2D_nodes(:, e)
                res(i)    = max(res(i), abs(s - HA(e)))
                resone(i) = max(resone(i), abs(s - (d%se_H(e) + sum(mesh%elem_w(:, e)*(eta_b(elnodes) - eta_a(elnodes))))))
                reslin(i) = max(reslin(i), abs(s - d%se_H(e)))
            end do
            lay(i) = 0.0_WP
            do n = 1, nNodO
                s = 0.0_WP
                do nz = 1, mesh%nlevels_nod2D(n) - 1
                    s = s + mesh%hnode(nz, n)
                end do
                lay(i) = max(lay(i), abs(s - ((mesh%zbar(1) - mesh%zbar(mesh%nlevels_nod2D(n))) + eta_b(n))))
            end do
            call max_all(res(i), partit); call max_all(resone(i), partit); call max_all(reslin(i), partit)
            call max_all(lay(i), partit)
            write(tag, '(a,i0)') 'step ', i
            call check(res(i) <= 1.0e-14_WP*maxH, 'Z3 '//trim(tag)//': sum_k helem^{n+1} = H_e(eta^{n+1}) '// &
                       'through the real vert_vel_ale + update_thickness_ale', partit%mype)
            call check(resone(i) > 1.0e-3_WP .and. reslin(i) > 1.0e-3_WP, 'Z3 '//trim(tag)// &
                       ': the c = 1 and c = 0 laws miss by > 1 mm (the check discriminates)', partit%mype)
            call check(lay(i) <= 1.0e-14_WP*maxH, 'Z3 '//trim(tag)//': sum_k hnode = H0 + eta^{n+1}', partit%mype)
            ! se_check_tail_values on this state: eta_n = eta^{n+1}, hbar - hbar_old = d eta
            d%eta_n(1:nNodL) = eta_b
            d%w = dz%w
            call se_check_tail_values(d, mesh, partit, tv)
            call check(tv(SE_TAIL_HLAW) <= 1.0e-14_WP*maxH .and. tv(SE_TAIL_LAYSUM) <= 1.0e-14_WP*maxH .and. &
                       tv(SE_TAIL_NBAD) == 0.0_WP, 'Z3 '//trim(tag)//': se_check_tail_values reports '// &
                       'the depth law and the layer sum at rounding', partit%mype)
            eta_a = eta_b
        end do
        deta = maxval(abs(eta_b(1:nNodO)))
        call max_all(deta, partit)
        if (partit%mype == 0) then
            write(*,'(a,es10.3,a,es10.3,a)') 'test_ssh_se: Z3 depth law through the tail (max H = ', maxH, &
                ' m, max|eta| = ', deta, ' m):'
            write(*,'(a,2es10.3,a,2es10.3,a,2es10.3,a,2es10.3)') 'test_ssh_se: Z3   steps 1, 2: max|sum helem - H_e(eta)| = ', &
                res, ' m; c = 1 law ', resone, '; c = 0 law ', reslin, '; layer sum ', lay
        end if
        deallocate(eta_a, eta_b, HA)
    end subroutine z3_tail_law

    subroutine z4_uniform_w()
        ! Z4 and Z6 (header of this block)
        type(t_dyn) :: d0, d, dl
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, nbad
        real(kind=WP), allocatable :: wf(:), infl(:)
        real(kind=WP), parameter :: W = 1.0e-6_WP
        real(kind=WP) :: res, vsum(2), vals(SE_NCHK), tw
        character(len=256) :: line
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        call z_thickness_cold()
        allocate(wf(nNodL), infl(nNodL))
        wf = W
        tw = dt*W
        call z_new_dyn(d0)                                 ! uv = 0 (U ring 0), eta_n = 0
        call se_init(d0, mesh, partit, dt)
        call se_elem_depth(d0, mesh, partit)
        d0%se_force = 0.0_WP
        d0%se_M = 40
        d = d0
        call se_subcycle(d, mesh, partit, dt, wf)
        res = 0.0_WP; vsum = 0.0_WP
        do n = 1, nNodO
            res = max(res, abs(d%se_etam(n) + tw))
            vsum(1) = vsum(1) + mesh%areasvol(n)*d%se_etam(n)
            vsum(2) = vsum(2) - dt*wf(n)*mesh%areasvol(n)
        end do
        call max_all(res, partit); call sum_all(vsum, partit)
        call check(res <= 1.0e-12_WP*tw, 'Z4 uniform W, U = 0: eta^M - eta^0 = -tau*W', partit%mype)
        call check(abs(vsum(1) - vsum(2)) <= 1.0e-12_WP*abs(vsum(2)), &
                   'Z4 uniform W: sum areasvol*(eta^M - eta^0) = -tau*sum W*areasvol', partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3,a,es10.3,a)') &
            'test_ssh_se: Z4 uniform W: max|eta^M - eta^0 + tau*W| = ', res, ' m of tau*W = ', tw, &
            '; volume ', vsum(1), ' m3 (-tau*sum W*areasvol = ', vsum(2), ')'
        ! the trim and SE_CHECK: ssh_rhs_old carries -W*areasvol, fw = -tau*W, msl net of it = 0
        d%se_check = .true.
        call se_trim_finish(d, mesh, partit, dt, 3, wf)          ! prints the SE_CHECK line (rank 0)
        call se_inflow(mesh, partit, d%se_umean, infl)
        nbad = 0
        do n = 1, nNodO          ! to 4 ulp: the compilers may contract a - b*c into an FMA
            if (abs(d%ssh_rhs_old(n) - (infl(n) - wf(n)*mesh%areasvol(n))) > &
                4.0_WP*epsilon(1.0_WP)*(abs(infl(n)) + abs(wf(n)*mesh%areasvol(n)))) nbad = nbad + 1
        end do
        call sum_all(nbad, partit)
        call check(nbad == 0, 'Z4 ssh_rhs_old = I(<<U>>) - W*areasvol on owned nodes', partit%mype)
        call se_check_values(d, mesh, partit, dt, d0%eta_n, vals)    ! no water_flux: no accumulation
        call check(abs(vals(SE_CHK_FW) + tw) <= 1.0e-12_WP*tw .and. vals(SE_CHK_FW) == d%se_fwmsl, &
                   'Z4 the freshwater part of the mean-sea-level change fw = -tau*W', partit%mype)
        call check(abs(vals(SE_CHK_MSL)) <= 1.0e-12_WP*tw, 'Z4 msl net of the freshwater = 0', partit%mype)
        call check(vals(SE_CHK_ETACOMP) <= 1.0e-12_WP*tw, 'Z4 eta-compatibility with W at rounding', partit%mype)
        line = se_check_line(3, vals)
        call check(index(line, ' maxU=') > 0 .and. index(line, ' fw=') > index(line, ' maxU='), &
                   'Z4 zstar SE_CHECK line ends with fw=: '//trim(line), partit%mype)
        if (partit%mype == 0) write(*,'(a,es10.3,a,es10.3,a,es10.3)') 'test_ssh_se: Z4 SE_CHECK: fw = ', &
            vals(SE_CHK_FW), ' m, msl net of it ', vals(SE_CHK_MSL), ' m, etacomp ', vals(SE_CHK_ETACOMP)

        ! linfs: Z6 c == 0, and W leaves eta exactly 0
        which_ALE = 'linfs'
        call z_new_dyn(dl)
        call se_init(dl, mesh, partit, dt)
        call se_elem_depth(dl, mesh, partit)
        nbad = count(dl%se_c /= 0.0_WP)
        call sum_all(nbad, partit)
        call check(nbad == 0, 'Z6 linfs: se_elem_depth leaves c == 0 (bitwise)', partit%mype)
        dl%se_force = 0.0_WP
        dl%se_M = 40
        call se_subcycle(dl, mesh, partit, dt, wf)
        nbad = count(dl%se_etam /= 0.0_WP) + count(dl%se_eta /= 0.0_WP)
        call sum_all(nbad, partit)
        call check(nbad == 0, 'Z4 linfs: W leaves eta exactly 0 (virtual salt)', partit%mype)
        which_ALE = 'zstar'
        deallocate(wf, infl)
    end subroutine z4_uniform_w

    subroutine z5_full_step()
        ! Z5 (header of this block)
        type(t_dyn) :: d, dz, dm
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF, n, e, nz, m
        real(kind=WP), allocatable :: wf(:), eta0(:), zero(:), hel(:,:)
        real(kind=WP) :: vals(SE_NCHK), tv(SE_NTAIL), sa(2), etascale, trimscale, wscale, maxH, sp(2)
        character(len=128) :: line
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        ! thicknesses that carry eta^0 = hashed_step_state's eta_n (seed 11): the tail from rest
        call z_thickness_cold()
        allocate(wf(nNodL), eta0(nNodL), zero(nNodL))
        zero = 0.0_WP
        do n = 1, nNodL
            eta0(n) = 0.1_WP*hval(partit%myList_nod2D(n), 11)
            wf(n)   = 1.0e-6_WP*hval(partit%myList_nod2D(n), 72)
        end do
        call z_new_dyn(dz)
        call z_tail(dz, zero, eta0)
        ! the hashed SE state (se_elem_depth under zstar), the w arrays, SE_CHECK on
        call hashed_step_state(d)
        allocate(d%w(mesh%nl, nNodL), d%w_e(mesh%nl, nNodL), d%w_i(mesh%nl, nNodL), d%cfl_z(mesh%nl, nNodL))
        d%w = 0.0_WP; d%w_e = 0.0_WP; d%w_i = 0.0_WP; d%cfl_z = 0.0_WP
        d%se_M = 40
        d%se_check = .true.
        maxH = z_maxH()
        if (partit%npes > 1) then
            ! the copies of <<U>> and U^M after every M = 1..40 from this state: a copy that differs
            ! at an intermediate substep can round back to equal by M = 40 (it did: se_subcycle)
            sp = 0.0_WP
            do m = 1, 40
                dm = d
                dm%se_M = m
                call se_subcycle(dm, mesh, partit, dt, wf)
                sp(1) = max(sp(1), shared_copy_max_elem(dm%se_umean, partit))
                sp(2) = max(sp(2), shared_copy_max_elem(dm%se_ubt(:, :, 0), partit))
            end do
            call max_all(sp(1), partit); call max_all(sp(2), partit)
            call check(all(sp == 0.0_WP), 'Z5 every M = 1..40: copies of <<U>> and U^M equal', partit%mype)
            if (partit%mype == 0) write(*,'(a,2es10.3)') 'test_ssh_se: Z5 M = 1..40: spreads <<U>>, U^M ', sp
        end if
        call se_subcycle(d, mesh, partit, dt, wf)
        call se_trim_finish(d, mesh, partit, dt, 5, wf)          ! prints the SE_CHECK line (rank 0)
        call se_check_values(d, mesh, partit, dt, eta0, vals)    ! no water_flux: no accumulation
        etascale = 0.0_WP; wscale = 0.0_WP
        do n = 1, nNodO
            etascale = max(etascale, abs(eta0(n)), abs(d%eta_n(n) - eta0(n)))
            wscale   = max(wscale, abs(d%eta_n(n) - eta0(n))/dt + abs(wf(n)))
        end do
        trimscale = 0.0_WP
        do e = 1, nElemO
            sa = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                sa = sa + mesh%helem(nz, e)*abs(d%uv(:, nz, e))
            end do
            trimscale = max(trimscale, maxval(sa))
        end do
        call max_all(etascale, partit); call max_all(wscale, partit); call max_all(trimscale, partit)
        call check(vals(SE_CHK_ETACOMP) <= 1.0e-12_WP*etascale, 'Z5 eta-compatibility with W at rounding', &
                   partit%mype)
        call check(vals(SE_CHK_TRIM) <= 1.0e-12_WP*trimscale, 'Z5 trim at rounding', partit%mype)
        call check(abs(vals(SE_CHK_MSL)) <= 1.0e-12_WP*etascale .and. vals(SE_CHK_FW) /= 0.0_WP, &
                   'Z5 msl net of the (nonzero) freshwater part = 0', partit%mype)
        call check(vals(SE_CHK_SPREADU) == 0.0_WP .and. vals(SE_CHK_SPREADUM) == 0.0_WP .and. &
                   vals(SE_CHK_NBAD) == 0.0_WP, 'Z5 copy spreads 0, no non-finite value', partit%mype)
        ! the tail with W, then the zstar tail checks
        call vert_vel_ale(d, mesh, dt, partit, water_flux=wf)
        call update_thickness_ale(mesh, partit)
        call se_check_tail_values(d, mesh, partit, tv)
        call check(tv(SE_TAIL_LAYSUM) <= 1.0e-14_WP*maxH, 'Z5 tail: sum_k hnode = H0 + eta^{n+1}', partit%mype)
        call check(tv(SE_TAIL_W1) <= 1.0e-12_WP*wscale, 'Z5 tail: w(1) = 0 with freshwater', partit%mype)
        call check(tv(SE_TAIL_HLAW) <= 1.0e-14_WP*maxH, 'Z5 tail: sum_k helem^{n+1} = H_e(eta^{n+1})', &
                   partit%mype)
        call check(tv(SE_TAIL_NBAD) == 0.0_WP, 'Z5 tail: no non-finite value', partit%mype)
        sp = 0.0_WP
        if (partit%npes > 1) then
            hel = real(mesh%helem, WP)
            sp(1) = shared_copy_max_elem(d%se_c, partit)
            sp(2) = shared_copy_max_elem(hel, partit)
            call max_all(sp(1), partit); call max_all(sp(2), partit)
            call check(all(sp == 0.0_WP), 'Z5 c and helem^{n+1}: shared copies equal', partit%mype)
        end if
        line = se_check_tail_line(5, tv)
        call check(index(line, 'SE_CHECK_TAIL step=5 laysum=') == 1 .and. index(line, ' w1=') > 0 .and. &
                   index(line, ' hlaw=') > 0 .and. index(trim(line), '  ') == 0 .and. &
                   index(trim(line), '= ') == 0, 'Z5 the SE_CHECK_TAIL line: '//trim(line), partit%mype)
        if (partit%mype == 0) then
            write(*,'(a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,3es10.3)') 'test_ssh_se: Z5 SE_CHECK: etacomp ', &
                vals(SE_CHK_ETACOMP), ' m of ', etascale, '; trim ', vals(SE_CHK_TRIM), ' m2/s; msl ', &
                vals(SE_CHK_MSL), ' m net of fw ', vals(SE_CHK_FW), '; spreads U, <<U>>, nbad ', &
                vals(SE_CHK_SPREADU), vals(SE_CHK_SPREADUM), vals(SE_CHK_NBAD)
            write(*,'(a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,es10.3,a,2es10.3)') 'test_ssh_se: Z5 tail: laysum ', &
                tv(SE_TAIL_LAYSUM), ' m, hlaw ', tv(SE_TAIL_HLAW), ' m (max H ', maxH, '), w(1) ', &
                tv(SE_TAIL_W1), ' m/s of ', wscale, '; spreads c, helem ', sp
        end if
        deallocate(wf, eta0, zero)
    end subroutine z5_full_step
    ! ==== end of the zstar block ===============================================

    subroutine r1_restart_registry()
        ! R1 (header): the split-explicit stores of restart_register_state. Registration only binds
        ! pointers, so every model array is allocated at size 1 (the tracer loop is empty); the
        ! mesh arrays hbar / hnode exist from the P and Z blocks.
        type(t_dyn),    target :: d
        type(t_tracer), target :: tr
        type(t_ice),    target :: ice
        type(t_restart) :: Ron, Roff
        character(len=13), parameter :: SE_NAMES(10) = [character(len=13) :: 'se_eta1', 'se_eta2', &
            'se_ubt0_u', 'se_ubt0_v', 'se_ubt1_u', 'se_ubt1_v', 'se_ubt2_u', 'se_ubt2_v', &
            'se_sbar_old_u', 'se_sbar_old_v']
        integer :: k, j, nse, nmiss
        logical :: same
        allocate(d%eta_n(1), d%d_eta(1), d%ssh_rhs_old(1), d%uv(2, 1, 1), d%uv_rhsAB(1, 2, 1, 1))
        allocate(d%w(1, 1), d%w_e(1, 1), d%w_i(1, 1), d%se_eta(1, 2), d%se_ubt(2, 1, 0:2))
        allocate(d%se_sbar_old(2, 1))
        d%AB_order = 2
        tr%num_tracers = 0
        allocate(ice%data(3))
        do k = 1, 3
            allocate(ice%data(k)%values(1))
        end do
        allocate(ice%uice(1), ice%vice(1), ice%thermo%t_skin(1))
        allocate(ice%work%sigma11(1), ice%work%sigma12(1), ice%work%sigma22(1))
        if (.not. allocated(mesh%hbar))  allocate(mesh%hbar(1))
        if (.not. allocated(mesh%hnode)) allocate(mesh%hnode(1, 1))
        Ron%nl = mesh%nl; Roff%nl = mesh%nl

        d%use_ssh_se = .true.
        call restart_register_state(Ron, d, tr, ice, mesh)
        d%use_ssh_se = .false.
        call restart_register_state(Roff, d, tr, ice, mesh)

        nse = 0
        do k = 1, Ron%nf
            if (Ron%f(k)%name(1:3) == 'se_') nse = nse + 1
        end do
        nmiss = 0
        do j = 1, size(SE_NAMES)
            if (.not. any([(trim(Ron%f(k)%name) == trim(SE_NAMES(j)), k = 1, Ron%nf)])) nmiss = nmiss + 1
        end do
        same = (Roff%nf == Ron%nf - 10)
        if (same) then
            do k = 1, Roff%nf
                if (trim(Roff%f(k)%name) /= trim(Ron%f(k)%name)) same = .false.
            end do
        end if
        call check(Ron%se .and. .not. Roff%se, 'R1 R%se is set with SE on only', partit%mype)
        call check(nse == 10 .and. nmiss == 0, 'R1 SE on: exactly the 10 SE stores '// &
                   '(se_eta1/2, se_ubt{0,1,2}_{u,v}, se_sbar_old_{u,v})', partit%mype)
        call check(count([(Roff%f(k)%name(1:3) == 'se_', k = 1, Roff%nf)]) == 0 .and. same, &
                   'R1 SE off: no se_* store, the other stores the same and in the same order', partit%mype)
        if (partit%mype == 0) write(*,'(a,i0,a,i0,a)') 'test_ssh_se: R1 restart registry: ', Ron%nf, &
            ' stores with SE on, ', Roff%nf, ' with SE off (no tracers here)'
    end subroutine r1_restart_registry

end program test_ssh_se
