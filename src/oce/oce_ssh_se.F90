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
    !     U^{m+1}   = U^m + dtbt*(-f e_z x U* - g H^A grad(eta^A) + F + V(U^m - U^n)
    !                             + D(U^m - U^n))
    ! with (notes, subcycling.tex:81-91)
    !     beta = 0.281105, zeta = 0.00976186 - 0.13451357 chi, gamma = 0.083445 - 0.513584 chi,
    !     delta = 1/2 + 2 zeta + gamma + 2 chi,
    ! chi the dissipation parameter, V the linear BT viscosity (FESOM3_SE_VISC_FLOW=1: the
    ! flow-dependent W(U^m) - W(U^n) of se_visc_flow in its place), D the barotropic bottom drag
    ! D(U)_e = -(r_e/H^n_e) U_e, r_e = C_d |u_bottom,e| frozen over the step (FESOM3_SE_BOTTDRAG).
    ! F is the depth-integrated 3-D increment with the pressure gradient and the 3-D AB2 Coriolis
    ! removed (the Coriolis term from stored transport sums, plan D4: Coriolis is kept out of the
    ! barotropic forcing, as in Banerjee et al. 2024 §2.1). The drag and the viscosity are anchored
    ! at U^n, so the 3-D drag and viscosity stay in F. Afterwards the 3-D velocity is trimmed to
    ! the mean transport <<U>>, and the existing ALE tail consumes the result through
    ! hbar/hbar_old.
    !
    ! IN PLACE: the knobs, the state in t_dyn, se_init (coefficients, cold start, CFL guard) and
    ! the step_oce guard (Task 0.3); the element depth H^n and the barotropic forcing F
    ! (se_elem_depth, se_bt_forcing; Task 1.1); the substeps (se_subcycle) with their operators
    ! (se_inflow, se_visc, se_tendency, se_depth_at) and the viscosity coefficients (se_kappa_build,
    ! from se_init; Task 1.2), the opt-in flow-dependent viscosity of the Zenodo SE code
    ! (se_visc_flow, FESOM3_SE_VISC_FLOW; plan Task 5.1 remedy B5), the barotropic bottom drag
    ! (plan Task 5.4; its first-substep Coriolis r-term was unstable and is reverted to D4, plan
    ! Task 5.5); the trim and the commit of the elevation (se_trim_finish, Task 1.3);
    ! the invariant suite FESOM3_SE_CHECK (se_check_core, se_check_tail; Task 1.4); zstar: the
    ! depth law c (se_elem_depth), the freshwater W in the eta equation (se_subcycle,
    ! se_trim_finish) and the zstar checks (mean sea level against the freshwater, the layer
    ! sum, w(1) and the depth law after the tail; Task 3.1). step_oce calls se_bt_forcing ->
    ! se_subcycle -> se_trim_finish in place of the CG solve block. The carried rings are in the
    ! checkpoint (mod_io_restart, restart_register_state; Task 4.2).
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
    !   FESOM3_SE_CHI      SM dissipation parameter, 0 <= chi <= 0.25                 default 0.25
    !                      (0.1 until 2026-10-18: a cold CORE2 run blew up at step ~335 through
    !                      the sea ice, plan Task 5.1 note; 0.25 damps that shelf mode). chi is
    !                      mesh-dependent: use the smallest value that is stable on the mesh
    !                      (minimal dissipation; Sergey Danilov, 2026-10-18; plan Task 5.5)
    !   FESOM3_SE_VISC_AH  BT viscosity A_h = value*l_e [m/s], >= 0; 0 = off          default 0.01
    !   FESOM3_SE_VISC_FLOW 1 = the flow-dependent BT viscosity W of se_visc_flow in place of
    !                      the linear A_h (0/1)                                       default 0
    !   FESOM3_SE_VISC_GAMMA0  its background gamma0 [m^3/s^2], a real >= 0           default 10
    !   FESOM3_SE_VISC_GAMMA1  its flow factor gamma1 [m^2], a real >= 0              default 2750
    !                      (the defaults of the Zenodo SE code, MOD_DYN.F90:135-136, and of
    !                      port_kokkos). FESOM3_SE_VISC_AH has no effect with FLOW=1 and the
    !                      GAMMAs none without it: setting a knob that has no effect stops.
    !   FESOM3_SE_BOTTDRAG 1 = barotropic bottom drag D in the substeps, 0 = none     default 1
    !                      (0/1; plan Task 5.4: Sergey's answer 3, the drag belongs in the
    !                      barotropic equation)
    !   FESOM3_SE_CHECK    1 = per-step invariant suite (0/1)                         default 0
    !                      one rank-0 line per step, key=value tokens (se_check_line):
    !                      SE_CHECK step=<n> etacomp= trim= msl= spreadU= spreadUm= maxeta= maxU=
    !                      (non-linfs: + fw=), and under zstar a second one after the ALE tail
    !                      (se_check_tail_line): SE_CHECK_TAIL step=<n> laysum= w1= hlaw=
    ! Value-checked like FESOM3_SHARED_SYNC: anything outside the range stops the run, so a run
    ! never carries a setting other than the one its job file asked for. The FESOM3_SE_* knobs
    ! are read only when FESOM3_SSH_SE=1. SE is SILENT when off: run_conserve_pi.sh and
    ! run_tri_regression.sh parse fixed output lines.
    !
    ! CFL GUARD (plan Task 0.4), from the substeps' own operator. Every gravity mode psi of the
    ! linear 2-D system, A psi = omega^2 psi with A eta = -I(H^n (-g grad eta))/areasvol
    ! (se_cfl_apply_a: the inflow and the gradient of se_subcycle / se_tendency), is a scalar
    ! oscillator that the AB3-AM4 substeps advance stably iff a = omega*dtbt < a_crit(chi), the
    ! limit of the weights of se_sm_coeffs (se_cfl_stable, se_cfl_acrit: 1.853 / 1.579 / 1.329
    ! at chi = 0 / 0.1 / 0.25). se_init takes omega_max from SE_CFL_NIT power iterations of A
    ! (se_cfl_omega_max; allreduce sums, so it is the same on every rank) and sets
    !     M_min = ceiling(tau/dtbt_max),  dtbt_max = s*a_crit(chi)/omega_max,  s = SE_CFL_SAFETY
    ! s = 0.9 covers what the scalar test does not see: the power-iteration value is a LOWER
    ! bound of omega_max (on CORE2 and pi it reaches a fixed point, last change 0; without any
    ! spectral gap, 100 iterations from a generic start on N nodes still miss omega_max by only
    ! about ln(N)/400, < 5 % for N up to 1e8), and Coriolis (f << omega_max), the BT viscosity
    ! and, under zstar, H(eta) shift the limit by well under 1 %. The margin need not buy
    ! damping: at 0.9 a_crit the fastest mode still decays (spectral radius 0.88 / 0.68 / 0.57
    ! per substep at chi = 0 / 0.1 / 0.25), and the scheme turns unstable abruptly just above
    ! a_crit (1.88 per substep at 1.1 a_crit, chi = 0.1). s >= 0.854 keeps the default M = 50
    ! admissible at chi = 0.25 on CORE2. CORE2, dt 1800 s: omega_max = 3.1518e-2 1/s, M_min =
    ! 40 at chi = 0.1, 48 at chi = 0.25. The earlier res-based bound of port_kokkos
    ! (fesom_ssh_se.cpp:255-296), dtbt <= 0.5*res/sqrt(g*H), is not a stability limit:
    ! omega_max*res/c is mesh-dependent (3.29 at the CORE2 trenches), and its M_min = 35 blew up
    ! at step 5 at chi = 0.1 (plan Task 5.1 note).
    !
    ! The parsers (se_parse_knob) and the guard decision (se_cfl_decision) return their verdict
    ! instead of stopping, so test/test_ssh_se.F90 checks every rejection; the driver-facing
    ! wrappers (se_env_requested, se_read_env, se_init, se_step_guard) stop through se_die
    ! (MPI_Abort, so no rank is left waiting in a collective).
    use mpi
    use, intrinsic :: iso_fortran_env, only: output_unit, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use mod_precision,   only: WP, MPI_WP
    use mod_constants,   only: g, r_earth
    use mod_param_phys,  only: C_d
    use mod_config,      only: ab_epsilon, which_ALE
    use mod_mesh,        only: t_mesh, MAX_NV
    use mod_dyn,         only: t_dyn
    use mod_partit,      only: t_partit
    use mod_part_bounds, only: local_dims, is_multirank
    use mod_halo,        only: exchange_elem_shared, exchange_elem, exchange_nod, &
                               allreduce_max, allreduce_sum, shared_copy_max_elem
    use mod_timer,       only: timer_start, timer_stop, TMR_OCE_SHSYNC, TMR_OCE_SE_BT
    use mod_mesh_rotate, only: trim_cyclic
    use mod_mesh_areas,  only: elem_center
    implicit none
    private
    public :: se_env_requested, se_read_env, se_parse_knob, se_init, se_step_guard
    public :: se_elem_depth, se_bt_forcing, se_subcycle, se_trim_finish
    public :: se_check_core, se_check_values, se_check_line, se_check_tail
    public :: se_inflow, se_visc, se_visc_flow, se_visc_on, se_tendency, se_depth_at, se_kappa_build
    public :: se_knob_conflict
    public :: se_sm_coeffs, se_cfl_m_min, se_cfl_decision
    public :: se_cfl_stable, se_cfl_rho, se_cfl_acrit, se_cfl_apply_a, se_cfl_omega_max, &
              SE_CFL_SAFETY, SE_CFL_NIT
    public :: SE_CFL_OK, SE_CFL_FORCED, SE_CFL_DIE
    public :: SE_NCHK, SE_CHK_ETACOMP, SE_CHK_TRIM, SE_CHK_MSL, SE_CHK_SPREADU, SE_CHK_SPREADUM, &
              SE_CHK_MAXETA, SE_CHK_MAXU, SE_CHK_NBAD, SE_CHK_FW
    public :: se_check_tail_values, se_check_tail_line
    public :: SE_NTAIL, SE_TAIL_LAYSUM, SE_TAIL_W1, SE_TAIL_HLAW, SE_TAIL_NBAD

    ! se_cfl_decision verdicts
    integer, parameter :: SE_CFL_OK     = 0   ! M >= M_min
    integer, parameter :: SE_CFL_FORCED = 1   ! M <  M_min, FESOM3_SE_M_FORCE=1: warn and run
    integer, parameter :: SE_CFL_DIE    = 2   ! M <  M_min: stop
    ! the CFL guard: safety factor s on a_crit/omega_max, power iterations for omega_max
    real(kind=WP), parameter :: SE_CFL_SAFETY = 0.9_WP
    integer,       parameter :: SE_CFL_NIT    = 100

    ! FESOM3_SE_CHECK: the entries of se_check_values' vals (meaning there)
    integer, parameter :: SE_NCHK = 9
    integer, parameter :: SE_CHK_ETACOMP = 1, SE_CHK_TRIM = 2, SE_CHK_MSL = 3, SE_CHK_SPREADU = 4, &
                          SE_CHK_SPREADUM = 5, SE_CHK_MAXETA = 6, SE_CHK_MAXU = 7, SE_CHK_NBAD = 8, &
                          SE_CHK_FW = 9
    ! ... and of se_check_tail_values' tvals (zstar, after the ALE tail)
    integer, parameter :: SE_NTAIL = 4
    integer, parameter :: SE_TAIL_LAYSUM = 1, SE_TAIL_W1 = 2, SE_TAIL_HLAW = 3, SE_TAIL_NBAD = 4

    ! the knobs se_read_env reads (FESOM3_SSH_SE itself: se_env_requested)
    character(len=21), parameter :: SE_KNOBS(9) = [character(len=21) :: &
        'FESOM3_SE_M', 'FESOM3_SE_M_FORCE', 'FESOM3_SE_CHI', 'FESOM3_SE_VISC_AH', 'FESOM3_SE_CHECK', &
        'FESOM3_SE_VISC_FLOW', 'FESOM3_SE_VISC_GAMMA0', 'FESOM3_SE_VISC_GAMMA1', 'FESOM3_SE_BOTTDRAG']

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
        ! The FESOM3_SE_* knobs into dyn; an unset knob keeps its t_dyn default (M 50, chi 0.25,
        ! visc_ah 0.01, the linear viscosity, gamma0 10, gamma1 2750, the bottom drag on, check and
        ! force off). Stops
        ! on a bad value and on a set knob that has no effect (se_knob_conflict). Called by se_init.
        type(t_dyn),    intent(inout) :: dyn
        type(t_partit), intent(in)    :: partit
        character(len=256) :: val, msg
        logical :: is_set, ok, set(size(SE_KNOBS))
        integer :: i

        do i = 1, size(SE_KNOBS)
            call get_knob(trim(SE_KNOBS(i)), val, is_set)
            set(i) = is_set
            if (.not. is_set) cycle
            call se_parse_knob(trim(SE_KNOBS(i)), val, dyn, ok, msg)
            if (.not. ok) call se_die(trim(msg), partit)
        end do
        msg = se_knob_conflict(dyn%se_visc_flow, set(4), set(7) .or. set(8))
        if (len_trim(msg) > 0) call se_die(trim(msg), partit)
    end subroutine se_read_env

    !===========================================================================
    pure function se_knob_conflict(flow, ah_set, gamma_set) result(msg)
        ! The viscosity knobs that have no effect: FESOM3_SE_VISC_AH with the flow-dependent
        ! viscosity, FESOM3_SE_VISC_GAMMA0/1 with the linear one. '' = no conflict.
        logical, intent(in) :: flow, ah_set, gamma_set
        character(len=96)   :: msg
        msg = ''
        if (flow .and. ah_set) msg = 'FESOM3_SE_VISC_AH has no effect with FESOM3_SE_VISC_FLOW=1: unset it'
        if (.not. flow .and. gamma_set) &
            msg = 'FESOM3_SE_VISC_GAMMA0/1 have no effect without FESOM3_SE_VISC_FLOW=1'
    end function se_knob_conflict

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
        case ('FESOM3_SE_VISC_FLOW')
            need = 'must be 0 or 1'
            call parse_01(str, lv, ok)
            if (ok) dyn%se_visc_flow = lv
        case ('FESOM3_SE_VISC_GAMMA0')
            need = 'must be a real >= 0'
            call parse_real(str, rv, ok)
            if (ok) ok = (rv >= 0.0_WP)
            if (ok) dyn%se_visc_g0 = rv
        case ('FESOM3_SE_VISC_GAMMA1')
            need = 'must be a real >= 0'
            call parse_real(str, rv, ok)
            if (ok) ok = (rv >= 0.0_WP)
            if (ok) dyn%se_visc_g1 = rv
        case ('FESOM3_SE_BOTTDRAG')
            need = 'must be 0 or 1'
            call parse_01(str, lv, ok)
            if (ok) dyn%se_bottdrag = lv
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
    pure subroutine se_cfl_poly(a, chi, c)
        ! The characteristic polynomial c(0) + c(1) z + ... + c(6) z^6 of the substeps for one
        ! gravity mode, a = omega*dtbt (se_cfl_stable). The weights of se_sm_coeffs, formed as in
        ! se_subcycle: AB3 (w0, w1, w2), AM4 (d3 on eta^{m+1}, d0, d1, d2 on eta^{m-2}).
        real(kind=WP), intent(in)  :: a, chi
        real(kind=WP), intent(out) :: c(0:6)
        real(kind=WP) :: b, ze, ga, de, w0, w1, w2, d0, d1, d2, d3, a2
        call se_sm_coeffs(chi, b, ze, ga, de)
        w0 =  1.5_WP + b
        w1 = -(0.5_WP + 2.0_WP*b)
        w2 =  b
        d3 =  de
        d0 =  1.0_WP - de - ga - ze
        d1 =  ga
        d2 =  ze
        a2 = a*a
        ! z^4 (z-1)^2 + a^2 (w0 z^2 + w1 z + w2)(d3 z^3 + d0 z^2 + d1 z + d2)
        c(0) = a2*(w2*d2)
        c(1) = a2*(w1*d2 + w2*d1)
        c(2) = a2*(w0*d2 + w1*d1 + w2*d0)
        c(3) = a2*(w0*d1 + w1*d0 + w2*d3)
        c(4) = a2*(w0*d0 + w1*d3) + 1.0_WP
        c(5) = a2*(w0*d3) - 2.0_WP
        c(6) = 1.0_WP
    end subroutine se_cfl_poly

    !===========================================================================
    pure logical function se_cfl_stable(a, chi, r)
        ! .true. when every root of the substeps' characteristic polynomial lies in |z| < r. The
        ! scalar test: one gravity mode psi of the 2-D system (A psi = omega^2 psi, se_cfl_apply_a;
        ! eta = alpha psi, U = beta H(-g grad psi)/omega) obeys alpha' = -omega beta, beta' =
        ! omega alpha, and se_subcycle with F = 0, f = 0 and no viscosity advances it as
        !     beta*     = w0 beta^m + w1 beta^{m-1} + w2 beta^{m-2}                       (AB3)
        !     alpha^{m+1} = alpha^m - a beta*,                       a = omega*dtbt
        !     alpha^A   = d3 alpha^{m+1} + d0 alpha^m + d1 alpha^{m-1} + d2 alpha^{m-2}     (AM4)
        !     beta^{m+1} = beta^m + a alpha^A
        ! so (alpha, beta) ~ z^m with p(z) = z^4 (z-1)^2 + a^2 (w0 z^2 + w1 z + w2)(d3 z^3 + d0 z^2
        ! + d1 z + d2) = 0 (se_cfl_poly). The roots of p(r z) are those of p divided by r, and
        ! p(r z) has all its roots in |z| < 1 iff the Schur-Cohn recursion passes: a real
        ! c_0 + ... + c_n z^n with |c_0| < |c_n| has all roots inside iff q(z) = (c_n p(z) -
        ! c_0 z^n p(1/z))/z, of degree n-1, has (Rouche on |z| = 1; q_k = c_n c_{k+1} -
        ! c_0 c_{n-1-k}); |c_0| >= |c_n| means |product of the roots| >= 1. No root finder.
        real(kind=WP), intent(in) :: a, chi, r
        real(kind=WP) :: c(0:6), q(0:5), rk
        integer       :: k, n
        call se_cfl_poly(a, chi, c)
        rk = 1.0_WP
        do k = 0, 6
            c(k) = c(k)*rk
            rk   = rk*r
        end do
        se_cfl_stable = .false.
        do n = 6, 1, -1
            if (.not. (abs(c(0)) < abs(c(n)))) return
            do k = 0, n - 1
                q(k) = c(n)*c(k+1) - c(0)*c(n-1-k)
            end do
            c(0:n-1) = q(0:n-1)/q(n-1)            ! leading coefficient c_n^2 - c_0^2 > 0
        end do
        se_cfl_stable = .true.
    end function se_cfl_stable

    !===========================================================================
    pure real(kind=WP) function se_cfl_rho(a, chi)
        ! The spectral radius max|z| of the substep recurrence of se_cfl_stable, the growth factor
        ! per substep of a gravity mode with a = omega*dtbt. Bisection on r with se_cfl_stable
        ! between 0 and twice the Cauchy bound 1 + max_k |c_k| (c_6 = 1), 64 halvings.
        real(kind=WP), intent(in) :: a, chi
        real(kind=WP) :: c(0:6), lo, hi, mid
        integer       :: it
        call se_cfl_poly(a, chi, c)
        lo = 0.0_WP
        hi = 2.0_WP*(1.0_WP + maxval(abs(c(0:5))))
        do it = 1, 64
            mid = 0.5_WP*(lo + hi)
            if (se_cfl_stable(a, chi, mid)) then
                hi = mid
            else
                lo = mid
            end if
        end do
        se_cfl_rho = hi
    end function se_cfl_rho

    !===========================================================================
    pure real(kind=WP) function se_cfl_acrit(chi)
        ! a_crit(chi): the substeps are stable for omega*dtbt < a_crit, i.e. the first a at which
        ! a root reaches |z| = 1 (se_cfl_stable with r = 1). A scan in steps of 0.005 from a = 0.5,
        ! then 60 bisections. Every chi in [0, 0.25] is stable on (0, 0.5] (radius 1 - O(a^2..a^4)
        ! at small a, 0.95 / 0.90 / 0.87 at a = 0.5 for chi = 0 / 0.1 / 0.25), and the step is
        ! far below the width of the stable range. 1.853 / 1.579 / 1.329 at chi = 0 / 0.1 / 0.25,
        ! the values of the root finder in plan Task 5.1 (test_ssh_se C3). 4 if none below 4.
        real(kind=WP), intent(in) :: chi
        real(kind=WP), parameter :: A0 = 0.5_WP, DA = 0.005_WP, AMAX = 4.0_WP
        real(kind=WP) :: lo, hi, mid
        integer       :: k, it
        hi = AMAX
        do k = 1, nint((AMAX - A0)/DA)
            if (.not. se_cfl_stable(A0 + real(k, WP)*DA, chi, 1.0_WP)) then
                hi = A0 + real(k, WP)*DA
                exit
            end if
        end do
        lo = hi - DA
        if (hi >= AMAX) lo = AMAX
        do it = 1, 60
            mid = 0.5_WP*(lo + hi)
            if (se_cfl_stable(mid, chi, 1.0_WP)) then
                lo = mid
            else
                hi = mid
            end if
        end do
        se_cfl_acrit = lo
    end function se_cfl_acrit

    !===========================================================================
    subroutine se_cfl_apply_a(mesh, partit, H, v, U, inflow, Av)
        ! Av = A v = -I(H (-g grad v))/areasvol on the OWNED nodes: the linear gravity-wave
        ! operator of the substeps, eta_tt = -A eta for eta_t = I(U)/areasvol, U_t = H (-g grad
        ! eta) (se_subcycle without F, Coriolis and viscosity; grad through gradient_sca exactly
        ! as se_tendency, I = se_inflow). I is minus the transpose of the element gradient (the
        ! median-dual flux), so areasvol*A = g G^T H G: A is self-adjoint in the areasvol-weighted
        ! inner product and >= 0 (test_ssh_se C3). v on owned + halo nodes, H on the owned
        ! elements; U (2, >= nElemL) and inflow (>= nNodL) are work arrays.
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: H(:)          ! (>= nElemO)
        real(kind=WP),  intent(in)    :: v(:)          ! (nNodL)
        real(kind=WP),  intent(inout) :: U(:,:)        ! (2, >= nElemL) work
        real(kind=WP),  intent(inout) :: inflow(:)     ! (>= nNodL) work
        real(kind=WP),  intent(inout) :: Av(:)         ! (>= nNodO) out on the owned nodes
        integer       :: e, n, elnodes(MAX_NV)
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: pre(MAX_NV)

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        do e = 1, nElemO
            elnodes = mesh%elem2D_nodes(:, e)
            pre = -(g*v(elnodes))
            U(1, e) = H(e)*sum(mesh%gradient_sca(1:4, e)*pre)
            U(2, e) = H(e)*sum(mesh%gradient_sca(5:8, e)*pre)
        end do
        call se_inflow(mesh, partit, U, inflow)
        do n = 1, nNodO
            Av(n) = -inflow(n)/mesh%areasvol(n)
        end do
    end subroutine se_cfl_apply_a

    !===========================================================================
    subroutine se_cfl_omega_max(mesh, partit, omega, relchg, vec)
        ! omega_max = sqrt(lambda_max(A)), the frequency of the fastest discrete gravity mode of
        ! the substeps (se_cfl_apply_a), by SE_CFL_NIT power iterations on the owned nodes:
        !     v_0 = hash(global node id),  lambda_k = ||A v_k||_W,  v_{k+1} = A v_k/lambda_k
        ! ||x||_W^2 = sum areasvol*x^2 over the owned nodes (allreduce), ||v_k||_W = 1, and
        ! exchange_nod after every iterate (A reads v at the halo vertices of owned elements).
        ! H = H^n = sum_k helem over the element's layers, the depth of the substeps (se_elem_depth's
        ! loop; under zstar the current thickness). For the W-self-adjoint A >= 0, lambda_k lies
        ! between the Rayleigh quotient (v_k, A v_k)_W and lambda_max: a LOWER bound, covered by
        ! the guard's safety factor. The start vector depends on the global id only, so the result
        ! depends on the partition only through the order of the sums (test_ssh_se C3: np 1 vs np
        ! 2). relchg = |lambda_N - lambda_{N-1}|/lambda_N, the change in the last iteration (0 if
        ! lambda = 0). vec (optional, >= nNodL): the last iterate v_N, halo-valid, ||v_N||_W = 1.
        ! Collective at npes > 1.
        type(t_mesh),   intent(in)            :: mesh
        type(t_partit), intent(in)            :: partit
        real(kind=WP),  intent(out)           :: omega
        real(kind=WP),  intent(out)           :: relchg
        real(kind=WP),  intent(out), optional :: vec(:)
        integer          :: n, e, nz, it
        integer          :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        integer(int64)   :: hsh
        real(kind=WP)    :: nrm, lam, lam_old
        real(kind=WP), allocatable :: H(:), v(:), Av(:), U(:,:), inflow(:)

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        allocate(H(nElemL), v(nNodL), Av(nNodL), U(2, nElemL), inflow(nNodL))
        H = 0.0_WP; Av = 0.0_WP; U = 0.0_WP; inflow = 0.0_WP
        do e = 1, nElemO
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                H(e) = H(e) + mesh%helem(nz, e)
            end do
        end do
        ! a Weyl sequence in the global id (Knuth's multiplicative hash, mod 2^32): integer
        ! arithmetic only, the same value at every copy of a node and on every compiler
        do n = 1, nNodL
            hsh  = mod(int(partit%myList_nod2D(n), int64)*2654435761_int64, 4294967296_int64)
            v(n) = real(hsh, WP)/4294967296.0_WP - 0.5_WP
        end do
        nrm = 0.0_WP
        do n = 1, nNodO
            nrm = nrm + mesh%areasvol(n)*v(n)*v(n)
        end do
        if (is_multirank(partit)) call allreduce_sum(nrm, partit)
        nrm = sqrt(nrm)
        do n = 1, nNodL
            v(n) = v(n)/nrm
        end do
        lam = 0.0_WP; lam_old = 0.0_WP
        do it = 1, SE_CFL_NIT
            call se_cfl_apply_a(mesh, partit, H, v, U, inflow, Av)
            nrm = 0.0_WP
            do n = 1, nNodO
                nrm = nrm + mesh%areasvol(n)*Av(n)*Av(n)
            end do
            if (is_multirank(partit)) call allreduce_sum(nrm, partit)
            lam_old = lam
            lam     = sqrt(nrm)
            if (.not. (lam > 0.0_WP)) exit
            do n = 1, nNodO
                v(n) = Av(n)/lam
            end do
            if (is_multirank(partit)) call exchange_nod(v, partit)
        end do
        omega  = sqrt(max(lam, 0.0_WP))
        relchg = 0.0_WP
        if (lam > 0.0_WP) relchg = abs(lam - lam_old)/lam
        if (present(vec)) vec(1:nNodL) = v(1:nNodL)
        deallocate(H, v, Av, U, inflow)
    end subroutine se_cfl_omega_max

    !===========================================================================
    pure integer function se_cfl_m_min(tau, lim)
        ! The fewest substeps with tau/M <= lim, the admissible substep s*a_crit/omega_max of
        ! se_init; 1 without a limit (lim = huge).
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
        real(kind=WP) :: lim, ub(2), omega, relchg, acrit
        double precision :: twall
        character(len=320) :: msg
        character(len=16)  :: tw

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
        allocate(dyn%se_kdrag(nElemL))
        allocate(dyn%se_umean(2, nElemL), dyn%se_force(2, nElemL), dyn%se_H(nElemL))
        allocate(dyn%se_c(MAX_NV, nElemO), dyn%se_kappa(MAX_NV, nElemO))
        allocate(dyn%se_zlen(MAX_NV, nElemO), dyn%se_vn(2, nElemO))
        dyn%se_etam  = 0.0_WP; dyn%se_etanew = 0.0_WP
        dyn%se_ubtn  = 0.0_WP; dyn%se_umean  = 0.0_WP; dyn%se_force = 0.0_WP
        dyn%se_H     = 0.0_WP; dyn%se_c      = 0.0_WP; dyn%se_kappa = 0.0_WP
        dyn%se_zlen  = 0.0_WP; dyn%se_vn     = 0.0_WP; dyn%se_kdrag = 0.0_WP

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

        ! ---- CFL guard (block CFL GUARD at the top) ----
        twall = MPI_Wtime()
        call se_cfl_omega_max(mesh, partit, omega, relchg)
        acrit = se_cfl_acrit(dyn%se_chi)
        lim   = huge(1.0_WP)
        if (omega > 0.0_WP) lim = SE_CFL_SAFETY*acrit/omega
        M_min = se_cfl_m_min(dt, lim)
        twall = MPI_Wtime() - twall
        if (partit%mype == 0) then
            write(*,'(a)') 'oce_ssh_se: split-explicit barotropic solver ON (Shchepetkin-McWilliams AB3-AM4)'
            if (dyn%se_visc_flow) then
                write(*,'(a,i0,a,f0.3,a,f6.4,a,es9.2,a,es9.2,a,l1,a,l1)') 'oce_ssh_se:   M = ', dyn%se_M, &
                    ', dtbt = ', dt/real(dyn%se_M, WP), ' s, chi = ', dyn%se_chi, &
                    ', visc = flow (gamma0 = ', dyn%se_visc_g0, ' m3/s2, gamma1 = ', dyn%se_visc_g1, &
                    ' m2), bottdrag = ', dyn%se_bottdrag, ', check = ', dyn%se_check
            else
                write(*,'(a,i0,a,f0.3,a,f6.4,a,es9.2,a,l1,a,l1)') 'oce_ssh_se:   M = ', dyn%se_M, &
                    ', dtbt = ', dt/real(dyn%se_M, WP), ' s, chi = ', dyn%se_chi, &
                    ', visc_ah = ', dyn%se_visc_ah, ' m/s, bottdrag = ', dyn%se_bottdrag, &
                    ', check = ', dyn%se_check
            end if
            write(*,'(a,4(a,f12.9))') 'oce_ssh_se:  ', ' beta = ', dyn%se_beta, ', zeta = ', &
                dyn%se_zeta, ', gamma = ', dyn%se_gamma, ', delta = ', dyn%se_delta
            write(tw, '(f12.3)') twall
            write(*,'(a,es10.4,a,i0,a,es8.1,a,f6.4,a,f4.2,a,f0.3,a,i0,a,a,a)') &
                'oce_ssh_se:   CFL guard: omega_max = ', omega, ' 1/s (', SE_CFL_NIT, &
                ' power iterations, last change ', relchg, '), a_crit(chi) = ', acrit, ', s = ', &
                SE_CFL_SAFETY, ' -> dtbt <= ', lim, ' s -> M_min = ', M_min, ' (', &
                trim(adjustl(tw)), ' s)'
        end if
        select case (se_cfl_decision(dyn%se_M, M_min, dyn%se_m_force))
        case (SE_CFL_FORCED)
            if (partit%mype == 0) write(*,'(a,i0,a,i0,a)') 'oce_ssh_se:   WARNING: M = ', &
                dyn%se_M, ' is below the CFL-guard minimum M_min = ', M_min, &
                '; FESOM3_SE_M_FORCE=1, running anyway'
        case (SE_CFL_DIE)
            write(msg,'(a,i0,a,i0,a,f0.3,a,f0.3,a,f0.3,a,f0.3,a,i0,a)') 'M = ', dyn%se_M, &
                ' is below the CFL-guard minimum M_min = ', M_min, ' (dtbt = ', &
                dt/real(dyn%se_M, WP), ' s > ', lim, ' s: omega_max*dtbt = ', &
                omega*dt/real(dyn%se_M, WP), ' > s*a_crit(chi) = ', SE_CFL_SAFETY*acrit, &
                '): set FESOM3_SE_M >= ', M_min, ', or FESOM3_SE_M_FORCE=1 to run anyway'
            call se_die(trim(msg), partit)
        end select

        dyn%se_ready = .true.
    end subroutine se_init

    !===========================================================================
    subroutine se_step_guard(dyn, partit)
        ! step_oce's first-call check. FESOM3_SSH_SE=1 needs a driver that ran se_init
        ! (fesom_lifecycle_native_mr, fesom_conserve): any other driver would silently run the
        ! CG solve. The SE routines take the partition as a required argument, so such a driver
        ! must also pass partit to step_oce.
        type(t_dyn),    intent(in)           :: dyn
        type(t_partit), intent(in), optional :: partit
        if (.not. se_env_requested(partit)) return
        if (.not. dyn%se_ready) &
            call se_die('FESOM3_SSH_SE=1 but this driver does not initialise split-explicit', partit)
        if (.not. present(partit)) &
            call se_die('FESOM3_SSH_SE=1 but step_oce got no partit (split-explicit needs it)', partit)
    end subroutine se_step_guard

    !===========================================================================
    subroutine se_elem_depth(dyn, mesh, partit)
        ! H^n_e = sum_k helem^n over the layers ulevels(e)..nlevels(e)-1 (the loop of se_init's
        ! cold-start transport), on the OWNED + eDim elements: helem is halo-valid, and the
        ! viscosity gather reads H at eDim neighbours. The substeps use the depth law of plan D3,
        !     H_e(eta) = H^n_e + sum_v elem_w(v,e)*c(v,e)*(eta_v - eta^n_v)        (se_depth_at)
        ! Under linfs c == 0 (se_c keeps the zero of se_init), so H_e(eta) = H^n_e.
        ! Under zstar c is what the ALE tail will do to the element depth when eta^n -> eta
        ! (plan Task 3.1), recomputed here once per step from zbar_3d_n^n (the tail of the
        ! previous step built it; this step's tail changes it only after the substeps):
        !   - vert_vel_ale (oce_ale.F90:433-452) stretches the layers nz = 1..nlev_v-2 of an
        !     open-surface vertex v (nlev_v = nlevels_nod2D(v); the deepest layer nlev_v-1 is
        !     not stretched, its top zbar_3d_n(nlev_v-1) is the fixed anchor) by
        !         dh(nz,v) = (zbar_3d_n(nz,v) - zbar_3d_n(nz+1,v))*(eta_v - eta^n_v)/Htilde_v,
        !         Htilde_v = zbar_3d_n(1,v) - zbar_3d_n(nlev_v-1,v);
        !   - update_thickness_ale (:582-637) commits hnode += dh and sets helem(nz,e) =
        !     sum_v elem_w(v,e)*hnode(nz,v) over the element's full range ule..nle-1 (ule =
        !     ulevels(e) = 1, nle = nlevels(e)).
        ! Summed over the element's layers the zbar differences telescope:
        !     sum_nz helem^{n+1} = H^n_e + sum_v elem_w(v,e)*c(v,e)*(eta_v - eta^n_v),
        !     c(v,e) = (zbar_3d_n(ule,v) - zbar_3d_n(K+1,v))/Htilde_v,   K = min(nle-1, nlev_v-2)
        ! (test_ssh_se Z3 checks this through the real tail). nlevels(e) = min_v nlevels_nod2D(v)
        ! (vertex bottom), so c == 1 (bitwise: numerator and Htilde are the same expression) at
        ! every vertex with nlev_v <= nle+1, the element's shallowest vertex among them, and
        ! 0 < c < 1 at the deeper vertices, whose layers below the element's bottom take part of
        ! the stretch. A slot beyond elem2D_nnodes(e) (a triangle's slot 4: elem_w = 0) and a
        ! vertex without an open surface (vert_vel_ale does not stretch it) get c = 0.
        ! Owned elements (se_depth_at's range); a gather over the element's own vertices in slot
        ! order from halo-valid zbar_3d_n, so the copies of a shared element agree bitwise
        ! (plan D2). Other ALE modes stop: the tail stretches the layers under zstar only.
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        integer       :: e, nz, k, v, ule, nle, nlv, kb
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: H, cv

        if (trim(which_ALE) /= 'linfs' .and. trim(which_ALE) /= 'zstar') &
            call se_die('split-explicit supports which_ALE = linfs and zstar, not '// &
                        trim(which_ALE), partit)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        do e = 1, nElemL
            H = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                H = H + mesh%helem(nz, e)
            end do
            dyn%se_H(e) = H
        end do
        if (trim(which_ALE) /= 'zstar') return

        do e = 1, nElemO
            ule = mesh%ulevels(e)
            nle = mesh%nlevels(e)
            do k = 1, MAX_NV
                cv = 0.0_WP
                if (k <= mesh%elem2D_nnodes(e)) then
                    v = mesh%elem2D_nodes(k, e)
                    if (mesh%ulevels_nod2D(v) == 1) then           ! vert_vel_ale's nzmin == 1
                        nlv = mesh%nlevels_nod2D(v)
                        kb  = min(nle - 1, nlv - 2)
                        cv  = (mesh%zbar_3d_n(ule, v) - mesh%zbar_3d_n(kb+1, v)) &
                            / (mesh%zbar_3d_n(1, v) - mesh%zbar_3d_n(nlv-1, v))
                    end if
                end if
                dyn%se_c(k, e) = cv
            end do
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
        !     s^{n-1} = 0 (se_init), as compute_vel_rhs's cold uv_rhsAB. So the Coriolis term is
        !     kept out of the barotropic forcing, as in Banerjee et al. (2024, §2.1). (Plan Task
        !     5.4 replaced this by f e_z x U*^0, the barotropic Coriolis of the first substep, as
        !     Sergey Danilov's answers 3 and 5 suggested; that blew up a cold CORE2 run at every
        !     chi, and the scheme's author confirmed that this form is unstable without a filter,
        !     so plan Task 5.5 restored the stored sums; docs/se_question_packet.md §6.)
        !   - viscosity and bottom drag: the substeps use V(U^m - U^n) and D(U^m - U^n), both 0 at
        !     m = 0: the 3-D viscosity and the 3-D explicit bottom drag (oce_dyn_ivertvisc, frozen
        !     in UV_rhs) stay in F, and the substeps damp only the change within the step.
        ! Momentum advection (also in uv_rhsAB), viscosity, the baroclinic pressure, the wind and
        ! ice stress and the 3-D bottom drag stay in F.
        ! The barotropic bottom drag rate (FESOM3_SE_BOTTDRAG, se_tendency's D; plan Task 5.4):
        !     kdrag_e = r_e/H^n_e,  r_e = C_d*|uv(:, nlevels(e)-1, e)|
        ! the C_d, level and velocity of the 3-D drag of impl_vert_visc_ale (oce_dyn_ivertvisc.F90:
        ! 196, uv not yet updated), frozen over the step: the explicit form of the Zenodo
        ! se_bottdrag (oce_ale_ssh_splitexpl_subcycl.F90:900-921, 1036-1100), there with the
        ! total depth -zbar_e_bot + mean hbar, here H^n. 0 with the drag off.
        ! Then s^{n-1} <- s^n; U^n (ring level 0, owned + eDim) is copied to se_ubtn for
        ! V(U^m - U^n) and D(U^m - U^n); and F and kdrag are synced over the shared elements, one
        ! owner as uv in update_vel (plan D2): the copies of G differ by roundoff (the viscosity
        ! filter's order-dependent sums), and kdrag holds a division (LESSONS L71: the rounding of
        ! a division may depend on the element's position in a vectorised loop). A no-op at 1 rank.
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: dt       ! tau, the baroclinic step
        logical,        intent(in)    :: lfirst   ! first Euler step (compute_vel_rhs's ff = 1)
        integer       :: e, nz, nb, elnodes(MAX_NV)
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
            if (dyn%se_bottdrag) then
                nb = mesh%nlevels(e) - 1
                dyn%se_kdrag(e) = C_d*sqrt(dyn%uv(1, nb, e)**2 + dyn%uv(2, nb, e)**2)/dyn%se_H(e)
            else
                dyn%se_kdrag(e) = 0.0_WP
            end if
        end do

        do e = 1, nElemL
            dyn%se_ubtn(:, e) = dyn%se_ubt(:, e, 0)
        end do

        if (is_multirank(partit)) then
            ! (nested ifs: Fortran does not short-circuit)
            if (partit%sync_shared_elem) then
                call timer_start(TMR_OCE_SHSYNC)
                call exchange_elem_shared(dyn%se_force, partit)
                if (dyn%se_bottdrag) call exchange_elem_shared(dyn%se_kdrag, partit)
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
        ! Also the length scale of the flow-dependent viscosity (se_visc_flow), on every interior
        ! slot whichever viscosity is selected:
        !     zlen(k,e) = sqrt(A_e + A_nb)        (0 on a boundary slot and a triangle's slot 4)
        ! elem_area is halo-valid (exchange_elem_full in compute_geometry), and A_e + A_nb is
        ! the same sum seen from either element, so zlen is bitwise symmetric.
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
        dyn%se_zlen  = 0.0_WP
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
                dyn%se_zlen(k, e)  = sqrt(mesh%elem_area(e) + mesh%elem_area(nb))
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
    subroutine se_visc_flow(dyn, mesh, partit, U, out)
        ! The flow-dependent BT viscosity of the Zenodo SE code (FESOM2.5 split-explicit,
        ! oce_ale_ssh_splitexpl_subcycl.F90:844-1003; port_kokkos fesom_ssh_se.cpp:844-875) on the
        ! OWNED elements, a gather over the slots k = 1..elem2D_nnodes(e) in fixed order (plan D2):
        !     W_e(U) = -(1/A_e) sum_k (U_e - U_nb)*nu_k,
        !     nu_k   = sqrt(max(gamma0, gamma1*|U_e - U_nb|/H_edge)*zlen(k,e))       [m^2/s]
        ! nb = elem_neighbors(k,e) (boundary slot nb == e: no flux through a coast), H_edge =
        ! (H_e + H_nb)/2 with H = H^n (se_H, frozen over the substeps), zlen = sqrt(A_e + A_nb)
        ! (se_kappa_build), gamma0/1 = se_visc_g0/g1. The flux -(U_e - U_nb)*nu_k through an
        ! interior edge has nu_k bitwise the same seen from either element (|dU|, H_edge and zlen
        ! are symmetric sums), so sum_e A_e*W_e = 0 to rounding; W of a uniform TRANSPORT is
        ! exactly 0 (it acts on U, not on U/H as se_visc); gamma1 = 0 leaves the linear operator
        ! with the coefficient sqrt(gamma0*zlen). The substeps use W(U^m) - W(U^n) in place of
        ! V(U^m - U^n), with W(U^n) once per step (se_vn, se_subcycle).
        ! Against the sources: the Zenodo edge loop scatters dt*nu*(dU/hh)*hh/A into both
        ! elements of an interior edge, the same flux, the bracket times 1/M per substep as our
        ! dtbt; its hh is the mean bottom depth of the two elements plus the mean hbar (eta^n) of
        ! the edge's two nodes, port_kokkos's the mean resting depth (H0e = sum_k helem - mean
        ! eta), ours H^n (the resting depth under linfs). Its quadratic term gamma2 (0 in its
        ! defaults; port_kokkos drops it) is not carried.
        ! Reads U and H at eDim neighbours: U must be halo-valid. The divisions stay inside this
        ! element loop with its inner slot loop, which is not vectorised, as in se_visc (LESSONS
        ! L71: a division in a vectorised loop rounds by the element's position).
        type(t_dyn),    intent(in)  :: dyn
        type(t_mesh),   intent(in)  :: mesh
        type(t_partit), intent(in)  :: partit
        real(kind=WP),  intent(in)  :: U(:,:)       ! (2, >= nElemL)
        real(kind=WP),  intent(out) :: out(:,:)     ! (2, >= nElemO)
        integer       :: e, k, nb
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: He, dx, dy, nu, vx, vy

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        do e = 1, nElemO
            He = dyn%se_H(e)
            vx = 0.0_WP
            vy = 0.0_WP
            do k = 1, mesh%elem2D_nnodes(e)
                nb = mesh%elem_neighbors(k, e)
                if (nb == e) cycle                                   ! boundary slot
                dx = U(1, e) - U(1, nb)
                dy = U(2, e) - U(2, nb)
                nu = sqrt(max(dyn%se_visc_g0, dyn%se_visc_g1*sqrt(dx*dx + dy*dy) &
                                              /(0.5_WP*(He + dyn%se_H(nb))))*dyn%se_zlen(k, e))
                vx = vx - dx*nu
                vy = vy - dy*nu
            end do
            out(1, e) = vx/mesh%elem_area(e)
            out(2, e) = vy/mesh%elem_area(e)
        end do
    end subroutine se_visc_flow

    !===========================================================================
    pure logical function se_visc_on(dyn)
        ! The substeps carry a BT viscosity: the flow-dependent one, or the linear one with A_h > 0.
        ! Then U is exchanged after every substep (its eDim neighbours are read).
        type(t_dyn), intent(in) :: dyn
        se_visc_on = dyn%se_visc_flow .or. (dyn%se_visc_ah > 0.0_WP)
    end function se_visc_on

    !===========================================================================
    subroutine se_depth_at(dyn, mesh, partit, eta, HA)
        ! H_e(eta) = H^n_e + sum_v elem_w(v,e)*c(v,e)*(eta_v - eta^n_v) on the OWNED elements:
        ! the one depth function of the substeps (plan D3), no ALE branch in the loop. Under
        ! linfs c == 0 and H_e(eta) = H^n_e exactly; under zstar it is the element depth the
        ! ALE tail builds for the elevation eta (se_elem_depth). eta and eta^n (eta_n) are read
        ! at the element's vertices (owned + halo).
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
    subroutine se_tendency(dyn, mesh, partit, ustar, etaA, HA, uvis, tend)
        ! The right-hand side of the U substep on the OWNED elements (plan Technical Details §2):
        !     tend = -f e_z x U* - g H^A grad(eta^A) + F + V(uvis) + D(U^m - U^n)
        ! -f e_z x U* = f*(U*_2, -U*_1), compute_vel_rhs's (v*ff, -u*ff); -g grad(eta^A) = Fx, Fy
        ! through gradient_sca exactly as in compute_vel_rhs and se_bt_forcing. So with U* = s^n
        ! = s^{n-1}, eta^A = eta^n, H^A = H^n, uvis = 0 and U^m = U^n the r-term inside F cancels
        ! and tend = G/tau to rounding (test_ssh_se C7: the sign/factor consistency of F and the
        ! operator).
        ! The viscosity term: the linear V(uvis) of se_visc, uvis = U^m - U^n, when se_visc_ah >
        ! 0; with se_visc_flow W(uvis) - W(U^n) of se_visc_flow, uvis = U^m and W(U^n) = se_vn
        ! (se_subcycle); else none. The bottom drag (FESOM3_SE_BOTTDRAG): D(U^m - U^n) =
        ! -kdrag_e*(U^m - U^n) with U^m = ring level 0 and U^n = se_ubtn read from dyn, kdrag =
        ! C_d|u_bottom|/H^n (se_bt_forcing): the anchored form of the viscosity, D(U^m) - D(U^n),
        ! so F keeps the 3-D drag and the substeps damp the change. Purely local and explicit:
        ! dtbt*kdrag <= 36 s*0.0025*|u_b|/20 m = 4.5e-3*|u_b|/(m/s) on the 20 m shelf at CORE2's
        ! dtbt, far below the forward-Euler limit 2 (and the AB3-AM4 one). Reads eta^A at the
        ! element's vertices, uvis at eDim neighbours.
        type(t_dyn),    intent(in)  :: dyn
        type(t_mesh),   intent(in)  :: mesh
        type(t_partit), intent(in)  :: partit
        real(kind=WP),  intent(in)  :: ustar(:,:)   ! (2, >= nElemO) U*, the AB3 transport
        real(kind=WP),  intent(in)  :: etaA(:)      ! (nNodL) eta^A, the AM4 elevation
        real(kind=WP),  intent(in)  :: HA(:)        ! (>= nElemO) H^A = H_e(eta^A)
        real(kind=WP),  intent(in)  :: uvis(:,:)    ! (2, >= nElemL) U^m - U^n (linear), U^m (flow)
        real(kind=WP),  intent(out) :: tend(:,:)    ! (2, >= nElemO)
        integer       :: e, elnodes(MAX_NV)
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: f, Fx, Fy, pre(MAX_NV)

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        if (dyn%se_visc_flow) then
            call se_visc_flow(dyn, mesh, partit, uvis, tend)
            do e = 1, nElemO
                tend(1, e) = tend(1, e) - dyn%se_vn(1, e)
                tend(2, e) = tend(2, e) - dyn%se_vn(2, e)
            end do
        else if (dyn%se_visc_ah > 0.0_WP) then
            call se_visc(dyn, mesh, partit, uvis, tend)
        else
            do e = 1, nElemO
                tend(:, e) = 0.0_WP
            end do
        end if
        if (dyn%se_bottdrag) then
            do e = 1, nElemO
                tend(1, e) = tend(1, e) - dyn%se_kdrag(e)*(dyn%se_ubt(1, e, 0) - dyn%se_ubtn(1, e))
                tend(2, e) = tend(2, e) - dyn%se_kdrag(e)*(dyn%se_ubt(2, e, 0) - dyn%se_ubtn(2, e))
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
    subroutine se_subcycle(dyn, mesh, partit, dt, water_flux)
        ! The M barotropic substeps m = 0..M-1 of one baroclinic step tau = dt, after
        ! se_bt_forcing (plan Technical Details §2), dtbt = tau/M:
        !     U*        = (3/2+beta) U^m - (1/2+2 beta) U^{m-1} + beta U^{m-2}     owned elements
        !     <<U>>    += U*/M                                                   owned elements
        !     eta^{m+1} = eta^m + (I(U*) - W*areasvol)*dtbt/areasvol
        !                                                        owned nodes, then exchange_nod
        !     eta^A     = delta eta^{m+1} + (1-delta-gamma-zeta) eta^m + gamma eta^{m-1}
        !                 + zeta eta^{m-2}                                       owned + halo nodes
        !     H^A       = H_e(eta^A)                                             owned elements
        !     U^{m+1}   = U^m + dtbt*se_tendency(U*, eta^A, H^A, U^m - U^n)      owned elements
        !                 (with se_visc_flow: se_tendency(U*, eta^A, H^A, U^m) and W(U^n) = se_vn
        !                 from se_ubtn once per step, before the substeps; the bottom drag
        !                 D(U^m - U^n) inside se_tendency, from ring level 0 and se_ubtn)
        ! eta^0 = eta_n (= eta^n), U^0 = ring level 0. The rings shift BY COPYING, so no ring
        ! offset becomes restart state. With the viscosity on, exchange_elem(U^{m+1}) follows:
        ! V reads U at eDim neighbours. Without it nothing reads U off the owned elements (I reads
        ! owned elements only, the bottom drag is local), the exchange is skipped and the eDim
        ! entries of the U ring stay stale (and unused). Per substep: one 2-D exchange_nod and, with the viscosity, one 2-D
        ! exchange_elem; no reductions. Timer TMR_OCE_SE_BT.
        ! On exit: eta^M in se_etam (eta^{M-1}, eta^{M-2} in se_eta(:,1), se_eta(:,2); all on
        ! owned + halo nodes), U^M in ring level 0 (U^{M-1}, U^{M-2} in levels 1, 2), <<U>> in
        ! se_umean (owned elements). eta_n and hbar are untouched: se_trim_finish commits them.
        ! FRESHWATER: W = water_flux (+ out of the ocean) enters the eta equation when which_ALE
        ! /= linfs and water_flux is present, on the open-surface owned nodes (ulevels_nod2D ==
        ! 1), as -W*areasvol next to the inflow: the term compute_hbar_ale subtracts from
        ! ssh_rhs_old (oce_ale.F90:253-259), and se_trim_finish puts the same term into
        ! ssh_rhs_old, so eta^M - eta^0 = tau*(I(<<U>>) - W*areasvol)/areasvol = tau*ssh_rhs_old/
        ! areasvol to rounding (SE_CHECK etacomp). linfs ignores W (virtual salt), bit for bit
        ! the loop without it.
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: dt       ! tau, the baroclinic step
        real(kind=WP),  intent(in), optional :: water_flux(:)   ! (>= nNodO) + out of the ocean (non-linfs)
        integer       :: m, M_sub, n, e
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: rM, rMinv, dtbt, w0, w1, w2, d0, d1, d2, d3
        logical       :: visc, fw
        real(kind=WP), allocatable :: ustar(:,:), unew(:,:), tend(:,:), dU(:,:)
        real(kind=WP), allocatable :: etaA(:), inflow(:), HA(:)

        call timer_start(TMR_OCE_SE_BT)
        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        M_sub = dyn%se_M
        rM    = real(M_sub, WP)
        dtbt  = dt/rM
        ! <<U>> accumulates U*·(1/M), a MULTIPLICATION by this one reciprocal, not U*/M: with
        ! the Intel anchor's -no-prec-div a division by a loop invariant may become a multiply by
        ! the hoisted reciprocal in the vectorised body of the element loop but stay a true
        ! division in its scalar peel/remainder iterations, so the result depended on the
        ! element's local index, and the copies of a shared element (other local index on each
        ! owner) differed by 1 ULP in <<U>> (test_ssh_se P2, plan Task 2.1). A product is
        ! correctly rounded on every code path.
        rMinv = 1.0_WP/rM
        ! AB3: w0 U^m + w1 U^{m-1} + w2 U^{m-2};  AM4: d3 eta^{m+1} + d0 eta^m + d1 eta^{m-1} + d2 eta^{m-2}
        w0 =  1.5_WP + dyn%se_beta
        w1 = -(0.5_WP + 2.0_WP*dyn%se_beta)
        w2 =  dyn%se_beta
        d3 =  dyn%se_delta
        d0 =  1.0_WP - dyn%se_delta - dyn%se_gamma - dyn%se_zeta
        d1 =  dyn%se_gamma
        d2 =  dyn%se_zeta
        visc = se_visc_on(dyn)
        fw   = .false.
        if (present(water_flux)) fw = (trim(which_ALE) /= 'linfs')

        allocate(ustar(2, nElemL), unew(2, nElemL), tend(2, nElemL), dU(2, nElemL))
        allocate(etaA(nNodL), inflow(nNodL), HA(nElemL))
        ustar = 0.0_WP; unew = 0.0_WP; tend = 0.0_WP; dU = 0.0_WP
        etaA  = 0.0_WP; inflow = 0.0_WP; HA = 0.0_WP

        do n = 1, nNodL
            dyn%se_etam(n) = dyn%eta_n(n)
        end do
        dyn%se_umean = 0.0_WP
        ! the flow-dependent viscosity's W(U^n), frozen over the substeps (se_ubtn is halo-valid).
        ! Into the scratch tend, then copied: se_visc_flow reads dyn, so dyn%se_vn cannot be its
        ! output argument.
        if (dyn%se_visc_flow) then
            call se_visc_flow(dyn, mesh, partit, dyn%se_ubtn, tend)
            do e = 1, nElemO
                dyn%se_vn(:, e) = tend(:, e)
            end do
        end if

        do m = 0, M_sub - 1
            ! AB3 transport and the running mean (U*·rMinv, see above; also caught by
            ! test_ssh_se Z5 at np 2)
            do e = 1, nElemO
                ustar(1, e) = w0*dyn%se_ubt(1, e, 0) + w1*dyn%se_ubt(1, e, 1) + w2*dyn%se_ubt(1, e, 2)
                ustar(2, e) = w0*dyn%se_ubt(2, e, 0) + w1*dyn%se_ubt(2, e, 1) + w2*dyn%se_ubt(2, e, 2)
                dyn%se_umean(1, e) = dyn%se_umean(1, e) + ustar(1, e)*rMinv
                dyn%se_umean(2, e) = dyn%se_umean(2, e) + ustar(2, e)*rMinv
            end do

            ! elevation; non-linfs: minus the freshwater W*areasvol (compute_hbar_ale's term)
            call se_inflow(mesh, partit, ustar, inflow)
            if (fw) then
                do n = 1, nNodO
                    if (mesh%ulevels_nod2D(n) > 1) cycle
                    inflow(n) = inflow(n) - water_flux(n)*mesh%areasvol(n)
                end do
            end if
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
            if (dyn%se_visc_flow) then
                call se_tendency(dyn, mesh, partit, ustar, etaA, HA, dyn%se_ubt(:, :, 0), tend)
            else
                if (visc) then
                    do e = 1, nElemL
                        dU(1, e) = dyn%se_ubt(1, e, 0) - dyn%se_ubtn(1, e)
                        dU(2, e) = dyn%se_ubt(2, e, 0) - dyn%se_ubtn(2, e)
                    end do
                end if
                call se_tendency(dyn, mesh, partit, ustar, etaA, HA, dU, tend)
            end if
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
    subroutine se_trim_finish(dyn, mesh, partit, dt, step, water_flux)
        ! After se_subcycle (plan Technical Details §2): the velocity, elevation and divergence
        ! that the ALE tail consumes, in place of update_vel + compute_hbar_ale + update_eta_n.
        !  1. u** = uv + UV_rhs (update_vel without the SI pressure correction), then the trim on
        !     the OWNED elements
        !         uv_k = u**_k - (sum_j helem^n_j u**_j - <<U>>)/H^n
        !     (the notes' corrector, subcycling.tex:116-117, with h^n): a depth-uniform velocity
        !     correction after which sum_k helem^n uv_k = <<U>>, so the layer divergences of the
        !     tail add up to the elevation change. H^n = se_H sums the same layers in the same
        !     order. Then the shared-element sync and the halo exchange of uv exactly as in
        !     update_vel: the copies of u** differ by roundoff (UV_rhs), <<U>> does not (plan D2).
        !  2. ssh_rhs_old = I(<<U>>), compute_hbar_ale's edge scatter of the mean transport
        !     (complete on owned nodes, partial on halo nodes, as there); when which_ALE /= linfs
        !     minus water_flux*areasvol on the open-surface owned nodes and exchange_nod, as
        !     compute_hbar_ale :253-259.
        !  3. on owned + halo nodes: hbar_old = hbar, hbar = eta^M, eta_n = eta^M (eta^M is
        !     halo-valid; the ice reads hbar and compute_vel_rhs eta_n at halo vertices).
        !     update_eta_n's alpha blend of hbar and hbar_old does not apply: eta^M is the
        !     elevation at n+1.
        !  4. FESOM3_SE_CHECK: se_check_core, while helem is still at n (with water_flux: the
        !     mean-sea-level change is checked against the integrated freshwater). Nothing when
        !     off.
        ! U^M stays in ring level 0 for the next step. dhe (read only by the SI stiffness
        ! update), d_eta and ssh_rhs (SI only) are not touched.
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(inout) :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: dt              ! tau, the baroclinic step
        integer,        intent(in)    :: step            ! step number (SE_CHECK line and stop)
        real(kind=WP),  intent(in)    :: water_flux(:)   ! (>= nNodO) freshwater, + out of the ocean (non-linfs)
        integer       :: e, n, nz
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: s(2), c(2)
        real(kind=WP), allocatable :: eta_old(:)

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)

        ! ---- 1. u** and the trim to <<U>> ----
        do e = 1, nElemO
            s = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                dyn%uv(1, nz, e) = dyn%uv(1, nz, e) + dyn%uv_rhs(1, nz, e)
                dyn%uv(2, nz, e) = dyn%uv(2, nz, e) + dyn%uv_rhs(2, nz, e)
                s(1) = s(1) + mesh%helem(nz, e)*dyn%uv(1, nz, e)
                s(2) = s(2) + mesh%helem(nz, e)*dyn%uv(2, nz, e)
            end do
            c(1) = (s(1) - dyn%se_umean(1, e))/dyn%se_H(e)
            c(2) = (s(2) - dyn%se_umean(2, e))/dyn%se_H(e)
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                dyn%uv(1, nz, e) = dyn%uv(1, nz, e) - c(1)
                dyn%uv(2, nz, e) = dyn%uv(2, nz, e) - c(2)
            end do
        end do
        if (is_multirank(partit)) then
            ! (nested ifs: Fortran does not short-circuit)
            if (partit%sync_shared_elem) then
                call timer_start(TMR_OCE_SHSYNC)
                call exchange_elem_shared(dyn%uv, partit)
                call timer_stop(TMR_OCE_SHSYNC)
            end if
            call exchange_elem(dyn%uv, partit)
        end if

        ! ---- 2. the divergence the tail and the diagnostics read ----
        call se_inflow(mesh, partit, dyn%se_umean, dyn%ssh_rhs_old)
        if (trim(which_ALE) /= 'linfs') then
            do n = 1, nNodO
                if (mesh%ulevels_nod2D(n) > 1) cycle
                dyn%ssh_rhs_old(n) = dyn%ssh_rhs_old(n) - water_flux(n)*mesh%areasvol(n)
            end do
            if (is_multirank(partit)) call exchange_nod(dyn%ssh_rhs_old, partit)
        end if

        ! ---- 3. commit the elevation ----
        if (dyn%se_check) then
            allocate(eta_old(nNodL))
            eta_old = dyn%eta_n(1:nNodL)
        end if
        do n = 1, nNodL
            mesh%hbar_old(n) = mesh%hbar(n)
            mesh%hbar(n)     = dyn%se_etam(n)
            dyn%eta_n(n)     = dyn%se_etam(n)
        end do

        ! ---- 4. invariants ----
        if (dyn%se_check) then
            call se_check_core(dyn, mesh, partit, dt, step, eta_old, water_flux)
            deallocate(eta_old)
        end if
    end subroutine se_trim_finish

    !===========================================================================
    subroutine se_check_values(dyn, mesh, partit, dt, eta_old, vals, water_flux)
        ! The FESOM3_SE_CHECK invariants after se_trim_finish's commit, while helem is still at
        ! n; reduced over the ranks (collective). eta^{n+1} = eta_n, eta^n = eta_old (the copy
        ! taken before the commit). vals(SE_CHK_*):
        !   ETACOMP   max|eta^{n+1} - eta^n - tau*ssh_rhs_old/areasvol| over the owned nodes [m]:
        !             the elevation and the divergence the tail sees agree (ssh_rhs_old =
        !             I(<<U>>) - W*areasvol); the eta-compatibility of the notes
        !   TRIM      max|sum_k helem^n uv_k - <<U>>| over the owned elements, both components
        !             [m^2/s]: the trimmed velocity carries the mean transport
        !   MSL       sum areasvol*eta^{n+1}/sum areasvol (owned nodes) minus the same mean at the
        !             start of the run [m] (dyn%se_msl0, taken from eta^n by the first call of
        !             this process), minus the change the freshwater explains, FW. 0 to rounding
        !             in a closed basin: the inflow I conserves volume
        !   SPREADU, SPREADUM  the copy spreads max|copy - designated copy| of U^M and <<U>> on
        !             the shared elements (plan D2: 0 by construction; 0 at 1 rank)
        !   MAXETA    max|eta^{n+1}| [m];  MAXU  max|U^M| [m^2/s]  (owned)
        !   NBAD      the number of non-finite values in eta_n and ssh_rhs_old (owned nodes),
        !             <<U>> and U^M (owned elements, per element) and uv (wet layers, owned)
        !   FW        the mean-sea-level change of the freshwater since the start of the run [m],
        !             dyn%se_fwmsl, - sum over the calls of tau*sum W*areasvol/sum areasvol
        !             (open-surface owned nodes). Accumulated only when which_ALE /= linfs and
        !             water_flux is present, so call this ONCE per step with water_flux
        !             (se_check_core does); 0 under linfs (virtual salt, no volume flux)
        ! Exposed (not only through se_check_core) so that test_ssh_se can check the values.
        type(t_dyn),    intent(inout) :: dyn          ! se_msl0 / se_fwmsl: set / accumulated
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: dt
        real(kind=WP),  intent(in)    :: eta_old(:)   ! (>= nNodO) eta^n
        real(kind=WP),  intent(out)   :: vals(SE_NCHK)
        real(kind=WP),  intent(in), optional :: water_flux(:)   ! (>= nNodO) + out of the ocean
        integer       :: n, e, nz, nbad
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: mx(6), sm(5), s(2)
        logical       :: fw

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        fw = .false.
        if (present(water_flux)) fw = (trim(which_ALE) /= 'linfs')
        mx = 0.0_WP; sm = 0.0_WP; nbad = 0
        do n = 1, nNodO
            mx(1) = max(mx(1), abs(dyn%eta_n(n) - eta_old(n) - dt*dyn%ssh_rhs_old(n)/mesh%areasvol(n)))
            mx(5) = max(mx(5), abs(dyn%eta_n(n)))
            sm(1) = sm(1) + mesh%areasvol(n)*dyn%eta_n(n)
            sm(2) = sm(2) + mesh%areasvol(n)
            sm(3) = sm(3) + mesh%areasvol(n)*eta_old(n)
            if (fw .and. mesh%ulevels_nod2D(n) == 1) sm(5) = sm(5) + water_flux(n)*mesh%areasvol(n)
            if (.not. (ieee_is_finite(dyn%eta_n(n)) .and. ieee_is_finite(dyn%ssh_rhs_old(n)))) nbad = nbad + 1
        end do
        do e = 1, nElemO
            s = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                s(1) = s(1) + mesh%helem(nz, e)*dyn%uv(1, nz, e)
                s(2) = s(2) + mesh%helem(nz, e)*dyn%uv(2, nz, e)
                if (.not. (ieee_is_finite(dyn%uv(1, nz, e)) .and. ieee_is_finite(dyn%uv(2, nz, e)))) &
                    nbad = nbad + 1
            end do
            mx(2) = max(mx(2), abs(s(1) - dyn%se_umean(1, e)), abs(s(2) - dyn%se_umean(2, e)))
            mx(6) = max(mx(6), abs(dyn%se_ubt(1, e, 0)), abs(dyn%se_ubt(2, e, 0)))
            if (.not. (all(ieee_is_finite(dyn%se_umean(:, e))) .and. all(ieee_is_finite(dyn%se_ubt(:, e, 0))))) &
                nbad = nbad + 1
        end do
        ! the spreads exchange with the neighbours: every rank calls them (0 without shared elements)
        mx(3) = shared_copy_max_elem(dyn%se_ubt(:, :, 0), partit)
        mx(4) = shared_copy_max_elem(dyn%se_umean, partit)
        sm(4) = real(nbad, WP)
        if (is_multirank(partit)) then
            call allreduce_max(mx, partit)
            call allreduce_sum(sm, partit)
        end if
        if (.not. dyn%se_msl0_set) then
            dyn%se_msl0     = sm(3)/sm(2)
            dyn%se_msl0_set = .true.
            dyn%se_fwmsl    = 0.0_WP
        end if
        if (fw) dyn%se_fwmsl = dyn%se_fwmsl - dt*sm(5)/sm(2)
        vals(SE_CHK_ETACOMP)  = mx(1)
        vals(SE_CHK_TRIM)     = mx(2)
        vals(SE_CHK_MSL)      = sm(1)/sm(2) - dyn%se_msl0 - dyn%se_fwmsl
        vals(SE_CHK_SPREADU)  = mx(3)
        vals(SE_CHK_SPREADUM) = mx(4)
        vals(SE_CHK_MAXETA)   = mx(5)
        vals(SE_CHK_MAXU)     = mx(6)
        vals(SE_CHK_NBAD)     = sm(4)
        vals(SE_CHK_FW)       = dyn%se_fwmsl
    end subroutine se_check_values

    !===========================================================================
    function se_check_line(step, vals) result(line)
        ! The fixed-format SE_CHECK line: blank-separated key=value tokens, values es11.4
        ! without blanks, e.g.
        !   SE_CHECK step=12 etacomp=1.2345E-16 trim=3.0518E-13 msl=-4.1633E-17 spreadU=0.0000E+00
        !            spreadUm=0.0000E+00 maxeta=1.0412E+00 maxU=2.5011E+02
        ! (one line); when which_ALE /= linfs one more token, fw= (SE_CHK_FW, the freshwater
        ! part of the mean-sea-level change, which msl= has already taken out). grep
        ! '^SE_CHECK ', then split each token at '='.
        integer,       intent(in) :: step
        real(kind=WP), intent(in) :: vals(SE_NCHK)
        character(len=256) :: line
        character(len=16)  :: st
        write(st, '(i0)') step
        line = 'SE_CHECK step='//trim(st)//se_kv('etacomp', vals(SE_CHK_ETACOMP))// &
               se_kv('trim', vals(SE_CHK_TRIM))//se_kv('msl', vals(SE_CHK_MSL))// &
               se_kv('spreadU', vals(SE_CHK_SPREADU))//se_kv('spreadUm', vals(SE_CHK_SPREADUM))// &
               se_kv('maxeta', vals(SE_CHK_MAXETA))//se_kv('maxU', vals(SE_CHK_MAXU))
        if (trim(which_ALE) /= 'linfs') line = trim(line)//se_kv('fw', vals(SE_CHK_FW))
    end function se_check_line

    function se_kv(key, x) result(t)
        ! ' key=value' with value in es11.4, no blanks (the SE_CHECK line tokens)
        character(len=*), intent(in) :: key
        real(kind=WP),    intent(in) :: x
        character(len=:), allocatable :: t
        character(len=16) :: buf
        write(buf, '(es11.4)') x
        t = ' '//key//'='//trim(adjustl(buf))
    end function se_kv

    !===========================================================================
    subroutine se_check_core(dyn, mesh, partit, dt, step, eta_old, water_flux)
        ! FESOM3_SE_CHECK, called by se_trim_finish after the commit (helem still at n):
        ! se_check_values, one SE_CHECK line on rank 0 (se_check_line), and a stop with the
        ! step number on any non-finite value. Collective.
        type(t_dyn),    intent(inout) :: dyn
        type(t_mesh),   intent(in)    :: mesh
        type(t_partit), intent(in)    :: partit
        real(kind=WP),  intent(in)    :: dt
        integer,        intent(in)    :: step
        real(kind=WP),  intent(in)    :: eta_old(:)
        real(kind=WP),  intent(in), optional :: water_flux(:)
        real(kind=WP)      :: vals(SE_NCHK)
        character(len=160) :: msg

        call se_check_values(dyn, mesh, partit, dt, eta_old, vals, water_flux)
        if (partit%mype == 0) then
            write(*,'(a)') trim(se_check_line(step, vals))
            flush(output_unit)
        end if
        if (vals(SE_CHK_NBAD) > 0.0_WP) then
            write(msg,'(a,i0,a,i0,a)') 'SE_CHECK: ', nint(vals(SE_CHK_NBAD)), &
                ' non-finite values (eta_n, ssh_rhs_old, <<U>>, U^M or uv) at step ', step, '; stopping'
            call se_die(trim(msg), partit)
        end if
    end subroutine se_check_core

    !===========================================================================
    subroutine se_check_tail_values(dyn, mesh, partit, tvals)
        ! The zstar invariants of FESOM3_SE_CHECK after the ALE tail (step_oce, after
        ! update_thickness_ale): hnode, helem and zbar_3d_n are at n+1, eta_n = hbar = eta^{n+1},
        ! hbar_old = eta^n (se_trim_finish), se_H = H^n and se_c are this step's (from
        ! zbar_3d_n^n). Reduced over the ranks (collective). tvals(SE_TAIL_*):
        !   LAYSUM  max over the owned open-surface nodes of |sum_k hnode(k,v) - (H0_v + eta_v)|
        !           [m], k = 1..nlevels_nod2D(v)-1, H0_v = zbar(1) - zbar(nlevels_nod2D(v)) the
        !           resting depth of the vertex column: the committed thicknesses carry the SE
        !           elevation (vert_vel_ale stretches by hbar - hbar_old = eta^{n+1} - eta^n)
        !   W1      max over the owned open-surface nodes of |w(1,v)| [m/s]. vert_vel_ale sets
        !           w(1) = I_h/area - (hbar - hbar_old)/tau - W, I_h the column inflow of
        !           sum_k helem^n uv; the SE elevation change is tau*(I(<<U>>) - W*areasvol)/
        !           areasvol, area == areasvol (vertex bottom) and the trim makes sum_k helem^n uv
        !           = <<U>>, so w(1) = 0 to rounding, with or without freshwater
        !   HLAW    max over the owned elements of |sum_k helem^{n+1}(k,e) - H_e(eta^{n+1})| [m],
        !           H_e(eta^{n+1}) = H^n_e + sum_v elem_w*c*(hbar - hbar_old), the depth law of
        !           the substeps (plan D3, se_elem_depth) against what the tail built
        !   NBAD    the number of non-finite values in hnode (wet layers) and w(1) (owned open-
        !           surface nodes) and helem (wet layers, owned elements)
        ! Exposed (not only through se_check_tail) so that test_ssh_se can check the values.
        type(t_dyn),    intent(in)  :: dyn
        type(t_mesh),   intent(in)  :: mesh
        type(t_partit), intent(in)  :: partit
        real(kind=WP),  intent(out) :: tvals(SE_NTAIL)
        integer       :: n, e, nz, nbad, elnodes(MAX_NV)
        integer       :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: mx(3), s, HA, nb(1)

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        mx = 0.0_WP; nbad = 0
        do n = 1, nNodO
            if (mesh%ulevels_nod2D(n) /= 1) cycle
            s = 0.0_WP
            do nz = 1, mesh%nlevels_nod2D(n) - 1
                s = s + mesh%hnode(nz, n)
                if (.not. ieee_is_finite(real(mesh%hnode(nz, n), WP))) nbad = nbad + 1
            end do
            mx(1) = max(mx(1), abs(s - ((mesh%zbar(1) - mesh%zbar(mesh%nlevels_nod2D(n))) + dyn%eta_n(n))))
            mx(2) = max(mx(2), abs(dyn%w(1, n)))
            if (.not. ieee_is_finite(dyn%w(1, n))) nbad = nbad + 1
        end do
        do e = 1, nElemO
            s = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                s = s + mesh%helem(nz, e)
                if (.not. ieee_is_finite(real(mesh%helem(nz, e), WP))) nbad = nbad + 1
            end do
            elnodes = mesh%elem2D_nodes(:, e)
            HA = dyn%se_H(e) + sum(mesh%elem_w(:, e)*dyn%se_c(:, e)*(mesh%hbar(elnodes) - mesh%hbar_old(elnodes)))
            mx(3) = max(mx(3), abs(s - HA))
        end do
        nb(1) = real(nbad, WP)
        if (is_multirank(partit)) then
            call allreduce_max(mx, partit)
            call allreduce_sum(nb, partit)
        end if
        tvals(SE_TAIL_LAYSUM) = mx(1)
        tvals(SE_TAIL_W1)     = mx(2)
        tvals(SE_TAIL_HLAW)   = mx(3)
        tvals(SE_TAIL_NBAD)   = nb(1)
    end subroutine se_check_tail_values

    !===========================================================================
    function se_check_tail_line(step, tvals) result(line)
        ! The fixed-format zstar tail line, key=value tokens as se_check_line's, e.g.
        !   SE_CHECK_TAIL step=12 laysum=2.2737E-12 w1=4.2352E-18 hlaw=9.0949E-13
        ! (one line; '^SE_CHECK ' does not match it, so the SE_CHECK parsers are unaffected).
        integer,       intent(in) :: step
        real(kind=WP), intent(in) :: tvals(SE_NTAIL)
        character(len=128) :: line
        character(len=16)  :: st
        write(st, '(i0)') step
        line = 'SE_CHECK_TAIL step='//trim(st)//se_kv('laysum', tvals(SE_TAIL_LAYSUM))// &
               se_kv('w1', tvals(SE_TAIL_W1))//se_kv('hlaw', tvals(SE_TAIL_HLAW))
    end function se_check_tail_line

    !===========================================================================
    subroutine se_check_tail(dyn, mesh, partit, step)
        ! FESOM3_SE_CHECK after the ALE tail (step_oce, after update_thickness_ale). Under zstar
        ! the committed thicknesses against the SE elevation and the depth law
        ! (se_check_tail_values), one SE_CHECK_TAIL line on rank 0, and a stop with the step
        ! number on any non-finite value. Collective. linfs has nothing to check here (hnode is
        ! frozen, w(1) is the free-surface velocity): a no-op.
        type(t_dyn),    intent(in) :: dyn
        type(t_mesh),   intent(in) :: mesh
        type(t_partit), intent(in) :: partit
        integer,        intent(in) :: step
        real(kind=WP)      :: tvals(SE_NTAIL)
        character(len=160) :: msg
        if (trim(which_ALE) == 'linfs') return
        call se_check_tail_values(dyn, mesh, partit, tvals)
        if (partit%mype == 0) then
            write(*,'(a)') trim(se_check_tail_line(step, tvals))
            flush(output_unit)
        end if
        if (tvals(SE_TAIL_NBAD) > 0.0_WP) then
            write(msg,'(a,i0,a,i0,a)') 'SE_CHECK_TAIL: ', nint(tvals(SE_TAIL_NBAD)), &
                ' non-finite values (hnode, w(1) or helem) at step ', step, '; stopping'
            call se_die(trim(msg), partit)
        end if
    end subroutine se_check_tail

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
