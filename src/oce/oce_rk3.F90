module oce_rk3
    ! RK3 time stepping of the ocean, an alternative to the AB2 step (step_oce). Plan:
    ! docs/plans/completed/2026-10-28-rk3.md. The three-stage Wicker-Skamarock Runge-Kutta scheme of Sergey
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
    ! GUARDS (rk3_init): linfs or zstar, no zlevel; double precision only; no cavities; alpha = theta = 1 (the CG corrector of stage 3 is update_vel /
    ! compute_ssh_rhs_ale with theta = 1, alpha = 1). step_oce stops if FESOM3_RK3=1 reaches it
    ! from a driver that did not run rk3_init (rk3_step_guard).
    !
    ! TRACER STAGES (Task 6). The stage tracer lives in valuesAB (T*, T**), values keeps T^n until
    ! the stage-3 solve. rk3_tracer_prep is the scratch part of init_tracers_AB on the STAGE
    ! tracer: del_ttf* zeroed (oce_tra_adv_flux2dtracer only adds) and the elemental gradient
    ! tr_xy of valuesAB (MUSCL and, in stage 3, the horizontal diffusion read it); no AB
    ! interpolation, no history roll (valuesold is not RK3 state). Every stage and every tracer
    ! calls it, stage 3 too. rk3_tracer_stage advances one stage s = 1, 2 of every tracer:
    !     T_s h_s = T^n h^n + tau_s*D_HO(T_{s-1}; u_{s-1}, w_{s-1})
    ! in the identity form of diff_tracers_ale's reconstruct, T_s = T^n + (D + T^n(h^n - h_s))/h_s,
    ! which returns T^n bitwise at zero transport. Advection only: no GM, no diffusion, no
    ! surface flux (plan D8). With the full w of the stage, a uniform tracer stays uniform.
    !
    ! MOMENTUM STAGES (Task 8). dyn%uv keeps u^n until the barotropic block of stage 3; the stage
    ! velocity u*, u** is dyn%rk3_us. rk3_momentum_stage(s) evaluates R_s = tau_s*(Coriolis +
    ! advection + PGF - g grad(eta^n)) on the stage velocity of s - 1 (compute_vel_rhs with
    ! uv_in and no_history) and sets
    !     u_s = u^n + R_s - <R_s> + c_s*dUb,     <X>_e = sum_k helem*X / H_e,  H_e = rk3_H(e)
    ! (rk3.tex trims to the barotropic transport at U^n, c_s = 0). With the CG corrector, a trim
    ! to Ub^n leaves stage 3 the barotropic Coriolis f x Ub^n over tau, forward Euler (energy
    ! growth (f tau)^2 per step for barotropic motions larger than the Rossby radius; the
    ! Weddell instability of 2026-10-30, plan Addendum A). So the CG path trims to the depth-mean
    ! velocity extrapolated in time, the AB-way rk3.tex names: dUb = Ub^n - Ub^{n-1}
    ! (rk3_bt_extrap, Ub = <u>), c_1 = 1/3, c_2 = 1/2 + ab_epsilon; the stage-3 Coriolis then
    ! acts on (3/2 + eps) Ub^n - (1/2 + eps) Ub^{n-1}, AB2's weights. Ub^{n-1} (rk3_ubold) is in
    ! the checkpoint; a cold start or a checkpoint without it starts with dUb = 0. The SE path
    ! keeps c_s = 0: its u^{n+1} is trimmed to the BT solver's U^M. rk3_viscosity takes the
    ! horizontal viscosity of u^n over tau once (stage 1) into rk3_visc; stage 3 adds it to its
    ! right-hand side. rk3_trim moves a velocity to a given transport, depth-uniformly. Every
    ! update is on the owned elements, then the shared copies are synced (FESOM3_SHARED_SYNC, as
    ! update_vel) and the eDim halo exchanged.
    !
    ! ZSTAR STAGE GEOMETRY (Task 18, plan Technical Details §6). Stage s = 1, 2 runs on the geometry
    ! h_{s-1} (h_0 = h^n) and ends on h_s: rk3_implied_eta sets hbar = eta~_s = eta^n + tau_s*
    ! I(sum_k helem*u_{s-1})/areasvol (hbar_old = eta^n since the step's start); vert_vel_ale over
    ! tau_s without water_flux then gives the diasurface w (w(top) = 0) and hnode_new = h_s by the
    ! zstar law; the tracers go T^n h^n -> T_s h_s with the flux thickness h_{s-1};
    ! rk3_stage_geometry(h_s) rebuilds helem, zbar_3d_n, Z_3d_n (update_thickness_ale without
    ! committing hnode, which stays h^n) and rk3_H; the momentum update then keeps sum_k h_s*u_s =
    ! sum_k h_s*u^n. Stage 3 runs on h_2 (PGF, momentum, the CG rhs, the tracer fluxes) and restores
    ! hbar = eta^n before the barotropic block, so compute_hbar_ale / se_rk3_finish see what AB2's
    ! do; its last vert_vel_ale (with W) gives h^{n+1}, which update_thickness_ale commits. Under
    ! linfs none of this changes anything (helem, hnode and hbar stay).
    !
    ! FESOM3_RK3_CHECK (Task 9): one rank-0 line per step,
    !     RK3_CHECK step=<n> trim1= trim2= trim3= etacomp= maxu= nbad= cflh= cflz=   (SE: + trimM=)
    ! trim1, trim2 = max|sum_k helem*u_s - sum_k helem*u^n| after stages 1 and 2 [m^2/s];
    ! trim3 = max|sum_k helem*u_tr - target| of the stage-3 transport velocity (CG: the
    ! transport of u^{n+1}; SE: <<U>>); etacomp = max|eta^{n+1} - eta^n - tau*(I(sum_k helem*u_tr)
    ! - W*areasvol)/areasvol| [m] (W only under zstar), the elevation is the one the tracers'
    ! transport implies; trimM (SE) = max|sum_k helem*u^{n+1} - U^M|; maxu = max|u^{n+1}|; nbad =
    ! the non-finite u^{n+1}, eta^{n+1}, T, S; cflh = max tau*(horizontal outflow)/(areasvol*hnode) of the
    ! stage-3 transport velocity, the horizontal Courant number of the FCT low order (the choice of
    ! FESOM3_RK3_LO_NSUB), and cflz = max CFL_z of its w (the split's Courant number; D7's stability limit
    ! for momentum and non-FCT tracers is a per-cell number of about 9 at maxcfl 1, plan §5).
    ! Owned elements / nodes, reduced over the ranks. With
    ! FESOM3_RK3_CHECK_TOL > 0 a trim or etacomp above it, or nbad > 0, stops the run (the ctest
    ! and run_conserve_pi.sh hard gates); nbad > 0 stops in any case.
    !
    ! The parser (rk3_parse_knob) returns its verdict instead of stopping, so test/test_rk3.F90
    ! checks every rejection; the driver-facing wrappers stop through rk3_die (MPI_Abort, so no
    ! rank is left waiting in a collective).
    use mpi
    use, intrinsic :: iso_fortran_env, only: output_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use mod_precision,   only: WP, MP, MPI_WP
    use mod_mesh,        only: t_mesh
    use mod_dyn,         only: t_dyn
    use mod_tracer,      only: t_tracer
    use mod_partit,      only: t_partit
    use mod_part_bounds, only: local_dims, owned_bounds, is_multirank
    use mod_halo,        only: exchange_nod, exchange_elem_full, exchange_elem, exchange_elem_shared, &
                               allreduce_max
    use oce_tracer_grad, only: tracer_gradient_elements
    use oce_adv_tra_driver, only: do_oce_adv_tra_rk3
    use oce_dyn_velrhs,  only: compute_vel_rhs
    use oce_dyn_visc,    only: viscosity_filter
    use oce_adv_tra_hor, only: adv_tra_hor_upw1
    use mod_param_phys,  only: alpha, theta
    use mod_config,      only: which_ALE, ab_epsilon
    use oce_ssh_se,      only: se_env_requested
    implicit none
    private
    public :: rk3_env_requested, rk3_read_env, rk3_parse_knob, rk3_init, rk3_step_guard, rk3_die
    public :: rk3_tracer_prep, rk3_tracer_stage
    public :: rk3_depth, rk3_transport, rk3_momentum_stage, rk3_viscosity, rk3_trim, rk3_elem_sync
    public :: rk3_momentum_rhs, rk3_momentum_update, rk3_implied_eta, rk3_stage_geometry, rk3_check_eta
    public :: rk3_bt_extrap, rk3_trim_coef
    public :: rk3_check_begin, rk3_check_trim, rk3_check_finish, rk3_check_line, rk3_inflow
    public :: rk3_cold_ab_start
    public :: RK3_NCHK, RK3_CHK_TRIM1, RK3_CHK_TRIM2, RK3_CHK_TRIM3, RK3_CHK_ETACOMP, RK3_CHK_TRIMM, &
              RK3_CHK_MAXU, RK3_CHK_NBAD, RK3_CHK_CFLH, RK3_CHK_CFLZ

    ! FESOM3_RK3_CHECK: the entries of the check values (meaning in the module header); the
    ! running values of the current step
    integer, parameter :: RK3_NCHK = 9
    integer, parameter :: RK3_CHK_TRIM1 = 1, RK3_CHK_TRIM2 = 2, RK3_CHK_TRIM3 = 3, RK3_CHK_ETACOMP = 4, &
                          RK3_CHK_TRIMM = 5, RK3_CHK_MAXU = 6, RK3_CHK_NBAD = 7, RK3_CHK_CFLH = 8, &
                          RK3_CHK_CFLZ = 9
    real(kind=WP), save :: chk(RK3_NCHK) = 0.0_WP

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
        if (trim(which_ALE) /= 'linfs' .and. trim(which_ALE) /= 'zstar') &
            call rk3_die('RK3 supports which_ALE = linfs and zstar, not '//trim(which_ALE), partit)
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
        ! the extrapolated stage trims of the CG path (module header): Ub^{n-1} = huge marks "not
        ! set yet" (cold start, or a checkpoint without rk3_ubold), rk3_bt_extrap then starts with dUb = 0
        allocate(dyn%rk3_ubold(2, nElemL), dyn%rk3_dub(2, nElemL))
        dyn%rk3_ubold = huge(1.0_WP); dyn%rk3_dub = 0.0_WP

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
    subroutine rk3_tracer_prep(tr_num, tracers, mesh, partit)
        ! The per-tracer scratch of a stage (module header): del_ttf, del_ttf_advhoriz and
        ! del_ttf_advvert zeroed over owned + halo nodes, tr_xy (allocated on first use, as in
        ! init_tracers_AB) = the elemental gradient of valuesAB on the owned elements, exchanged
        ! over the full element halo.
        integer,        intent(in)           :: tr_num
        type(t_tracer), intent(inout)        :: tracers
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: n, nz, nElemA
        integer :: nNodO, nNodL, nEdgeO, nElemO

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        do n = 1, nNodL
            do nz = 1, mesh%nl-1
                tracers%work%del_ttf         (nz, n) = 0.0_WP
                tracers%work%del_ttf_advhoriz(nz, n) = 0.0_WP
                tracers%work%del_ttf_advvert (nz, n) = 0.0_WP
            end do
        end do
        nElemA = mesh%elem2D
        if (is_multirank(partit)) &
            nElemA = partit%myDim_elem2D + partit%eDim_elem2D + partit%eXDim_elem2D
        if (.not. allocated(tracers%work%tr_xy)) then
            allocate(tracers%work%tr_xy(2, mesh%nl-1, nElemA))
            tracers%work%tr_xy = 0.0_WP
        end if
        call tracer_gradient_elements(tracers%data(tr_num)%valuesAB, tracers%work%tr_xy, mesh, partit)
        if (is_multirank(partit)) call exchange_elem_full(tracers%work%tr_xy, partit)
    end subroutine rk3_tracer_prep

    !===========================================================================
    subroutine rk3_tracer_stage(stage, tau_s, vel, w, dynamics, tracers, mesh, partit)
        ! Stage s = 1, 2 of every tracer (module header): valuesAB := T^n first (stage 1), then
        ! per tracer rk3_tracer_prep, the high order of valuesAB (do_oce_adv_tra_rk3) with the
        ! stage velocity vel and the full vertical velocity w of the stage (plan D7: no split
        ! before stage 3), the stage update on the owned nodes and exchange_nod(valuesAB).
        integer,        intent(in)            :: stage
        real(kind=WP),  intent(in)            :: tau_s
        real(kind=WP),  intent(in)            :: vel(:,:,:)
        real(kind=WP),  intent(in)            :: w(:,:)
        type(t_dyn),    intent(inout), target :: dynamics
        type(t_tracer), intent(inout), target :: tracers
        type(t_mesh),   intent(in)            :: mesh
        type(t_partit), intent(in), optional  :: partit
        integer :: tr_num, n, nzmin, nzmax
        integer :: nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP), pointer :: T0(:,:), Ts(:,:)

        if (stage /= 1 .and. stage /= 2) call rk3_die('rk3_tracer_stage: stage must be 1 or 2', partit)
        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        do tr_num = 1, tracers%num_tracers
            T0 => tracers%data(tr_num)%values
            Ts => tracers%data(tr_num)%valuesAB
            if (stage == 1) then
                do n = 1, nNodL
                    Ts(:, n) = T0(:, n)
                end do
            end if
            call rk3_tracer_prep(tr_num, tracers, mesh, partit)
            call do_oce_adv_tra_rk3(tau_s, vel, w, dynamics%w_i, w, tr_num, stage, 1, &
                                    dynamics, tracers, mesh, partit)
            do n = 1, nNodO
                nzmax = mesh%nlevels_nod2D(n) - 1
                nzmin = mesh%ulevels_nod2D(n)
                tracers%work%del_ttf(nzmin:nzmax,n) = tracers%work%del_ttf_advhoriz(nzmin:nzmax,n) &
                                                    + tracers%work%del_ttf_advvert(nzmin:nzmax,n)
                tracers%work%del_ttf(nzmin:nzmax,n) = tracers%work%del_ttf(nzmin:nzmax,n) + T0(nzmin:nzmax,n)* &
                                                      (mesh%hnode(nzmin:nzmax,n)-mesh%hnode_new(nzmin:nzmax,n))
                Ts(nzmin:nzmax,n) = T0(nzmin:nzmax,n) + &
                                    tracers%work%del_ttf(nzmin:nzmax,n)/mesh%hnode_new(nzmin:nzmax,n)
            end do
            if (is_multirank(partit)) call exchange_nod(Ts, partit)
        end do
    end subroutine rk3_tracer_stage

    !===========================================================================
    subroutine rk3_depth(dyn, mesh, partit)
        ! rk3_H(e) = sum_k helem over ulevels(e)..nlevels(e)-1 on the owned + eDim elements (the
        ! loop of se_elem_depth). Every step: a restart rebuilds helem after rk3_init.
        type(t_dyn),    intent(inout)        :: dyn
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: e, nz
        integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
        real(kind=WP) :: H

        call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
        do e = 1, nElemL
            H = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                H = H + mesh%helem(nz, e)
            end do
            dyn%rk3_H(e) = H
        end do
    end subroutine rk3_depth

    !===========================================================================
    subroutine rk3_transport(vel, mesh, nelem, out)
        ! out(:, e) = sum_k helem*vel(:, k, e) on the elements 1..nelem
        real(kind=WP),  intent(in)  :: vel(:,:,:)
        type(t_mesh),   intent(in)  :: mesh
        integer,        intent(in)  :: nelem
        real(kind=WP),  intent(out) :: out(:,:)
        integer :: e, nz
        real(kind=WP) :: s(2)
        do e = 1, nelem
            s = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                s(1) = s(1) + mesh%helem(nz, e)*vel(1, nz, e)
                s(2) = s(2) + mesh%helem(nz, e)*vel(2, nz, e)
            end do
            out(:, e) = s
        end do
    end subroutine rk3_transport

    !===========================================================================
    subroutine rk3_elem_sync(vel, partit)
        ! the shared-element sync of update_vel (FESOM3_SHARED_SYNC) and the eDim halo exchange
        real(kind=WP),  intent(inout)        :: vel(:,:,:)
        type(t_partit), intent(in), optional :: partit
        if (.not. is_multirank(partit)) return
        if (partit%sync_shared_elem) call exchange_elem_shared(vel, partit)
        call exchange_elem(vel, partit)
    end subroutine rk3_elem_sync

    !===========================================================================
    subroutine rk3_momentum_stage(stage, tau_s, dynamics, mesh, partit)
        ! Stage s = 1, 2 of the momentum (module header): rk3_momentum_rhs, then
        ! rk3_momentum_update on the same geometry (linfs; under zstar the step puts the tracers
        ! and the stage geometry between the two)
        integer,        intent(in)            :: stage
        real(kind=WP),  intent(in)            :: tau_s
        type(t_dyn),    intent(inout), target :: dynamics
        type(t_mesh),   intent(in),    target :: mesh
        type(t_partit), intent(in), optional  :: partit
        call rk3_momentum_rhs(stage, tau_s, dynamics, mesh, partit)
        call rk3_momentum_update(dynamics, mesh, partit)
    end subroutine rk3_momentum_stage

    !===========================================================================
    subroutine rk3_momentum_rhs(stage, tau_s, dynamics, mesh, partit)
        ! R_s from compute_vel_rhs on u^n (s = 1) or u* (s = 2, dyn%rk3_us) into uv_rhs (owned
        ! elements), on the current geometry. compute_vel_rhs reads w_e: the stage's
        ! vert_vel_ale with the split off puts the full w there (w_e = w, w_i = 0).
        integer,        intent(in)            :: stage
        real(kind=WP),  intent(in)            :: tau_s
        type(t_dyn),    intent(inout), target :: dynamics
        type(t_mesh),   intent(in),    target :: mesh
        type(t_partit), intent(in), optional  :: partit
        select case (stage)
        case (1)
            call compute_vel_rhs(dynamics, mesh, tau_s, .true., partit, no_history=.true.)
        case (2)
            call compute_vel_rhs(dynamics, mesh, tau_s, .true., partit, uv_in=dynamics%rk3_us, no_history=.true.)
        case default
            call rk3_die('rk3_momentum_rhs: stage must be 1 or 2', partit)
        end select
    end subroutine rk3_momentum_rhs

    !===========================================================================
    subroutine rk3_momentum_update(dynamics, mesh, partit, cs)
        ! rk3_us = u^n + R_s - <R_s> + cs*rk3_dub on the owned elements, <.> with the CURRENT helem
        ! and rk3_H (zstar: h_s, so that sum_k h_s*u_s = sum_k h_s*(u^n + cs*dUb)), synced and
        ! exchanged. cs = the stage's coefficient of the extrapolated trim (rk3_trim_coef; absent = 0,
        ! the trim of rk3.tex to the transport of u^n)
        type(t_dyn),    intent(inout), target :: dynamics
        type(t_mesh),   intent(in),    target :: mesh
        type(t_partit), intent(in), optional  :: partit
        real(kind=WP),  intent(in), optional  :: cs
        integer :: e, nz
        integer :: nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP) :: s(2), c

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        c = 0.0_WP
        if (present(cs)) c = cs
        do e = 1, nElemO
            s = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                s(1) = s(1) + mesh%helem(nz, e)*dynamics%uv_rhs(1, nz, e)
                s(2) = s(2) + mesh%helem(nz, e)*dynamics%uv_rhs(2, nz, e)
            end do
            s = s/dynamics%rk3_H(e)
            if (c /= 0.0_WP) s = s - c*dynamics%rk3_dub(:, e)
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                dynamics%rk3_us(1, nz, e) = dynamics%uv(1, nz, e) + (dynamics%uv_rhs(1, nz, e) - s(1))
                dynamics%rk3_us(2, nz, e) = dynamics%uv(2, nz, e) + (dynamics%uv_rhs(2, nz, e) - s(2))
            end do
        end do
        call rk3_elem_sync(dynamics%rk3_us, partit)
    end subroutine rk3_momentum_update

    !===========================================================================
    real(kind=WP) function rk3_trim_coef(stage, dynamics)
        ! c_s of the extrapolated stage trim (module header): CG c_1 = 1/3, c_2 = 1/2 + ab_epsilon;
        ! SE 0 (rk3.tex: its u^{n+1} is trimmed to the BT solver's U^M)
        integer,     intent(in) :: stage
        type(t_dyn), intent(in) :: dynamics
        rk3_trim_coef = 0.0_WP
        if (dynamics%use_ssh_se) return
        select case (stage)
        case (1)
            rk3_trim_coef = 1.0_WP/3.0_WP
        case (2)
            rk3_trim_coef = 0.5_WP + ab_epsilon
        end select
    end function rk3_trim_coef

    !===========================================================================
    subroutine rk3_bt_extrap(dyn, mesh, partit)
        ! Start of a CG step, u^n in uv, h^n in helem, rk3_H = H^n: rk3_dub = Ub^n - Ub^{n-1} with
        ! Ub = sum_k helem*uv / rk3_H, then rk3_ubold = Ub^n for the next step, on the owned
        ! elements (the trims use the owned ones only). An element whose rk3_ubold still holds
        ! huge (rk3_init: cold start, or a checkpoint without rk3_ubold) gets dUb = 0. The owners of a
        ! shared element hold the same synced uv and helem, so the same dUb.
        type(t_dyn),    intent(inout)        :: dyn
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: e, nz
        integer :: nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP) :: ub(2)

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        do e = 1, nElemO
            ub = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                ub(1) = ub(1) + mesh%helem(nz, e)*dyn%uv(1, nz, e)
                ub(2) = ub(2) + mesh%helem(nz, e)*dyn%uv(2, nz, e)
            end do
            ub = ub/dyn%rk3_H(e)
            if (dyn%rk3_ubold(1, e) == huge(1.0_WP)) dyn%rk3_ubold(:, e) = ub
            dyn%rk3_dub(:, e) = ub - dyn%rk3_ubold(:, e)
            dyn%rk3_ubold(:, e) = ub
        end do
    end subroutine rk3_bt_extrap

    !===========================================================================
    subroutine rk3_implied_eta(tau_s, vel, dyn, mesh, partit)
        ! zstar: hbar = eta~_s = eta^n + tau_s*I(sum_k helem*vel)/areasvol on the owned open-surface
        ! nodes (eta^n = dyn%eta_n, the elevation at the step's start), then exchange_nod: the
        ! elevation the stage's transport implies, without the freshwater (W enters in stage 3)
        real(kind=WP),  intent(in)           :: tau_s
        real(kind=WP),  intent(in)           :: vel(:,:,:)
        type(t_dyn),    intent(in)           :: dyn
        type(t_mesh),   intent(inout)        :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: n, nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP), allocatable :: inflow(:)
        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        allocate(inflow(nNodL))
        call rk3_inflow(vel, mesh, partit, inflow)
        do n = 1, nNodO
            if (mesh%ulevels_nod2D(n) > 1) cycle
            mesh%hbar(n) = dyn%eta_n(n) + tau_s*inflow(n)/mesh%areasvol(n)
        end do
        if (is_multirank(partit)) call exchange_nod(mesh%hbar, partit)
    end subroutine rk3_implied_eta

    !===========================================================================
    subroutine rk3_stage_geometry(dyn, mesh, partit)
        ! zstar: the geometry of the stage thickness h_s = hnode_new (module header):
        ! update_thickness_ale's rebuild without committing hnode -- zbar_3d_n and Z_3d_n bottom-up
        ! from the fixed anchor over the owned + halo nodes (hnode_new is halo-valid from
        ! vert_vel_ale), helem = sum_v elem_w*hnode_new over the owned elements' full range and
        ! exchange_elem, then rk3_H. A no-op under linfs.
        type(t_dyn),    intent(inout)        :: dyn
        type(t_mesh),   intent(inout)        :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: n, nz, elem, nzmin, nzmax
        integer :: nNodO, nNodL, nEdgeO, nElemO
        if (trim(which_ALE) /= 'zstar') return
        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        do n = 1, nNodL
            nzmin = mesh%ulevels_nod2D(n)
            nzmax = mesh%nlevels_nod2D(n) - 2
            if (nzmin > 1) cycle
            do nz = nzmax, nzmin, -1
                mesh%zbar_3d_n(nz,n) = mesh%zbar_3d_n(nz+1,n) + mesh%hnode_new(nz,n)
                mesh%Z_3d_n(nz,n)    = mesh%zbar_3d_n(nz+1,n) + mesh%hnode_new(nz,n)/2.0_WP
            end do
        end do
        do elem = 1, nElemO
            nzmin = mesh%ulevels(elem)
            nzmax = mesh%nlevels(elem) - 1
            if (nzmin > 1) cycle
            do nz = nzmin, nzmax
                mesh%helem(nz,elem) = sum(mesh%elem_w(:,elem)*mesh%hnode_new(nz,mesh%elem2D_nodes(:,elem)))
            end do
        end do
        if (is_multirank(partit)) call exchange_elem(mesh%helem, partit)
        call rk3_depth(dyn, mesh, partit)
    end subroutine rk3_stage_geometry

    !===========================================================================
    subroutine rk3_viscosity(dynamics, mesh, tau, partit)
        ! rk3_visc = the horizontal viscosity increment of u^n over tau (viscosity_filter adds
        ! into uv_rhs, which starts at 0 here); stage 1, kept for stage 3 (and the SE forcing)
        type(t_dyn),    intent(inout), target :: dynamics
        type(t_mesh),   intent(in),    target :: mesh
        real(kind=WP),  intent(in)            :: tau
        type(t_partit), intent(in), optional  :: partit
        dynamics%uv_rhs = 0.0_WP
        call viscosity_filter(dynamics%opt_visc, dynamics, mesh, tau, partit)
        dynamics%rk3_visc = dynamics%uv_rhs
    end subroutine rk3_viscosity

    !===========================================================================
    subroutine rk3_trim(vel, target, dynamics, mesh, partit)
        ! vel(:, k, e) += (target(:, e) - sum_k helem*vel)/H_e on the owned elements (rk3_H),
        ! then synced and exchanged: the depth-uniform change that gives vel the transport target
        real(kind=WP),  intent(inout)        :: vel(:,:,:)
        real(kind=WP),  intent(in)           :: target(:,:)
        type(t_dyn),    intent(in)           :: dynamics
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: e, nz
        integer :: nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP) :: s(2)

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        do e = 1, nElemO
            s = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                s(1) = s(1) + mesh%helem(nz, e)*vel(1, nz, e)
                s(2) = s(2) + mesh%helem(nz, e)*vel(2, nz, e)
            end do
            s = (target(:, e) - s)/dynamics%rk3_H(e)
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                vel(1, nz, e) = vel(1, nz, e) + s(1)
                vel(2, nz, e) = vel(2, nz, e) + s(2)
            end do
        end do
        call rk3_elem_sync(vel, partit)
    end subroutine rk3_trim

    !===========================================================================
    subroutine rk3_check_begin(dyn, mesh, partit)
        ! FESOM3_RK3_CHECK, start of a step: the values zeroed, rk3_sbar0 = sum_k helem*u^n
        type(t_dyn),    intent(inout)        :: dyn
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: nNodO, nNodL, nEdgeO, nElemO
        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        chk = 0.0_WP
        call rk3_transport(dyn%uv, mesh, nElemO, dyn%rk3_sbar0)
    end subroutine rk3_check_begin

    !===========================================================================
    subroutine rk3_check_trim(k, vel, dyn, mesh, partit, target)
        ! chk(k) = max over the owned elements of |sum_k helem*vel - target| (target absent:
        ! rk3_sbar0, the transport of u^n); local, reduced in rk3_check_finish
        integer,        intent(in)           :: k
        real(kind=WP),  intent(in)           :: vel(:,:,:)
        type(t_dyn),    intent(in)           :: dyn
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in), optional :: partit
        real(kind=WP),  intent(in), optional :: target(:,:)
        integer :: e, nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP), allocatable :: s(:,:)
        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        allocate(s(2, nElemO))
        call rk3_transport(vel, mesh, nElemO, s)
        do e = 1, nElemO
            if (present(target)) then
                chk(k) = max(chk(k), maxval(abs(s(:, e) - target(:, e))))
            else
                chk(k) = max(chk(k), maxval(abs(s(:, e) - dyn%rk3_sbar0(:, e))))
            end if
        end do
    end subroutine rk3_check_trim

    !===========================================================================
    subroutine rk3_inflow(vel, mesh, partit, out)
        ! out = I(sum_k helem*vel): the edge scatter of compute_hbar_ale (positive = convergence)
        ! over the owned edges into owned + halo nodes; complete on the owned nodes
        real(kind=WP),  intent(in)           :: vel(:,:,:)
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in), optional :: partit
        real(kind=WP),  intent(out)          :: out(:)
        integer :: ed, el(2), enodes(2), nz, nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP) :: c1, c2
        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        out(1:nNodL) = 0.0_WP
        do ed = 1, nEdgeO
            enodes = mesh%edges(:, ed)
            el     = mesh%edge_tri(:, ed)
            c1 = 0.0_WP
            do nz = mesh%ulevels(el(1)), mesh%nlevels(el(1)) - 1
                c1 = c1 + (vel(2,nz,el(1))*mesh%edge_cross_dxdy(1,ed) - vel(1,nz,el(1))*mesh%edge_cross_dxdy(2,ed)) &
                        *mesh%helem(nz,el(1))
            end do
            c2 = 0.0_WP
            if (el(2) /= el(1)) then
                do nz = mesh%ulevels(el(2)), mesh%nlevels(el(2)) - 1
                    c2 = c2 - (vel(2,nz,el(2))*mesh%edge_cross_dxdy(3,ed) - vel(1,nz,el(2))*mesh%edge_cross_dxdy(4,ed)) &
                            *mesh%helem(nz,el(2))
                end do
            end if
            out(enodes(1)) = out(enodes(1)) + (c1 + c2)
            out(enodes(2)) = out(enodes(2)) - (c1 + c2)
        end do
    end subroutine rk3_inflow

    !===========================================================================
    subroutine rk3_check_eta(dt, eta_old, vel_tr, dyn, mesh, partit, water_flux)
        ! FESOM3_RK3_CHECK, stage 3 after the last vert_vel_ale (the geometry of the transport,
        ! h_2 under zstar, is still in place): etacomp of eta^{n+1} = eta_n against eta_old = eta^n
        ! (owned + halo) and the transport velocity vel_tr, cflh of vel_tr and cflz of its w (module
        ! header); water_flux = W (non-linfs only). Local values, reduced in rk3_check_finish.
        real(kind=WP),  intent(in)           :: dt
        real(kind=WP),  intent(in)           :: eta_old(:)
        real(kind=WP),  intent(in)           :: vel_tr(:,:,:)
        type(t_dyn),    intent(in)           :: dyn
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in)           :: partit
        real(kind=WP),  intent(in), optional :: water_flux(:)
        integer :: n, e, nz, en(2), nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP) :: res, fw
        real(kind=WP), allocatable :: inflow(:), one(:,:), fl(:,:), out(:,:)

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        allocate(inflow(nNodL))
        call rk3_inflow(vel_tr, mesh, partit, inflow)
        do n = 1, nNodO
            fw = 0.0_WP
            if (trim(which_ALE) /= 'linfs' .and. present(water_flux)) fw = water_flux(n)*mesh%areasvol(n)
            res = dyn%eta_n(n) - eta_old(n) - dt*(inflow(n) - fw)/mesh%areasvol(n)
            chk(RK3_CHK_ETACOMP) = max(chk(RK3_CHK_ETACOMP), abs(res))
        end do
        ! the Courant numbers: the horizontal outflow of the stage-3 transport (the upwind flux of T = 1
        ! is the transport into an edge's first node) and the CFL_z of its w
        allocate(one(mesh%nl-1, nNodL), fl(mesh%nl-1, max(nEdgeO, 1)), out(mesh%nl-1, nNodL))
        one = 1.0_WP; out = 0.0_WP
        call adv_tra_hor_upw1(vel_tr, one, mesh, fl, o_init_zero=.true., partit=partit)
        do e = 1, nEdgeO
            en = mesh%edges(:, e)
            out(:, en(1)) = out(:, en(1)) + max(0.0_WP, -fl(:, e))
            out(:, en(2)) = out(:, en(2)) + max(0.0_WP,  fl(:, e))
        end do
        do n = 1, nNodO
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                chk(RK3_CHK_CFLH) = max(chk(RK3_CHK_CFLH), dt*out(nz, n)/(mesh%areasvol(n)*mesh%hnode(nz, n)))
            end do
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n)
                chk(RK3_CHK_CFLZ) = max(chk(RK3_CHK_CFLZ), dyn%cfl_z(nz, n))
            end do
        end do
    end subroutine rk3_check_eta

    !===========================================================================
    subroutine rk3_check_finish(step, dyn, tracers, mesh, partit)
        ! FESOM3_RK3_CHECK, end of a step: maxu and nbad, the reductions, the rank-0 line and the
        ! stops (module header)
        integer,        intent(in) :: step
        type(t_dyn),    intent(in) :: dyn
        type(t_tracer), intent(in) :: tracers
        type(t_mesh),   intent(in) :: mesh
        type(t_partit), intent(in) :: partit
        integer :: n, e, nz, t, ierr, nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP) :: res
        logical :: bad
        character(len=320) :: line

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        do n = 1, nNodO
            if (.not. ieee_is_finite(dyn%eta_n(n))) chk(RK3_CHK_NBAD) = chk(RK3_CHK_NBAD) + 1.0_WP
            do t = 1, tracers%num_tracers
                do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                    if (.not. ieee_is_finite(tracers%data(t)%values(nz, n))) chk(RK3_CHK_NBAD) = chk(RK3_CHK_NBAD) + 1.0_WP
                end do
            end do
        end do
        do e = 1, nElemO
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                if (.not. (ieee_is_finite(dyn%uv(1, nz, e)) .and. ieee_is_finite(dyn%uv(2, nz, e)))) then
                    chk(RK3_CHK_NBAD) = chk(RK3_CHK_NBAD) + 1.0_WP
                else
                    chk(RK3_CHK_MAXU) = max(chk(RK3_CHK_MAXU), abs(dyn%uv(1, nz, e)), abs(dyn%uv(2, nz, e)))
                end if
            end do
        end do
        ! nbad is a count: reduce it as a sum (the others as a max)
        res = chk(RK3_CHK_NBAD)
        if (partit%npes > 1) then
            call allreduce_max(chk, partit)
            call MPI_Allreduce(MPI_IN_PLACE, res, 1, MPI_WP, MPI_SUM, partit%MPI_COMM_FESOM, ierr)
        end if
        chk(RK3_CHK_NBAD) = res
        line = rk3_check_line(step, chk, dyn%use_ssh_se)
        if (partit%mype == 0) write(*,'(a)') trim(line)
        bad = (chk(RK3_CHK_NBAD) > 0.0_WP)
        if (dyn%rk3_check_tol > 0.0_WP) then
            bad = bad .or. chk(RK3_CHK_TRIM1) > dyn%rk3_check_tol .or. chk(RK3_CHK_TRIM2) > dyn%rk3_check_tol &
                      .or. chk(RK3_CHK_TRIM3) > dyn%rk3_check_tol .or. chk(RK3_CHK_ETACOMP) > dyn%rk3_check_tol
            if (dyn%use_ssh_se) bad = bad .or. chk(RK3_CHK_TRIMM) > dyn%rk3_check_tol
        end if
        if (bad) call rk3_die('RK3_CHECK VIOLATED (FESOM3_RK3_CHECK_TOL or a non-finite value): '//trim(line), partit)
    end subroutine rk3_check_finish

    !===========================================================================
    function rk3_check_line(step, vals, se) result(line)
        integer,       intent(in) :: step
        real(kind=WP), intent(in) :: vals(RK3_NCHK)
        logical,       intent(in) :: se
        character(len=320) :: line
        character(len=24)  :: v(RK3_NCHK)
        integer :: k
        do k = 1, RK3_NCHK
            write(v(k), '(es12.4)') vals(k)
        end do
        write(line, '(a,i0,8(a,a))') 'RK3_CHECK step=', step, ' trim1=', trim(adjustl(v(1))), &
            ' trim2=', trim(adjustl(v(2))), ' trim3=', trim(adjustl(v(3))), ' etacomp=', trim(adjustl(v(4))), &
            ' maxu=', trim(adjustl(v(6))), ' nbad=', trim(adjustl(v(7))), ' cflh=', trim(adjustl(v(8))), &
            ' cflz=', trim(adjustl(v(9)))
        if (se) line = trim(line)//' trimM='//trim(adjustl(v(5)))
    end function rk3_check_line

    !===========================================================================
    subroutine rk3_cold_ab_start(dyn, tracers, partit)
        ! An AB2 run that resumes an RK3 checkpoint (plan D11, Task 16): the AB history in the
        ! checkpoint is not AB state (RK3 never rolls valuesold, and uv_rhsAB is its scratch), so the
        ! resumed run starts AB2 cold, as a cold run does: uv_rhsAB = 0, valuesold(k) = values for
        ! every history level, and with split-explicit se_sbar_old = 0 (lfirst sets only a2 = 1; the
        ! a1*se_sbar_old term of se_bt_forcing stays, so the cold start depends on that zero). The
        ! caller also takes lfirst on its first step. One rank-0 line.
        type(t_dyn),    intent(inout) :: dyn
        type(t_tracer), intent(inout) :: tracers
        type(t_partit), intent(in)    :: partit
        integer :: t, k
        dyn%uv_rhsAB = 0.0_WP
        do t = 1, tracers%num_tracers
            do k = 1, size(tracers%data(t)%valuesold, 1)
                tracers%data(t)%valuesold(k, :, :) = tracers%data(t)%values
            end do
        end do
        if (allocated(dyn%se_sbar_old)) dyn%se_sbar_old = 0.0_WP
        if (partit%mype == 0) write(*,'(a)') 'restart: RK3 checkpoint resumed by an AB2 run: cold AB start ' // &
            '(Euler first step, uv_rhsAB = 0, valuesold = values, se_sbar_old = 0)'
    end subroutine rk3_cold_ab_start

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
