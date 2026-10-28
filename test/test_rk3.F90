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
    !   R1c vert_vel_ale's hooks (Task 4): uv_in = a copy equals the default bitwise (w, w_e, w_i,
    !       cfl_z on owned + halo nodes), GM off and on (fer_w too); uv_in = u2 with dyn%uv = u1
    !       equals the default with dyn%uv = u2; bolus=.false. leaves fer_w untouched and w,
    !       w_e, w_i as the default
    !   R2  split=.false. (RK3 stages 1-2, plan D7): use_wsplit on with a cap so low that split
    !       faces exist in the default call (precondition asserted: owned faces with w_i /= 0);
    !       the default call keeps w_e + w_i = w to one spacing; split=.false. gives w bitwise
    !       the default's, w_e == w bitwise and w_i == 0 on every owned + halo face
    !   P   pressure_bv(pressure_only=.true.) (Task 5): density_m_rho0 and hpressure equal the
    !       full call's bitwise on owned + halo nodes (the JM EOS on a hashed T/S); bvfreq,
    !       bvfreq_raw and dbsfc are not touched (a sentinel survives)
    !   R3  the momentum stages (rk3_momentum_stage, Task 8), momadv 1, use_wsplit on, stage 1
    !       on u^n and stage 2 on u*: the stage transport sum_k helem*u_s equals sum_k helem*u^n
    !       to 1e-14 of sum_k helem*|u|; u_s - (u^n + R_s) is depth-uniform (1e-15 of max|u|);
    !       the copies of a shared element agree (np 2); rk3_trim to a hashed target gives that
    !       transport (1e-14) by a depth-uniform change; rk3_viscosity == the viscosity_filter
    !       increment of u^n from zero
    !   R4  temporal order: momadv off, eta = 0, pgf = 0 (Coriolis alone), a baroclinic u^n (zero
    !       depth mean, so the trims are idle): stage 1, stage 2 (rk3_momentum_stage) and the
    !       stage-3 rhs on u** give, per element and layer, c^{n+1} = A(z) c^n with c = u + iv,
    !       z = -i f tau, A = 1 + z + z^2/2 + z^3/6 (relative 1e-12, modulus and phase);
    !       positive control: one Euler step misses A by more than 1e-3
    !   R5  a high-order stage (rk3_tracer_stage, Task 6), linfs, use_wsplit on but the stage's
    !       full w (vert_vel_ale split=.false.), T FCT and S not (both take the HO path in
    !       stages 1-2): a uniform T0 stays T0 to 1e-13 relative in stage 1 (u^n) and stage 2
    !       (another velocity), w(top) of both signs (precondition asserted); open surface: the
    !       content change of a hashed tracer is exactly tau_s times the top flux
    !       -sum area*w(top)*T(top) (1e-13 of the content); closed surface (the depth mean
    !       removed from u, so w(top) ~ 0): the content is conserved to 1e-13
    !   R6  the stage-3 substep driver at N = 1 (do_oce_adv_tra_rk3, Task 7): fct_LO and
    !       del_ttf_advhoriz/advvert equal do_oce_adv_tra's bitwise on the same valuesAB (T**
    !       /= T^n) and tr_xy, use_wsplit on with split faces (FCT tracer); a non-FCT tracer is
    !       delegated to do_oce_adv_tra (bitwise too)
    !   R7  the low order in N = 1, 2, 3 substeps with the split (cap low enough for split faces):
    !       a uniform T0 (T^n = T** = T0) gives fct_LO = T0 and T^{n+1} = T0 to 1e-13 relative;
    !       closed surface (depth mean removed): the content of T^{n+1} is conserved to 1e-13;
    !       open surface: its change is tau*sum(adv_flux_ver(top) - the substep-mean low-order
    !       surface transport rk3_lo_top), to 1e-13 of the content
    !   R8  low-order boundedness at a horizontal Courant number of 1.5 (the velocity scaled to
    !       it; the cap keeps the explicit vertical part small): N = 2 keeps fct_LO within the
    !       global [min, max] of T^n (1e-12 of the range); positive control: N = 1 leaves it by
    !       more than 1e-3 of the range
    !   R9  stage-3 FCT for N = 2, 3 on a hashed field: T^{n+1} - fct_LO within the admissible
    !       increments [fct_ttf_min, fct_ttf_max] of the limiter (1e-12 of the range), the
    !       limiter active somewhere (a factor < 1). On a smooth field at a Courant number of
    !       0.01 (depth-uniform, linear in the 3-D position: no interior extremum) the limiter
    !       still clips at the coastal extrema of the global pi, so "no flux is limited" holds
    !       on the CLEAN cells only: both factors == 1 at the cell, at its edge neighbours on the
    !       level and at the cells above and below, i.e. every flux into the cell unlimited
    !       (asserted: >= 90 % of the wet cells). There T^{n+1} equals fct_LO + the scatter of
    !       the full antidiffusive flux HO(T**) - the substep-mean low-order flux
    !       (rk3_lo_fh/fv), the HO recomputed here from the kernels, to 1e-13; positive
    !       control: with the low-order flux of ONE step from T^n in place of the substep mean,
    !       the clean cells differ by more than 1e-9
    !   R10t the stage-3 tracer solve at rest (Task 10): u = 0, w = 0, one LO step: stages 1-2
    !       leave T** = T^n bitwise (the identity form of the stage update) and
    !       solve_tracers_rk3 equals solve_tracers_ale bitwise (owned + halo), with hashed Kv,
    !       Ki and surface fluxes (horizontal diffusion, TDMA, surface fluxes)
    !   R11 three-stage constancy with the split on (Task 10): a uniform T0 in T (FCT) and S
    !       (not) through stage 1 (u^n, full w), stage 2 (u*, full w) and solve_tracers_rk3 on
    !       u** (split: w_e explicit, w_i in the LO / the TDMA; split faces asserted), Kv /= 0,
    !       open surface: T^{n+1} = T0 to 1e-13 relative. Positive control (the rule D7
    !       dropped): w_e in stages 1-2; the non-FCT defect is O(tau^2), its ratio between tau and
    !       tau/2 (the cap halved with tau, so the split fraction is the same) is 4 +- 1
    use mpi
    use mod_precision,    only: WP, MP, WP_full
    use mod_constants,    only: density_0
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init
    use mod_dyn,          only: t_dyn
    use mod_part_bounds,  only: local_dims
    use mod_param_phys,   only: rvo_upwind, Fer_GM, N2smth_h
    use oce_rk3,          only: rk3_env_requested, rk3_parse_knob, rk3_tracer_stage, rk3_tracer_prep, &
                                rk3_depth, rk3_transport, rk3_momentum_stage, rk3_viscosity, rk3_trim
    use oce_dyn_visc,     only: viscosity_filter
    use oce_ale_tracer,   only: solve_tracers_ale, solve_tracers_rk3
    use mod_halo,         only: shared_copy_max_elem
    use mod_tracer,       only: t_tracer
    use oce_muscl_adv,    only: muscl_adv_init
    use oce_adv_tra_driver, only: do_oce_adv_tra, do_oce_adv_tra_rk3
    use oce_adv_tra_hor,  only: adv_tra_hor_upw1, adv_tra_hor_muscl
    use oce_adv_tra_ver,  only: adv_tra_ver_qr4c, adv_tra_ver_upw1
    use oce_adv_tra_flux, only: oce_tra_adv_flux2dtracer
    use oce_dyn_velrhs,   only: compute_vel_rhs
    use oce_ale,          only: vert_vel_ale
    use oce_pressure_bv,  only: pressure_bv
    use test_util,        only: setup_test_mesh, check, finish, sum_all, max_all, hval
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
    call r1c_vert_vel_hooks()
    call r2_split_off()
    call p_pressure_only()
    call r3_momentum_stage()
    call r4_inertial()
    call r5_ho_stage()
    call r6_substep_n1()
    call r10t_rest()
    call r11_constancy()
    call r7_substeps()
    call r8_lo_bounded()
    call r9_fct_stage3()
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

    subroutine new_tracers(tr)
        ! T (FCT) and S (not FCT), MUSCL + QR4C, the work arrays of fesom_conserve
        type(t_tracer), intent(inout) :: tr
        integer :: t
        tr%num_tracers = 2
        allocate(tr%data(2))
        do t = 1, 2
            allocate(tr%data(t)%values(nl-1, nNodL), tr%data(t)%valuesAB(nl-1, nNodL))
            allocate(tr%data(t)%valuesold(2, nl-1, nNodL))
            tr%data(t)%values = 0.0_WP; tr%data(t)%valuesAB = 0.0_WP; tr%data(t)%valuesold = 0.0_WP
            tr%data(t)%ID = t
            tr%data(t)%AB_order    = 2
            tr%data(t)%tra_adv_hor = 'MUSCL'
            tr%data(t)%tra_adv_ver = 'QR4C'
            tr%data(t)%tra_adv_ph  = 0.0_WP
            tr%data(t)%tra_adv_pv  = 1.0_WP
            tr%data(t)%i_vert_diff = .true.
        end do
        tr%data(1)%tra_adv_lim = 'FCT'
        tr%data(2)%tra_adv_lim = 'NONE'
        allocate(tr%work%fct_LO(nl-1, nNodL), tr%work%adv_flux_hor(nl-1, nEdgeO), tr%work%adv_flux_ver(nl, nNodL))
        allocate(tr%work%fct_ttf_max(nl-1, nNodL), tr%work%fct_ttf_min(nl-1, nNodL))
        allocate(tr%work%fct_plus(nl-1, nNodL), tr%work%fct_minus(nl-1, nNodL))
        allocate(tr%work%del_ttf(nl-1, nNodL), tr%work%del_ttf_advhoriz(nl-1, nNodL), tr%work%del_ttf_advvert(nl-1, nNodL))
        tr%work%fct_LO = 0.0_WP; tr%work%adv_flux_hor = 0.0_WP; tr%work%adv_flux_ver = 0.0_WP
        tr%work%fct_ttf_max = 0.0_WP; tr%work%fct_ttf_min = 0.0_WP
        tr%work%fct_plus = 0.0_WP; tr%work%fct_minus = 0.0_WP
        tr%work%del_ttf = 0.0_WP; tr%work%del_ttf_advhoriz = 0.0_WP; tr%work%del_ttf_advvert = 0.0_WP
        call muscl_adv_init(tr%work, mesh, partit)
    end subroutine new_tracers

    subroutine set_tracer(T, base, amp, seed)
        ! base + amp*hash on the wet levels of every owned + halo node (0 below the bottom)
        real(kind=WP), intent(inout) :: T(:,:)
        real(kind=WP), intent(in)    :: base, amp
        integer,       intent(in)    :: seed
        integer :: n, nz
        T = 0.0_WP
        do n = 1, nNodL
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                T(nz, n) = base + amp*hval(partit%myList_nod2D(n), nz + seed)
            end do
        end do
    end subroutine set_tracer

    real(kind=WP) function content(T)
        ! sum over the owned nodes and wet layers of T*hnode*areasvol, all ranks
        real(kind=WP), intent(in) :: T(:,:)
        integer :: n, nz
        content = 0.0_WP
        do n = 1, nNodO
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                content = content + T(nz, n)*mesh%hnode(nz, n)*mesh%areasvol(n)
            end do
        end do
        call sum_all(content, partit)
    end function content

    real(kind=WP) function maxdev(T, T0)
        ! max over owned nodes and wet layers of |T - T0|, all ranks
        real(kind=WP), intent(in) :: T(:,:), T0
        integer :: n, nz
        maxdev = 0.0_WP
        do n = 1, nNodO
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                maxdev = max(maxdev, abs(T(nz, n) - T0))
            end do
        end do
        call max_all(maxdev, partit)
    end function maxdev

    real(kind=WP) function top_flux(d, T)
        ! the content rate of the high-order stage through the open surface: -sum area*w(top)*T(top)
        type(t_dyn),   intent(in) :: d
        real(kind=WP), intent(in) :: T(:,:)
        integer :: n, nz
        top_flux = 0.0_WP
        do n = 1, nNodO
            nz = mesh%ulevels_nod2D(n)
            top_flux = top_flux - mesh%area(n)*d%w(nz, n)*T(nz, n)
        end do
        call sum_all(top_flux, partit)
    end function top_flux

    subroutine remove_depth_mean(u)
        ! u -= sum_k helem*u / sum_k helem per element: zero depth-integrated transport
        real(kind=WP), intent(inout) :: u(:,:,:)
        integer :: e, nz
        real(kind=WP) :: H, s(2)
        do e = 1, size(u, 3)
            H = 0.0_WP; s = 0.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                H = H + mesh%helem(nz, e)
                s = s + mesh%helem(nz, e)*u(:, nz, e)
            end do
            if (H <= 0.0_WP) cycle
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                u(:, nz, e) = u(:, nz, e) - s/H
            end do
        end do
    end subroutine remove_depth_mean

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

    !===========================================================================
    subroutine r1c_vert_vel_hooks()
        type(t_dyn) :: d
        real(kind=WP), allocatable :: u1(:,:,:), u2(:,:,:), w0(:,:), we0(:,:), wi0(:,:), c0(:,:), f0(:,:)
        integer :: igm, mype
        character(len=16) :: tag
        mype = partit%mype
        allocate(u1(2, nl-1, nElemF), u2(2, nl-1, nElemF))
        call hashed_uv(u2, 91)
        do igm = 0, 1
            write(tag, '(a,i0)') 'GM=', igm
            call new_dyn(d, 2)
            d%use_wsplit = .true.; d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
            Fer_GM = (igm == 1)
            if (Fer_GM) then
                allocate(d%fer_uv(2, nl-1, nElemF), d%fer_w(nl, nNodL))
                call hashed_uv(d%fer_uv, 300)
                d%fer_uv = 0.1_WP*d%fer_uv
                d%fer_w = 0.0_WP
            end if
            call hashed_state(d, 21)
            u1 = d%uv
            call vert_vel_ale(d, mesh, dt, partit)
            w0 = d%w; we0 = d%w_e; wi0 = d%w_i; c0 = d%cfl_z
            if (Fer_GM) f0 = d%fer_w
            call vert_vel_ale(d, mesh, dt, partit, uv_in=u1)
            call check(same_nod(d%w, w0) .and. same_nod(d%w_e, we0) .and. same_nod(d%w_i, wi0) .and. &
                       same_nod(d%cfl_z, c0), 'R1c uv_in = uv is transparent, '//trim(tag), mype)
            if (Fer_GM) call check(same_nod(d%fer_w, f0), 'R1c uv_in = uv: fer_w transparent', mype)
            ! bolus off: fer_w untouched, w / w_e / w_i as the default
            if (Fer_GM) d%fer_w = -7.0_WP
            call vert_vel_ale(d, mesh, dt, partit, bolus=.false.)
            call check(same_nod(d%w, w0) .and. same_nod(d%w_e, we0) .and. same_nod(d%w_i, wi0), &
                       'R1c bolus=.false.: w, w_e, w_i as the default, '//trim(tag), mype)
            if (Fer_GM) call check(all(d%fer_w(:, 1:nNodL) == -7.0_WP), 'R1c bolus=.false. leaves fer_w', mype)
            ! completeness: uv_in = u2 with uv = u1 == the default with uv = u2
            call vert_vel_ale(d, mesh, dt, partit, uv_in=u2)
            w0 = d%w; we0 = d%w_e; wi0 = d%w_i; c0 = d%cfl_z
            d%uv = u2
            call vert_vel_ale(d, mesh, dt, partit)
            call check(same_nod(d%w, w0) .and. same_nod(d%w_e, we0) .and. same_nod(d%w_i, wi0) .and. &
                       same_nod(d%cfl_z, c0), 'R1c uv_in = u2 replaces uv, '//trim(tag), mype)
            call free_dyn(d)
        end do
        Fer_GM = .false.
    end subroutine r1c_vert_vel_hooks

    !===========================================================================
    subroutine r2_split_off()
        type(t_dyn) :: d
        real(kind=WP), allocatable :: w0(:,:)
        integer :: n, nz, nsplit, nbad, mype
        mype = partit%mype
        call new_dyn(d, 2)
        d%use_wsplit = .true.; d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
        call hashed_state(d, 31)
        call vert_vel_ale(d, mesh, dt, partit)
        w0 = d%w
        nsplit = 0; nbad = 0
        do n = 1, nNodO
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n)
                if (d%w_i(nz, n) /= 0.0_WP) nsplit = nsplit + 1
                if (abs(d%w_e(nz, n) + d%w_i(nz, n) - d%w(nz, n)) > spacing(abs(d%w(nz, n)))) nbad = nbad + 1
            end do
        end do
        call sum_all(nsplit, partit); call sum_all(nbad, partit)
        call check(nsplit > 100, 'R2 precondition: split faces in the default call', mype)
        call check(nbad == 0, 'R2 default call: w_e + w_i == w to one spacing', mype)
        d%w_i = 3.0_WP; d%w_e = 3.0_WP
        call vert_vel_ale(d, mesh, dt, partit, split=.false.)
        call check(same_nod(d%w, w0), 'R2 split=.false.: w bitwise the default', mype)
        nbad = 0
        do n = 1, nNodL
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n)
                if (d%w_e(nz, n) /= d%w(nz, n) .or. d%w_i(nz, n) /= 0.0_WP) nbad = nbad + 1
            end do
        end do
        call sum_all(nbad, partit)
        call check(nbad == 0, 'R2 split=.false.: w_e == w, w_i == 0 on every owned + halo face', mype)
        if (mype == 0) write(*,'(a,i0)') 'test_rk3: R2 split faces in the default call: ', nsplit
    end subroutine r2_split_off

    !===========================================================================
    subroutine p_pressure_only()
        real(kind=WP), allocatable :: T(:,:), S(:,:), rref(:,:), rho1(:,:), hp1(:,:), bv(:,:), bvr(:,:), db(:,:)
        real(kind=WP), allocatable :: rho2(:,:), hp2(:,:)
        integer :: n, nz, mype
        logical :: smth_save
        mype = partit%mype
        smth_save = N2smth_h
        N2smth_h = .true.
        allocate(T(nl-1, nNodL), S(nl-1, nNodL), rref(nl-1, nNodL))
        allocate(rho1(nl-1, nNodL), hp1(nl, nNodL), bv(nl, nNodL), bvr(nl, nNodL), db(nl, nNodL))
        T = 0.0_WP; S = 0.0_WP; rref = density_0
        do n = 1, nNodL
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                T(nz, n) = 12.0_WP - 0.002_WP*real(nz, WP)*10.0_WP + 2.0_WP*hval(partit%myList_nod2D(n), nz)
                S(nz, n) = 35.0_WP + 0.5_WP*hval(partit%myList_nod2D(n), nz + 40)
            end do
        end do
        rho1 = 0.0_WP; hp1 = 0.0_WP; bv = 0.0_WP; bvr = 0.0_WP; db = 0.0_WP
        call pressure_bv(T, S, rref, mesh, rho1, hp1, bv, partit, dbsfc=db, bvfreq_raw=bvr)
        call check(any(bv(:, 1:nNodO) /= 0.0_WP), 'P the full call computes N^2 (non-vacuous)', mype)
        allocate(rho2(nl-1, nNodL), hp2(nl, nNodL))
        rho2 = 0.0_WP; hp2 = 0.0_WP; bv = -777.0_WP; bvr = -777.0_WP; db = -777.0_WP
        call pressure_bv(T, S, rref, mesh, rho2, hp2, bv, partit, dbsfc=db, bvfreq_raw=bvr, pressure_only=.true.)
        call check(same_nod(rho2, rho1), 'P pressure_only: density_m_rho0 bitwise the full call', mype)
        call check(same_nod(hp2, hp1), 'P pressure_only: hpressure bitwise the full call', mype)
        call check(all(bv(:, 1:nNodL) == -777.0_WP) .and. all(bvr(:, 1:nNodL) == -777.0_WP) .and. &
                   all(db(:, 1:nNodL) == -777.0_WP), 'P pressure_only: bvfreq, bvfreq_raw, dbsfc untouched', mype)
        N2smth_h = smth_save
    end subroutine p_pressure_only

    !===========================================================================
    subroutine new_rk3(d)
        ! the RK3 stage arrays rk3_init allocates (without its env and guards)
        type(t_dyn), intent(inout) :: d
        allocate(d%rk3_us(2, nl-1, nElemF), d%rk3_visc(2, nl-1, nElemF), d%rk3_H(nElemL))
        d%rk3_us = 0.0_WP; d%rk3_visc = 0.0_WP; d%rk3_H = 0.0_WP
        d%use_rk3 = .true.; d%rk3_ready = .true.
    end subroutine new_rk3

    subroutine r3_momentum_stage()
        type(t_dyn) :: d
        real(kind=WP), allocatable :: s0(:,:), s1(:,:), tgt(:,:), u0(:,:,:), r(:,:,:), visc(:,:,:)
        real(kind=WP) :: etr, edu, umax, smax, spread, c(2), ev
        integer :: stage, e, nz, mype
        mype = partit%mype
        call new_dyn(d, 1)
        call new_rk3(d)
        d%use_wsplit = .true.; d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
        call hashed_state(d, 101)
        call rk3_depth(d, mesh, partit)
        allocate(s0(2, nElemL), s1(2, nElemL), tgt(2, nElemL))
        call rk3_transport(d%uv, mesh, nElemO, s0)
        u0 = d%uv
        do stage = 1, 2
            if (stage == 1) then
                call vert_vel_ale(d, mesh, dt, partit, split=.false., bolus=.false.)
            else
                call vert_vel_ale(d, mesh, dt, partit, uv_in=d%rk3_us, split=.false., bolus=.false.)
            end if
            call rk3_momentum_stage(stage, dt/real(4-stage, WP), d, mesh, partit)
            r = d%uv_rhs
            call rk3_transport(d%rk3_us, mesh, nElemO, s1)
            etr = 0.0_WP; edu = 0.0_WP; umax = 0.0_WP; smax = 0.0_WP
            do e = 1, nElemO
                c = 0.0_WP
                do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                    c = c + mesh%helem(nz, e)*abs(d%uv(:, nz, e))
                    umax = max(umax, maxval(abs(d%uv(:, nz, e))))
                end do
                smax = max(smax, maxval(c))
                etr  = max(etr, maxval(abs(s1(:, e) - s0(:, e))))
                nz = mesh%ulevels(e)
                c = d%rk3_us(:, nz, e) - (d%uv(:, nz, e) + r(:, nz, e))
                do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                    edu = max(edu, maxval(abs(d%rk3_us(:, nz, e) - (d%uv(:, nz, e) + r(:, nz, e)) - c)))
                end do
            end do
            call max_all(etr, partit); call max_all(edu, partit); call max_all(umax, partit); call max_all(smax, partit)
            call check(etr <= 1.0e-14_WP*smax, 'R3 the stage transport equals sum_k helem*u^n', mype)
            call check(edu <= 1.0e-15_WP*umax*10.0_WP, 'R3 u_s - (u^n + R_s) is depth-uniform', mype)
            spread = shared_copy_max_elem(d%rk3_us, partit)
            call check(spread == 0.0_WP, 'R3 shared copies of u_s agree', mype)
            if (mype == 0) write(*,'(a,i0,a,es9.2,a,es9.2,a,es9.2)') 'test_rk3: R3 stage ', stage, &
                ' |transport - U^n|/max = ', etr/smax, ', depth spread/umax = ', edu/umax, ', copy spread ', spread
        end do
        call check(all(d%uv == u0), 'R3 dyn%uv keeps u^n', mype)
        ! rk3_trim to a hashed target
        do e = 1, nElemL
            tgt(1, e) = 50.0_WP*hval(partit%myList_elem2D(e), 3)
            tgt(2, e) = 50.0_WP*hval(partit%myList_elem2D(e), 4)
        end do
        r = d%rk3_us
        call rk3_trim(d%rk3_us, tgt, d, mesh, partit)
        call rk3_transport(d%rk3_us, mesh, nElemO, s1)
        etr = 0.0_WP; edu = 0.0_WP
        do e = 1, nElemO
            etr = max(etr, maxval(abs(s1(:, e) - tgt(:, e))))
            nz = mesh%ulevels(e)
            c = d%rk3_us(:, nz, e) - r(:, nz, e)
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                edu = max(edu, maxval(abs(d%rk3_us(:, nz, e) - r(:, nz, e) - c)))
            end do
        end do
        call max_all(etr, partit); call max_all(edu, partit)
        call check(etr <= 1.0e-14_WP*max(smax, 50.0_WP), 'R3 rk3_trim reaches the target transport', mype)
        call check(edu <= 1.0e-14_WP, 'R3 rk3_trim changes u depth-uniformly', mype)
        ! rk3_viscosity: the viscosity_filter increment of u^n from zero
        call rk3_viscosity(d, mesh, dt, partit)
        visc = d%rk3_visc
        d%uv_rhs = 0.0_WP
        call viscosity_filter(d%opt_visc, d, mesh, dt, partit)
        call check(same_elem(visc, d%uv_rhs), 'R3 rk3_visc == the viscosity increment of u^n', mype)
        ev = maxval(abs(visc(:, :, 1:nElemO)))
        call max_all(ev, partit)
        call check(ev > 0.0_WP, 'R3 the viscosity increment is not zero', mype)
    end subroutine r3_momentum_stage

    !===========================================================================
    subroutine r4_inertial()
        type(t_dyn) :: d
        real(kind=WP), allocatable :: u0(:,:,:)
        complex(kind=WP) :: z, A, c0, c1, ce
        real(kind=WP) :: err, erre, zmax
        integer :: e, nz, mype
        mype = partit%mype
        call new_dyn(d, 0)
        call new_rk3(d)
        call hashed_uv(d%uv, 111)
        call remove_depth_mean(d%uv)
        u0 = d%uv
        call rk3_depth(d, mesh, partit)
        call rk3_momentum_stage(1, dt/3.0_WP, d, mesh, partit)
        call rk3_momentum_stage(2, 0.5_WP*dt, d, mesh, partit)
        call compute_vel_rhs(d, mesh, dt, .true., partit, uv_in=d%rk3_us, no_history=.true.)
        err = 0.0_WP; erre = 0.0_WP; zmax = 0.0_WP
        do e = 1, nElemO
            z = cmplx(0.0_WP, -mesh%coriolis(e)*dt, kind=WP)
            zmax = max(zmax, abs(z))
            A = 1.0_WP + z + z*z/2.0_WP + z*z*z/6.0_WP
            do nz = mesh%ulevels(e), mesh%nlevels(e) - 1
                c0 = cmplx(u0(1, nz, e), u0(2, nz, e), kind=WP)
                c1 = cmplx(u0(1, nz, e) + d%uv_rhs(1, nz, e), u0(2, nz, e) + d%uv_rhs(2, nz, e), kind=WP)
                if (abs(c0) < 1.0e-6_WP) cycle
                err = max(err, abs(c1 - A*c0)/abs(c0))
                ce = c0*(1.0_WP + z)
                erre = max(erre, abs(ce - A*c0)/abs(c0))
            end do
        end do
        call max_all(err, partit); call max_all(erre, partit); call max_all(zmax, partit)
        call check(zmax > 0.1_WP, 'R4 precondition: max |f tau| > 0.1', mype)
        call check(err <= 1.0e-12_WP, 'R4 three stages = the RK3 amplification', mype)
        call check(erre > 1.0e-3_WP, 'R4 positive control: Euler misses it', mype)
        if (mype == 0) write(*,'(a,f6.3,a,es9.2,a,es9.2)') 'test_rk3: R4 max|f tau| = ', zmax, &
            ', max|c - A c^n|/|c^n| = ', err, ', Euler ', erre
    end subroutine r4_inertial

    !===========================================================================
    subroutine r5_ho_stage()
        type(t_dyn)    :: d
        type(t_tracer) :: tr
        real(kind=WP), allocatable :: us(:,:,:)
        real(kind=WP) :: dev, c0, c1, ex, wmax, rel
        integer :: n, npos, nneg, t, mype
        mype = partit%mype
        call new_dyn(d, 2)
        d%use_wsplit = .true.; d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
        call new_tracers(tr)
        call hashed_state(d, 41)
        allocate(us(2, nl-1, nElemF))
        call hashed_uv(us, 42)

        ! (a) constancy through stages 1 and 2, open surface of both signs
        call set_tracer(tr%data(1)%values, 7.3_WP, 0.0_WP, 0)
        call set_tracer(tr%data(2)%values, 34.7_WP, 0.0_WP, 0)
        call vert_vel_ale(d, mesh, dt, partit, split=.false.)
        wmax = 0.0_WP; npos = 0; nneg = 0
        do n = 1, nNodO
            wmax = max(wmax, abs(d%w(1, n)))
        end do
        call max_all(wmax, partit)
        do n = 1, nNodO
            if (d%w(1, n) >  1.0e-6_WP*wmax) npos = npos + 1
            if (d%w(1, n) < -1.0e-6_WP*wmax) nneg = nneg + 1
        end do
        call sum_all(npos, partit); call sum_all(nneg, partit)
        call check(npos > 10 .and. nneg > 10, 'R5 precondition: w(top) of both signs', mype)
        call rk3_tracer_stage(1, dt/3.0_WP, d%uv, d%w, d, tr, mesh, partit)
        do t = 1, 2
            dev = maxdev(tr%data(t)%valuesAB, tr%data(t)%values(1, 1))
            call check(dev <= 1.0e-13_WP*tr%data(t)%values(1, 1), 'R5 stage 1 keeps a uniform tracer '// &
                       trim(tr%data(t)%tra_adv_lim), mype)
            if (mype == 0) write(*,'(a,i0,a,es9.2)') 'test_rk3: R5 stage 1 tracer ', t, ' max|T* - T0| = ', dev
        end do
        call vert_vel_ale(d, mesh, dt, partit, uv_in=us, split=.false.)
        call rk3_tracer_stage(2, 0.5_WP*dt, us, d%w, d, tr, mesh, partit)
        do t = 1, 2
            dev = maxdev(tr%data(t)%valuesAB, tr%data(t)%values(1, 1))
            call check(dev <= 1.0e-13_WP*tr%data(t)%values(1, 1), 'R5 stage 2 keeps a uniform tracer '// &
                       trim(tr%data(t)%tra_adv_lim), mype)
            if (mype == 0) write(*,'(a,i0,a,es9.2)') 'test_rk3: R5 stage 2 tracer ', t, ' max|T** - T0| = ', dev
        end do

        ! (b) open surface: the content change is tau_s times the top flux
        call set_tracer(tr%data(1)%values, 10.0_WP, 4.0_WP, 5)
        call set_tracer(tr%data(2)%values, 35.0_WP, 1.0_WP, 9)
        call vert_vel_ale(d, mesh, dt, partit, split=.false.)
        do t = 1, 2
            tr%data(t)%valuesAB = tr%data(t)%values
        end do
        call rk3_tracer_stage(1, dt/3.0_WP, d%uv, d%w, d, tr, mesh, partit)
        do t = 1, 2
            c0 = content(tr%data(t)%values)
            c1 = content(tr%data(t)%valuesAB)
            ex = dt/3.0_WP*top_flux(d, tr%data(t)%values)
            rel = abs(c1 - c0 - ex)/abs(c0)
            call check(rel <= 1.0e-13_WP, 'R5 open surface: content change = tau_s*top flux', mype)
            call check(abs(ex)/abs(c0) > 1.0e-11_WP, 'R5 open surface: the top flux is not negligible', mype)
            if (mype == 0) write(*,'(a,i0,a,es9.2,a,es9.2)') 'test_rk3: R5 open tracer ', t, &
                ' |dC - tau*F|/C = ', rel, ', tau*F/C = ', ex/c0
        end do

        ! (c) closed surface (depth mean removed): the content is conserved
        call remove_depth_mean(d%uv)
        call vert_vel_ale(d, mesh, dt, partit, split=.false.)
        call rk3_tracer_stage(1, dt/3.0_WP, d%uv, d%w, d, tr, mesh, partit)
        do t = 1, 2
            c0 = content(tr%data(t)%values)
            c1 = content(tr%data(t)%valuesAB)
            rel = abs(c1 - c0)/abs(c0)
            call check(rel <= 1.0e-13_WP, 'R5 closed surface: content conserved', mype)
            if (mype == 0) write(*,'(a,i0,a,es9.2)') 'test_rk3: R5 closed tracer ', t, ' |dC|/C = ', rel
        end do
    end subroutine r5_ho_stage

    !===========================================================================
    subroutine r6_substep_n1()
        type(t_dyn)    :: d
        type(t_tracer) :: tr
        real(kind=WP), allocatable :: lo0(:,:), dh0(:,:), dv0(:,:)
        integer :: t, nsplit, mype
        mype = partit%mype
        call new_dyn(d, 2)
        d%use_wsplit = .true.; d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
        call new_tracers(tr)
        call hashed_state(d, 51)
        call vert_vel_ale(d, mesh, dt, partit)
        nsplit = count(d%w_i(:, 1:nNodO) /= 0.0_WP)
        call sum_all(nsplit, partit)
        call check(nsplit > 100, 'R6 precondition: split faces', mype)
        do t = 1, 2
            call set_tracer(tr%data(t)%values,   10.0_WP*t, 3.0_WP, 11*t)
            call set_tracer(tr%data(t)%valuesAB, 10.0_WP*t, 3.0_WP, 13*t)
            call rk3_tracer_prep(t, tr, mesh, partit)
            call do_oce_adv_tra(dt, d%uv, d%w, d%w_i, d%w_e, t, d, tr, mesh, partit)
            lo0 = tr%work%fct_LO; dh0 = tr%work%del_ttf_advhoriz; dv0 = tr%work%del_ttf_advvert
            tr%work%fct_LO = -1.0_WP
            call rk3_tracer_prep(t, tr, mesh, partit)
            call do_oce_adv_tra_rk3(dt, d%uv, d%w, d%w_i, d%w_e, t, 3, 1, d, tr, mesh, partit)
            call check(same_nod(tr%work%del_ttf_advhoriz, dh0) .and. same_nod(tr%work%del_ttf_advvert, dv0), &
                       'R6 N = 1: del_ttf_adv* bitwise do_oce_adv_tra, '//trim(tr%data(t)%tra_adv_lim), mype)
            if (t == 1) call check(same_nod(tr%work%fct_LO, lo0), 'R6 N = 1: fct_LO bitwise do_oce_adv_tra', mype)
            call check(any(dv0(:, 1:nNodO) /= 0.0_WP), 'R6 non-vacuous tendency', mype)
        end do
    end subroutine r6_substep_n1

    !===========================================================================
    subroutine stage3_update(tr, t, Tnew)
        ! T^{n+1} after the stage-3 advection alone: the identity form of diff_tracers_ale's
        ! reconstruct on the owned nodes (no diffusion)
        type(t_tracer), intent(in)    :: tr
        integer,        intent(in)    :: t
        real(kind=WP),  intent(inout) :: Tnew(:,:)
        integer :: n, nz
        real(kind=WP) :: dl
        Tnew = tr%data(t)%values
        do n = 1, nNodO
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                dl = tr%work%del_ttf_advhoriz(nz, n) + tr%work%del_ttf_advvert(nz, n) &
                   + tr%data(t)%values(nz, n)*(mesh%hnode(nz, n) - mesh%hnode_new(nz, n))
                Tnew(nz, n) = tr%data(t)%values(nz, n) + dl/mesh%hnode_new(nz, n)
            end do
        end do
    end subroutine stage3_update

    real(kind=WP) function lo_top_sum(tr)
        integer :: n
        type(t_tracer), intent(in) :: tr
        lo_top_sum = 0.0_WP
        do n = 1, nNodO
            lo_top_sum = lo_top_sum + tr%work%rk3_lo_top(n) - tr%work%adv_flux_ver(mesh%ulevels_nod2D(n), n)
        end do
        call sum_all(lo_top_sum, partit)
    end function lo_top_sum

    subroutine r7_substeps()
        type(t_dyn)    :: d
        type(t_tracer) :: tr
        real(kind=WP), allocatable :: Tnew(:,:)
        real(kind=WP) :: dev, devlo, c0, c1, ex, rel
        integer :: nsub, nsplit, mype
        character(len=8) :: tag
        mype = partit%mype
        call new_dyn(d, 2)
        d%use_wsplit = .true.; d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
        call new_tracers(tr)
        call hashed_state(d, 61)
        call vert_vel_ale(d, mesh, dt, partit)
        nsplit = count(d%w_i(:, 1:nNodO) /= 0.0_WP)
        call sum_all(nsplit, partit)
        call check(nsplit > 100, 'R7 precondition: split faces', mype)
        allocate(Tnew(nl-1, nNodL))
        do nsub = 1, 3
            write(tag, '(a,i0)') 'N=', nsub
            ! constancy
            call set_tracer(tr%data(1)%values, 7.3_WP, 0.0_WP, 0)
            tr%data(1)%valuesAB = tr%data(1)%values
            call rk3_tracer_prep(1, tr, mesh, partit)
            call do_oce_adv_tra_rk3(dt, d%uv, d%w, d%w_i, d%w_e, 1, 3, nsub, d, tr, mesh, partit)
            call stage3_update(tr, 1, Tnew)
            devlo = maxdev(tr%work%fct_LO, 7.3_WP)
            dev   = maxdev(Tnew, 7.3_WP)
            call check(devlo <= 1.0e-13_WP*7.3_WP .and. dev <= 1.0e-13_WP*7.3_WP, &
                       'R7 constancy of fct_LO and T^{n+1}, '//trim(tag), mype)
            if (mype == 0) write(*,'(a,a,a,es9.2,a,es9.2)') 'test_rk3: R7 ', trim(tag), &
                ' max|LO - T0| = ', devlo, ', max|T^{n+1} - T0| = ', dev
            ! open surface: content change = tau*(adf(top) - mean LO surface transport)
            call set_tracer(tr%data(1)%values,   10.0_WP, 4.0_WP, 17)
            call set_tracer(tr%data(1)%valuesAB, 10.0_WP, 4.0_WP, 19)
            call rk3_tracer_prep(1, tr, mesh, partit)
            call do_oce_adv_tra_rk3(dt, d%uv, d%w, d%w_i, d%w_e, 1, 3, nsub, d, tr, mesh, partit)
            call stage3_update(tr, 1, Tnew)
            c0 = content(tr%data(1)%values); c1 = content(Tnew)
            ex = -dt*lo_top_sum(tr)
            rel = abs(c1 - c0 - ex)/abs(c0)
            call check(rel <= 1.0e-13_WP, 'R7 open surface: content change = tau*substep-mean surface flux, '// &
                       trim(tag), mype)
            call check(abs(ex)/abs(c0) > 1.0e-11_WP, 'R7 open surface: the surface flux is not negligible', mype)
            if (mype == 0) write(*,'(a,a,a,es9.2,a,es9.2)') 'test_rk3: R7 ', trim(tag), &
                ' open |dC - tau*F|/C = ', rel, ', tau*F/C = ', ex/c0
        end do
        ! closed surface: conservation
        call remove_depth_mean(d%uv)
        call vert_vel_ale(d, mesh, dt, partit)
        do nsub = 1, 3
            write(tag, '(a,i0)') 'N=', nsub
            call set_tracer(tr%data(1)%values,   10.0_WP, 4.0_WP, 17)
            call set_tracer(tr%data(1)%valuesAB, 10.0_WP, 4.0_WP, 19)
            call rk3_tracer_prep(1, tr, mesh, partit)
            call do_oce_adv_tra_rk3(dt, d%uv, d%w, d%w_i, d%w_e, 1, 3, nsub, d, tr, mesh, partit)
            call stage3_update(tr, 1, Tnew)
            c0 = content(tr%data(1)%values); c1 = content(Tnew)
            rel = abs(c1 - c0)/abs(c0)
            call check(rel <= 1.0e-13_WP, 'R7 closed surface: content conserved, '//trim(tag), mype)
            if (mype == 0) write(*,'(a,a,a,es9.2)') 'test_rk3: R7 ', trim(tag), ' closed |dC|/C = ', rel
        end do
    end subroutine r7_substeps

    !===========================================================================
    real(kind=WP) function hor_courant(d)
        ! max over owned nodes and wet layers of tau*(horizontal outflow)/(areasvol*hnode): the
        ! upwind flux of T = 1 is the edge transport into the edge's first node
        type(t_dyn), intent(in) :: d
        real(kind=WP), allocatable :: one(:,:), fl(:,:), out(:,:)
        integer :: e, n, nz, en(2)
        allocate(one(nl-1, nNodL), fl(nl-1, nEdgeO), out(nl-1, nNodL))
        call set_tracer(one, 1.0_WP, 0.0_WP, 0)
        call adv_tra_hor_upw1(d%uv, one, mesh, fl, o_init_zero=.true., partit=partit)
        out = 0.0_WP
        do e = 1, nEdgeO
            en = mesh%edges(:, e)
            out(:, en(1)) = out(:, en(1)) + max(0.0_WP, -fl(:, e))
            out(:, en(2)) = out(:, en(2)) + max(0.0_WP,  fl(:, e))
        end do
        hor_courant = 0.0_WP
        do n = 1, nNodO
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                hor_courant = max(hor_courant, dt*out(nz, n)/(mesh%areasvol(n)*mesh%hnode(nz, n)))
            end do
        end do
        call max_all(hor_courant, partit)
    end function hor_courant

    subroutine minmax(T, tmin, tmax)
        real(kind=WP), intent(in)  :: T(:,:)
        real(kind=WP), intent(out) :: tmin, tmax
        integer :: n, nz
        tmin = huge(1.0_WP); tmax = -huge(1.0_WP)
        do n = 1, nNodO
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                tmin = min(tmin, T(nz, n)); tmax = max(tmax, T(nz, n))
            end do
        end do
        tmin = -tmin
        call max_all(tmin, partit); call max_all(tmax, partit)
        tmin = -tmin
    end subroutine minmax

    subroutine r8_lo_bounded()
        type(t_dyn)    :: d
        type(t_tracer) :: tr
        real(kind=WP) :: c, tmin, tmax, lmin, lmax, over(2)
        integer :: nsub, mype
        mype = partit%mype
        call new_dyn(d, 2)
        d%use_wsplit = .true.; d%wsplit_maxcfl = 0.05_WP; d%wsplit_mincfl = 0.025_WP
        call new_tracers(tr)
        call hashed_state(d, 71)
        c = hor_courant(d)
        d%uv = d%uv*(1.5_WP/c)
        c = hor_courant(d)
        call vert_vel_ale(d, mesh, dt, partit)
        call set_tracer(tr%data(1)%values, 10.0_WP, 4.0_WP, 23)
        tr%data(1)%valuesAB = tr%data(1)%values
        call minmax(tr%data(1)%values, tmin, tmax)
        do nsub = 1, 2
            call rk3_tracer_prep(1, tr, mesh, partit)
            call do_oce_adv_tra_rk3(dt, d%uv, d%w, d%w_i, d%w_e, 1, 3, nsub, d, tr, mesh, partit)
            call minmax(tr%work%fct_LO, lmin, lmax)
            over(nsub) = max(tmin - lmin, lmax - tmax, 0.0_WP)/(tmax - tmin)
        end do
        call check(abs(c - 1.5_WP) < 1.0e-9_WP, 'R8 precondition: horizontal Courant number 1.5', mype)
        call check(over(2) <= 1.0e-12_WP, 'R8 N = 2: the low order stays within [min, max] of T^n', mype)
        call check(over(1) > 1.0e-3_WP, 'R8 positive control: N = 1 leaves [min, max]', mype)
        if (mype == 0) write(*,'(a,f6.3,a,es9.2,a,es9.2)') 'test_rk3: R8 Courant ', c, &
            ', overshoot/range N=1 ', over(1), ', N=2 ', over(2)
    end subroutine r8_lo_bounded

    !===========================================================================
    subroutine r9_fct_stage3()
        type(t_dyn)    :: d
        type(t_tracer) :: tr
        real(kind=WP), allocatable :: Tnew(:,:), fh(:,:), fv(:,:), Tfull(:,:), Twrong(:,:)
        logical, allocatable :: dirty(:,:), clean(:,:)
        real(kind=WP) :: tmin, tmax, viol, err, errw, fmin, c, lon, lat, xn
        real(kind=WP), parameter :: lon0 = 1.5707963267948966_WP, lat0 = 0.7853981633974483_WP
        integer :: nsub, n, nz, e, en(2), nlim, ncl, nwet, mype
        character(len=8) :: tag
        mype = partit%mype
        call new_dyn(d, 2)
        d%use_wsplit = .true.; d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
        call new_tracers(tr)
        allocate(Tnew(nl-1, nNodL), Tfull(nl-1, nNodL), Twrong(nl-1, nNodL))
        allocate(dirty(nl-1, nNodL), clean(nl-1, nNodL))
        ! (a) a hashed field: the Zalesak bounds hold, the limiter acts
        call hashed_state(d, 81)
        call vert_vel_ale(d, mesh, dt, partit)
        do nsub = 2, 3
            write(tag, '(a,i0)') 'N=', nsub
            call set_tracer(tr%data(1)%values,   10.0_WP, 4.0_WP, 27)
            call set_tracer(tr%data(1)%valuesAB, 10.0_WP, 4.0_WP, 29)
            call minmax(tr%data(1)%values, tmin, tmax)
            call rk3_tracer_prep(1, tr, mesh, partit)
            call do_oce_adv_tra_rk3(dt, d%uv, d%w, d%w_i, d%w_e, 1, 3, nsub, d, tr, mesh, partit)
            call stage3_update(tr, 1, Tnew)
            viol = 0.0_WP; nlim = 0
            do n = 1, nNodO
                do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                    viol = max(viol, Tnew(nz,n) - tr%work%fct_LO(nz,n) - tr%work%fct_ttf_max(nz,n), &
                                     tr%work%fct_ttf_min(nz,n) - (Tnew(nz,n) - tr%work%fct_LO(nz,n)))
                    if (tr%work%fct_plus(nz,n) < 1.0_WP .or. tr%work%fct_minus(nz,n) < 1.0_WP) nlim = nlim + 1
                end do
            end do
            call max_all(viol, partit); call sum_all(nlim, partit)
            call check(viol <= 1.0e-12_WP*(tmax - tmin), 'R9 Zalesak bounds hold, '//trim(tag), mype)
            call check(nlim > 0, 'R9 the limiter acts on the hashed field, '//trim(tag), mype)
            if (mype == 0) write(*,'(a,a,a,es9.2,a,i0)') 'test_rk3: R9 ', trim(tag), ' bound violation/range = ', &
                viol/(tmax - tmin), ', limited cells ', nlim
        end do
        ! (b) a smooth field at a small Courant number: no limiting, LO + the full antidiffusion
        call hashed_state(d, 83)
        c = hor_courant(d)
        d%uv = d%uv*(0.01_WP/c)
        call vert_vel_ale(d, mesh, dt, partit)
        allocate(fh(nl-1, nEdgeO), fv(nl, nNodL))
        do nsub = 2, 3
            write(tag, '(a,i0)') 'N=', nsub
            ! depth-uniform (no vertical antidiffusion) and linear in the 3-D Cartesian position,
            ! T = 10 + 2*(x.n) with n towards (90E, 45N): its extrema on the sphere are over land
            do n = 1, nNodL
                lon = mesh%geo_coord_nod2D(1, n); lat = mesh%geo_coord_nod2D(2, n)
                xn  = cos(lat)*cos(lon)*cos(lat0)*cos(lon0) + cos(lat)*sin(lon)*cos(lat0)*sin(lon0) &
                    + sin(lat)*sin(lat0)
                do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                    tr%data(1)%values(nz, n) = 10.0_WP + 2.0_WP*xn
                end do
            end do
            tr%data(1)%valuesAB = tr%data(1)%values
            call rk3_tracer_prep(1, tr, mesh, partit)
            call do_oce_adv_tra_rk3(dt, d%uv, d%w, d%w_i, d%w_e, 1, 3, nsub, d, tr, mesh, partit)
            call stage3_update(tr, 1, Tnew)
            fmin = 1.0_WP
            do n = 1, nNodO
                do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                    fmin = min(fmin, tr%work%fct_plus(nz,n), tr%work%fct_minus(nz,n))
                end do
            end do
            ! the clean cells: every flux into the cell unlimited
            dirty = .false.
            do n = 1, nNodL
                do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                    if (tr%work%fct_plus(nz,n) < 1.0_WP .or. tr%work%fct_minus(nz,n) < 1.0_WP) dirty(nz, n) = .true.
                end do
            end do
            clean = .not. dirty
            do e = 1, nEdgeO
                en = mesh%edges(:, e)
                do nz = 1, nl-1
                    if (dirty(nz, en(1)) .or. dirty(nz, en(2))) then
                        clean(nz, en(1)) = .false.; clean(nz, en(2)) = .false.
                    end if
                end do
            end do
            do n = 1, nNodO
                do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                    if (nz > mesh%ulevels_nod2D(n)) then
                        if (dirty(nz-1, n)) clean(nz, n) = .false.
                    end if
                    if (nz < mesh%nlevels_nod2D(n) - 1) then
                        if (dirty(nz+1, n)) clean(nz, n) = .false.
                    end if
                end do
            end do
            ! LO + the scatter of HO(T**) - the substep-mean LO flux; the control with the LO flux
            ! of one step from T^n
            call full_adf(d, tr, tr%work%rk3_lo_fh, tr%work%rk3_lo_fv, Tfull)
            call adv_tra_hor_upw1(d%uv, tr%data(1)%values, mesh, fh, o_init_zero=.true., partit=partit)
            call adv_tra_ver_upw1(d%w, tr%data(1)%values, mesh, fv, o_init_zero=.true., partit=partit)
            call full_adf(d, tr, fh, fv, Twrong)
            err = 0.0_WP; errw = 0.0_WP; ncl = 0; nwet = 0
            do n = 1, nNodO
                do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                    nwet = nwet + 1
                    if (.not. clean(nz, n)) cycle
                    ncl  = ncl + 1
                    err  = max(err,  abs(Tnew(nz, n) - Tfull(nz, n)))
                    errw = max(errw, abs(Tnew(nz, n) - Twrong(nz, n)))
                end do
            end do
            call max_all(err, partit); call max_all(errw, partit)
            call sum_all(ncl, partit); call sum_all(nwet, partit)
            call check(real(ncl, WP) >= 0.9_WP*real(nwet, WP), 'R9 smooth field: >= 90 % clean cells, '//trim(tag), mype)
            call check(err <= 1.0e-13_WP*10.0_WP, 'R9 smooth field: T^{n+1} = LO + full antidiffusion, '//trim(tag), mype)
            call check(errw > 1.0e-9_WP, 'R9 positive control: the one-step LO flux differs, '//trim(tag), mype)
            if (mype == 0) write(*,'(a,a,a,f8.5,a,i0,a,i0,a,es9.2,a,es9.2)') 'test_rk3: R9 ', trim(tag), &
                ' smooth: min factor ', fmin, ', clean ', ncl, ' of ', nwet, ', max|T - (LO + adf)| = ', err, &
                ', control ', errw
        end do

    end subroutine r9_fct_stage3

    subroutine full_adf(d, tr, lofh, lofv, Tout)
        ! R9: fct_LO + the scatter of HO(T**) - (lofh, lofv) on the owned nodes
        type(t_dyn),    intent(in)    :: d
        type(t_tracer), intent(in)    :: tr
        real(kind=WP),  intent(in)    :: lofh(:,:), lofv(:,:)
        real(kind=WP),  intent(inout) :: Tout(:,:)
        real(kind=WP), allocatable :: fh(:,:), fv(:,:)
        real(kind=WP_full), allocatable :: dh(:,:), dv(:,:)
        integer :: n, nz
        fh = lofh; fv = lofv
        allocate(dh(nl-1, nNodL), dv(nl-1, nNodL))
        call adv_tra_hor_muscl(d%uv, tr%data(1)%valuesAB, mesh, tr%data(1)%tra_adv_ph, fh, tr%work%tr_xy, &
                               tr%work%edge_up_dn_tri, o_init_zero=.false., partit=partit)
        call adv_tra_ver_qr4c(d%w, tr%data(1)%valuesAB, mesh, tr%data(1)%tra_adv_pv, fv, &
                              o_init_zero=.false., partit=partit)
        dh = 0.0_WP; dv = 0.0_WP
        call oce_tra_adv_flux2dtracer(dt, dh, dv, fh, fv, mesh, partit=partit)
        Tout = tr%work%fct_LO
        do n = 1, nNodO
            do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
                Tout(nz, n) = tr%work%fct_LO(nz, n) + (dh(nz, n) + dv(nz, n))/mesh%hnode_new(nz, n)
            end do
        end do
    end subroutine full_adf

    !===========================================================================
    subroutine r10t_rest()
        type(t_dyn)    :: d
        type(t_tracer) :: ta, tb
        real(kind=WP), allocatable :: Ki(:,:), hf(:), wf(:), vs(:), rs(:), sf(:)
        integer :: t, mype
        logical :: same
        mype = partit%mype
        call new_dyn(d, 2)
        call new_rk3(d)
        d%use_wsplit = .true.; d%wsplit_maxcfl = 1.0e-3_WP; d%wsplit_mincfl = 0.5e-3_WP
        call rest_inputs(d, Ki, hf, wf, vs, rs, sf)
        call new_tracers(ta)
        do t = 1, 2
            call set_tracer(ta%data(t)%values, 10.0_WP + 20.0_WP*(t-1), 3.0_WP, 31*t)
            call set_tracer(ta%data(t)%valuesold(1,:,:), 10.0_WP + 20.0_WP*(t-1), 3.0_WP, 37*t)
        end do
        tb = ta
        call vert_vel_ale(d, mesh, dt, partit)                    ! u = 0: w = w_e = w_i = 0
        call check(all(d%w == 0.0_WP), 'R10t precondition: w = 0', mype)
        call solve_tracers_ale(dt, d, ta, mesh, Ki, hf, wf, vs, rs, sf, 0.0_WP, partit)
        call rk3_tracer_stage(1, dt/3.0_WP, d%rk3_us, d%w, d, tb, mesh, partit)
        call rk3_tracer_stage(2, 0.5_WP*dt, d%rk3_us, d%w, d, tb, mesh, partit)
        same = .true.
        do t = 1, 2
            same = same .and. same_nod(tb%data(t)%valuesAB, tb%data(t)%values)
        end do
        call check(same, 'R10t at rest T** == T^n bitwise', mype)
        call solve_tracers_rk3(dt, 1, d, tb, mesh, Ki, hf, wf, vs, rs, sf, 0.0_WP, partit)
        do t = 1, 2
            call check(same_nod(tb%data(t)%values, ta%data(t)%values), &
                       'R10t at rest solve_tracers_rk3 == solve_tracers_ale bitwise', mype)
            call check(any(ta%data(t)%values(:, 1:nNodO) /= tb%data(t)%valuesAB(:, 1:nNodO)), &
                       'R10t the solve changes the tracer (non-vacuous)', mype)
        end do
    end subroutine r10t_rest

    subroutine rest_inputs(d, Ki, hf, wf, vs, rs, sf)
        ! hashed Kv (> 0), Ki and surface fluxes for the stage-3 solve
        type(t_dyn), intent(inout) :: d
        real(kind=WP), allocatable, intent(out) :: Ki(:,:), hf(:), wf(:), vs(:), rs(:), sf(:)
        integer :: n, nz
        allocate(Ki(nl-1, nNodL), hf(nNodL), wf(nNodL), vs(nNodL), rs(nNodL), sf(nNodL))
        do n = 1, nNodL
            do nz = 1, nl
                d%work%Kv(nz, n) = 1.0e-4_WP*(1.5_WP + hval(partit%myList_nod2D(n), nz + 70))
            end do
            do nz = 1, nl-1
                Ki(nz, n) = 500.0_WP*(1.5_WP + hval(partit%myList_nod2D(n), nz + 90))
            end do
            hf(n) = 50.0_WP*hval(partit%myList_nod2D(n), 1)
            wf(n) = 1.0e-8_WP*hval(partit%myList_nod2D(n), 2)
            vs(n) = 1.0e-7_WP*hval(partit%myList_nod2D(n), 3)
            rs(n) = 1.0e-7_WP*hval(partit%myList_nod2D(n), 4)
            sf(n) = 0.0_WP
        end do
    end subroutine rest_inputs

    !===========================================================================
    subroutine r11_constancy()
        real(kind=WP) :: dev(2), devc1(2), devc2(2), ratio
        integer :: mype
        mype = partit%mype
        call r11_run(dt, 1.0e-3_WP, .false., dev)
        call check(dev(1) <= 1.0e-13_WP*7.3_WP,  'R11 three-stage constancy, FCT tracer', mype)
        call check(dev(2) <= 1.0e-13_WP*34.7_WP, 'R11 three-stage constancy, non-FCT tracer', mype)
        call r11_run(dt, 1.0e-3_WP, .true., devc1)
        call r11_run(0.5_WP*dt, 0.5e-3_WP, .true., devc2)
        ratio = devc1(2)/max(devc2(2), tiny(1.0_WP))
        call check(devc1(2) > 1.0e-10_WP, 'R11 positive control: w_e in stages 1-2 breaks constancy', mype)
        call check(abs(ratio - 4.0_WP) <= 1.0_WP, 'R11 positive control: the defect is O(tau^2)', mype)
        if (mype == 0) write(*,'(a,2es9.2,a,es9.2,a,es9.2,a,f6.3)') 'test_rk3: R11 max|T - T0| (FCT, not) =', dev, &
            '; control (w_e in stages 1-2): non-FCT tau ', devc1(2), ', tau/2 ', devc2(2), ', ratio ', ratio
    end subroutine r11_constancy

    subroutine r11_run(tau, maxcfl, control, dev)
        ! the three stages of the tracers on hashed u^n, u*, u** (each stage's w from its own
        ! velocity), uniform T0 = 7.3 (FCT) and 34.7 (not), split on; dev = max|T^{n+1} - T0|
        real(kind=WP), intent(in)  :: tau, maxcfl
        logical,       intent(in)  :: control
        real(kind=WP), intent(out) :: dev(2)
        type(t_dyn)    :: d
        type(t_tracer) :: tr
        real(kind=WP), allocatable :: Ki(:,:), hf(:), wf(:), vs(:), rs(:), sf(:), u2(:,:,:)
        integer :: nsplit
        call new_dyn(d, 2)
        call new_rk3(d)
        d%use_wsplit = .true.; d%wsplit_maxcfl = maxcfl; d%wsplit_mincfl = 0.5_WP*maxcfl
        call rest_inputs(d, Ki, hf, wf, vs, rs, sf)
        hf = 0.0_WP; wf = 0.0_WP; vs = 0.0_WP; rs = 0.0_WP
        call new_tracers(tr)
        call set_tracer(tr%data(1)%values, 7.3_WP, 0.0_WP, 0)
        call set_tracer(tr%data(2)%values, 34.7_WP, 0.0_WP, 0)
        call hashed_uv(d%uv, 121)
        allocate(u2(2, nl-1, nElemF))
        call hashed_uv(u2, 122)
        call hashed_uv(d%rk3_us, 123)
        ! stage 1 on u^n, stage 2 on u2 (u*), full w -- or w_e of the stage's split (control)
        call vert_vel_ale(d, mesh, tau, partit, split=control)
        if (control) then
            call rk3_tracer_stage(1, tau/3.0_WP, d%uv, d%w_e, d, tr, mesh, partit)
        else
            call rk3_tracer_stage(1, tau/3.0_WP, d%uv, d%w, d, tr, mesh, partit)
        end if
        call vert_vel_ale(d, mesh, tau, partit, uv_in=u2, split=control)
        if (control) then
            call rk3_tracer_stage(2, 0.5_WP*tau, u2, d%w_e, d, tr, mesh, partit)
        else
            call rk3_tracer_stage(2, 0.5_WP*tau, u2, d%w, d, tr, mesh, partit)
        end if
        ! stage 3 on u** (rk3_us) with the split
        call vert_vel_ale(d, mesh, tau, partit, uv_in=d%rk3_us)
        nsplit = count(d%w_i(:, 1:nNodO) /= 0.0_WP)
        call sum_all(nsplit, partit)
        call check(nsplit > 100, 'R11 precondition: split faces in stage 3', partit%mype)
        call solve_tracers_rk3(tau, 1, d, tr, mesh, Ki, hf, wf, vs, rs, sf, 0.0_WP, partit)
        dev(1) = maxdev(tr%data(1)%values, 7.3_WP)
        dev(2) = maxdev(tr%data(2)%values, 34.7_WP)
    end subroutine r11_run

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
