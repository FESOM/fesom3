module oce_adv_tra_driver
    ! Tracer-advection driver, transcribed from FESOM2 v2.7.3
    ! oce_adv_tra_driver.F90:46-419 (subroutine do_oce_adv_tra).
    !
    ! Assembles the byte-proven kernels (M1.1 horizontal, M1.2 vertical, M1.3 FCT
    ! limiter, the edge->node scatter) into the per-tracer advection operator. For a
    ! tracer with tra_adv_lim=='FCT' it runs the Zalesak path:
    !   LO = UPW1(values) horiz + vert            -> fct_LO  (low-order solution)
    !   HO = {hor scheme, ver scheme}(valuesAB)   -> antidiffusive flux (HO-LO)
    !   oce_tra_adv_fct clips the antidiffusive flux
    !   oce_tra_adv_flux2dtracer(use_lo) scatters -> del_ttf_advhoriz/advvert
    ! For a non-FCT tracer (do_zero_flux=.true.) the HO scheme is applied directly to
    ! valuesAB (o_init_zero=.true.) and scattered without the low-order reconstruction.
    !
    ! Dispatch is on the per-tracer scheme selectors (tracers%data(tr_num)%tra_adv_*):
    !   tra_adv_hor : UPW1 / MUSCL
    !   tra_adv_ver : UPW1 / QR4C            (CDIFF / PPM declared but not yet ported)
    !   tra_adv_lim : FCT or anything else (non-FCT)
    ! tra_adv_ph / tra_adv_pv are the horizontal / vertical 4th-order fractions
    ! (opth / optv) passed to MUSCL/QR4C.
    !
    ! SCOPE / clean-architecture deviations (D7, "USE-globals -> explicit args"):
    !  * 1-rank only (myDim_* == global). The two FESOM2 halo exchanges
    !    (exchange_nod(fct_LO) after the LO build, the implicit ones in the kernels)
    !    are no-ops at 1 rank and are dropped here; multi-rank is M1.5.
    !  * use_wsplit (w = w_e + w_i, oce_wsplit / compute_Wvel_split): the FCT low-order
    !    solution is advanced explicitly with w_e, then implicitly with w_i
    !    (adv_tra_vert_impl, upwind backward Euler on fct_LO), and the low-order upwind
    !    flux is RECOMPUTED with the full w so that the antidiffusive flux is
    !    HO(w) - LO(w) (FESOM2 :282-292; wiring pinned by test_wimpl_tra C6). With
    !    use_wsplit=.false. w_e == w and w_i == 0 (compute_Wvel_split), the branch is
    !    skipped and the LO/HO vertical fluxes use the same field, as in FESOM2.
    !  * the DVD (downgradient-variance-decomposition) diagnostic blocks are dropped (not v1).
    !  * work arrays live in tracers%work (MP); kernels take WP. At the DP/SP anchor
    !    MP==WP so the pointers/args bind by kind value; revisit for FP16 (D3).
    !
    ! do_oce_adv_tra_rk3 (docs/plans/2026-10-28-rk3.md, D6/D7, Technical Details §5) is the RK3
    ! counterpart for what do_oce_adv_tra cannot do; do_oce_adv_tra itself stays untouched:
    !  * stages 1-2: the HIGH ORDER only (no low order, no limiter) on the stage tracer in
    !    valuesAB, horizontal MUSCL/UPW1 and vertical QR4C/UPW1 with the FULL w for every
    !    tracer, FCT or not (no w split before stage 3), fluxes from zero, then the scatter;
    !  * stage 3, FCT tracers: the low order in N substeps of dt/N (FESOM3_RK3_LO_NSUB), each
    !    explicit with w_e from T^(k) and implicit with w_i (the stage-3 split), the low-order
    !    fluxes (horizontal; vertical with the full w) averaged over the substeps; then the
    !    high order on valuesAB (T**) minus that mean as the antidiffusive flux, the limiter
    !    against T^n and the low-order solution T^(N), and the scatter. Under zstar the substeps
    !    run on the interpolated thicknesses h^(k) = h^n + (k/N)(h^{n+1} - h^n) (h^(0) = hnode,
    !    h^(N) = hnode_new exactly): each goes h^(k) -> h^(k+1) with 1/N of the transport, so the
    !    substeps keep constants. At N = 1 this is
    !    do_oce_adv_tra's FCT path bitwise (test_rk3 R6); the RK3 step itself calls
    !    do_oce_adv_tra there and for every non-FCT tracer (stage 3 with N = 1 is today's
    !    operator on T**).
    use mod_precision,    only: WP, WP_full
    use mod_mesh,         only: t_mesh
    use mod_dyn,          only: t_dyn
    use mod_tracer,       only: t_tracer
    use mod_partit,       only: t_partit
    use mod_part_bounds,  only: owned_bounds, is_multirank
    use mod_halo,         only: exchange_nod
    use oce_adv_tra_hor,  only: adv_tra_hor_upw1, adv_tra_hor_muscl
    use oce_adv_tra_ver,  only: adv_tra_ver_upw1, adv_tra_ver_qr4c, adv_tra_vert_impl
    use oce_adv_tra_fct,  only: oce_tra_adv_fct
    use oce_adv_tra_flux, only: oce_tra_adv_flux2dtracer
    implicit none
    private
    public :: do_oce_adv_tra, do_oce_adv_tra_rk3

contains

    subroutine do_oce_adv_tra(dt, vel, w, wi, we, tr_num, dynamics, tracers, mesh, partit)
        ! do_oce_adv_tra (oce_adv_tra_driver.F90:46-419). vel = horizontal velocity at
        ! elements; w / we / wi = full / explicit / implicit vertical velocity at nodes.
        ! M2.12b: optional partit -> owned/halo loop bounds + exchange_nod(fct_LO) after
        ! the low-order solution (FESOM2 :294), passed on to every kernel.
        real(kind=WP),  intent(in)            :: dt
        type(t_mesh),   intent(in)            :: mesh
        type(t_dyn),    intent(in)            :: dynamics
        type(t_tracer), intent(inout), target :: tracers
        integer,        intent(in)            :: tr_num
        real(kind=WP),  intent(in)            :: vel(2, mesh%nl-1, mesh%elem2D)
        real(kind=WP),  intent(in), target    :: w (mesh%nl, mesh%nod2D)
        real(kind=WP),  intent(in)            :: wi(mesh%nl, mesh%nod2D)
        real(kind=WP),  intent(in), target    :: we(mesh%nl, mesh%nod2D)
        type(t_partit), intent(in), optional  :: partit

        ! pointers into the per-tracer data / shared work (mirrors FESOM2 lines 92-106)
        real(kind=WP), pointer :: ttf(:,:), ttfAB(:,:), fct_LO(:,:)
        real(kind=WP), pointer :: adv_flux_hor(:,:), adv_flux_ver(:,:)
        real(kind=WP_full), pointer :: dttf_h(:,:), dttf_v(:,:), fct_LO_tend(:,:)
        real(kind=WP), pointer :: fct_ttf_min(:,:), fct_ttf_max(:,:), fct_plus(:,:), fct_minus(:,:)
        real(kind=WP), pointer :: tr_xy(:,:,:)
        integer,       pointer :: edge_up_dn_tri(:,:)
        real(kind=WP), pointer :: pwvel(:,:)

        real(kind=WP) :: opth, optv
        logical       :: do_zero_flux
        integer       :: e, n, nz, enodes(2), el(2), nl1, nl2, nu1, nu2, nl12, nu12
        integer       :: nNodO, nNodL, nEdgeO, nElemO

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)

        ttf             => tracers%data(tr_num)%values
        ttfAB           => tracers%data(tr_num)%valuesAB
        opth            =  tracers%data(tr_num)%tra_adv_ph
        optv            =  tracers%data(tr_num)%tra_adv_pv
        fct_LO          => tracers%work%fct_LO
        adv_flux_ver    => tracers%work%adv_flux_ver
        adv_flux_hor    => tracers%work%adv_flux_hor
        tr_xy           => tracers%work%tr_xy
        edge_up_dn_tri  => tracers%work%edge_up_dn_tri
        fct_ttf_min     => tracers%work%fct_ttf_min
        fct_ttf_max     => tracers%work%fct_ttf_max
        fct_plus        => tracers%work%fct_plus
        fct_minus       => tracers%work%fct_minus
        dttf_h          => tracers%work%del_ttf_advhoriz
        dttf_v          => tracers%work%del_ttf_advvert
#if defined(USE_SINGLE_PRECISION)
        ! allocated on first use, shaped as fct_LO, so every driver gets it
        if (.not. allocated(tracers%work%fct_LO_tend)) then
            allocate(tracers%work%fct_LO_tend(size(fct_LO,1), size(fct_LO,2)))
            tracers%work%fct_LO_tend = 0.0_WP_full
        end if
        fct_LO_tend     => tracers%work%fct_LO_tend
#endif

        !_______________________________________________________________________
        ! FCT: low-order horizontal+vertical solution + low-order antidiffusive part
        if (trim(tracers%data(tr_num)%tra_adv_lim) == 'FCT') then
            ! low-order upwind horizontal flux (zero the flux first)
            call adv_tra_hor_upw1(vel, ttf, mesh, adv_flux_hor, o_init_zero=.true., partit=partit)

#if defined(USE_SINGLE_PRECISION)
            ! SINGLE PRECISION (upstream #1054/#1056/#1206): the low-order tendency is kept in
            ! FLUX form in the WP_full fct_LO_tend. Recovering it as LO*hnode_new - ttf*hnode
            ! from the real32 LO loses every per-step change below half an ulp of the tracer,
            ! and the vertical advection stops telescoping (upstream: -0.5 W/m2 heat in SP).
            ! Double precision keeps the reconstruction below (upstream changed it in DP too,
            ! #1056, which changes the DP rounding; FESOM_F keeps DP bit-identical).
            do n = 1, nNodL
                do nz = 1, mesh%nl-1
                    fct_LO_tend(nz,n) = 0.0_WP_full
                    fct_LO(nz,n)      = 0.0_WP      ! as the DP path: no stale entries off the column
                end do
            end do
            do e = 1, nEdgeO
                enodes = mesh%edges(:, e)
                el     = mesh%edge_tri(:, e)
                nl1 = mesh%nlevels(el(1))-1
                nu1 = mesh%ulevels(el(1))
                nl2 = 0; nu2 = 0
                if (el(2) /= el(1)) then
                    nl2 = mesh%nlevels(el(2))-1
                    nu2 = mesh%ulevels(el(2))
                end if
                nl12 = max(nl1, nl2)
                nu12 = nu1
                if (nu2 > 0) nu12 = min(nu1, nu2)
                do nz = nu12, nl12
                    fct_LO_tend(nz, enodes(1)) = fct_LO_tend(nz, enodes(1)) + adv_flux_hor(nz, e)
                    fct_LO_tend(nz, enodes(2)) = fct_LO_tend(nz, enodes(2)) - adv_flux_hor(nz, e)
                end do
            end do
            call adv_tra_ver_upw1(we, ttf, mesh, adv_flux_ver, o_init_zero=.true., partit=partit)
            do n = 1, nNodO
                nu1 = mesh%ulevels_nod2D(n)
                nl1 = mesh%nlevels_nod2D(n)
                do nz = nu1, nl1-1
                    fct_LO_tend(nz,n) = (fct_LO_tend(nz,n) + (real(adv_flux_ver(nz,n), WP_full) &
                                       - adv_flux_ver(nz+1,n)))*dt/mesh%areasvol(n)
                    fct_LO(nz,n) = (ttf(nz,n)*mesh%hnode(nz,n) + fct_LO_tend(nz,n))/mesh%hnode_new(nz,n)
                end do
            end do
            if (dynamics%use_wsplit) then
                call adv_tra_vert_impl(dt, wi, fct_LO, mesh, partit=partit, tend=fct_LO_tend)
                call adv_tra_ver_upw1(w, ttf, mesh, adv_flux_ver, o_init_zero=.true., partit=partit)
            end if
#else
            ! fct_LO = scatter of the LO horizontal flux over edges
            do n = 1, nNodL
                do nz = 1, mesh%nl-1
                    fct_LO(nz,n) = 0.0_WP
                end do
            end do
            do e = 1, nEdgeO
                enodes = mesh%edges(:, e)
                el     = mesh%edge_tri(:, e)
                nl1 = mesh%nlevels(el(1))-1
                nu1 = mesh%ulevels(el(1))
                nl2 = 0; nu2 = 0
                if (el(2) /= el(1)) then
                    nl2 = mesh%nlevels(el(2))-1
                    nu2 = mesh%ulevels(el(2))
                end if
                nl12 = max(nl1, nl2)
                nu12 = nu1
                if (nu2 > 0) nu12 = min(nu1, nu2)
                do nz = nu12, nl12
                    fct_LO(nz, enodes(1)) = fct_LO(nz, enodes(1)) + adv_flux_hor(nz, e)
                    fct_LO(nz, enodes(2)) = fct_LO(nz, enodes(2)) - adv_flux_hor(nz, e)
                end do
            end do

            ! low-order upwind vertical flux (explicit part: we), then finish the LO solution
            call adv_tra_ver_upw1(we, ttf, mesh, adv_flux_ver, o_init_zero=.true., partit=partit)
            do n = 1, nNodO
                nu1 = mesh%ulevels_nod2D(n)
                nl1 = mesh%nlevels_nod2D(n)
                do nz = nu1, nl1-1
                    fct_LO(nz,n) = (ttf(nz,n)*mesh%hnode(nz,n) &
                                  + (fct_LO(nz,n) + (adv_flux_ver(nz,n)-adv_flux_ver(nz+1,n)))*dt/mesh%areasvol(n)) &
                                  / mesh%hnode_new(nz,n)
                end do
            end do

            if (dynamics%use_wsplit) then
                ! w-split (FESOM2 :282-292): the implicit part w_i advects the low-order
                ! solution implicitly (upwind backward Euler, in place), then the low-order
                ! upwind vertical flux is recomputed with the FULL w, so that the HO call
                ! below (o_init_zero=.false.) leaves HO(w) - LO(w) as the antidiffusive flux.
                ! (LO(w_e) there would double-count the w_i transport; test_wimpl_tra C6.)
                call adv_tra_vert_impl(dt, wi, fct_LO, mesh, partit=partit)
                call adv_tra_ver_upw1(w, ttf, mesh, adv_flux_ver, o_init_zero=.true., partit=partit)
            end if
#endif
            ! M2.12b: share the low-order solution to the halo (FESOM2 :294) — the FCT
            ! limiter's a1 reads fct_LO at owned+halo nodes.
            if (is_multirank(partit)) call exchange_nod(fct_LO, partit)
        end if

        do_zero_flux = .true.
        if (trim(tracers%data(tr_num)%tra_adv_lim) == 'FCT') do_zero_flux = .false.

        !_______________________________________________________________________
        ! horizontal advection (FCT: high-order antidiffusive; else: full high-order)
        select case (trim(tracers%data(tr_num)%tra_adv_hor))
        case ('MUSCL')
            call adv_tra_hor_muscl(vel, ttfAB, mesh, opth, adv_flux_hor, tr_xy, edge_up_dn_tri, o_init_zero=do_zero_flux, partit=partit)
        case ('UPW1')
            call adv_tra_hor_upw1 (vel, ttfAB, mesh,       adv_flux_hor,                                 o_init_zero=do_zero_flux, partit=partit)
        case default
            error stop 'do_oce_adv_tra: unknown tra_adv_hor (expected UPW1/MUSCL)'
        end select

        if (trim(tracers%data(tr_num)%tra_adv_lim) == 'FCT') then
            pwvel => w
        else
            pwvel => we
        end if

        !_______________________________________________________________________
        ! vertical advection (FCT: high-order antidiffusive; else: full high-order)
        select case (trim(tracers%data(tr_num)%tra_adv_ver))
        case ('QR4C')
            call adv_tra_ver_qr4c(pwvel, ttfAB, mesh, optv, adv_flux_ver, o_init_zero=do_zero_flux, partit=partit)
        case ('UPW1')
            call adv_tra_ver_upw1(pwvel, ttfAB, mesh,       adv_flux_ver, o_init_zero=do_zero_flux, partit=partit)
        case ('CDIFF', 'PPM')
            error stop 'do_oce_adv_tra: tra_adv_ver CDIFF/PPM not yet ported (M2)'
        case default
            error stop 'do_oce_adv_tra: unknown tra_adv_ver (expected UPW1/QR4C)'
        end select

        !_______________________________________________________________________
        ! limit (FCT) and scatter to the per-node tendencies del_ttf_advhoriz/advvert
        if (trim(tracers%data(tr_num)%tra_adv_lim) == 'FCT') then
            call oce_tra_adv_fct(dt, ttf, fct_LO, adv_flux_hor, adv_flux_ver, &
                                 fct_ttf_min, fct_ttf_max, fct_plus, fct_minus, mesh, partit=partit)
#if defined(USE_SINGLE_PRECISION)
            call oce_tra_adv_flux2dtracer(dt, dttf_h, dttf_v, adv_flux_hor, adv_flux_ver, mesh, &
                                          partit=partit, lo_tend=fct_LO_tend)
#else
            call oce_tra_adv_flux2dtracer(dt, dttf_h, dttf_v, adv_flux_hor, adv_flux_ver, mesh, &
                                          use_lo=.true., ttf=ttf, lo=fct_LO, partit=partit)
#endif
        else
            call oce_tra_adv_flux2dtracer(dt, dttf_h, dttf_v, adv_flux_hor, adv_flux_ver, mesh, partit=partit)
        end if
    end subroutine do_oce_adv_tra

    !===========================================================================
    subroutine do_oce_adv_tra_rk3(dt, vel, w, wi, we, tr_num, stage, nsub, dynamics, tracers, mesh, partit)
        ! RK3 tracer advection (module header). dt = the stage step tau_s (tau/3, tau/2, tau);
        ! vel = the stage velocity at elements; w / we / wi = full / explicit / implicit vertical
        ! velocity (stages 1-2 read w only). Accumulates into del_ttf_advhoriz / del_ttf_advvert
        ! (the caller zeroes them, rk3_tracer_prep). nsub = the low-order substeps of stage 3.
        real(kind=WP),  intent(in)            :: dt
        type(t_mesh),   intent(in)            :: mesh
        type(t_dyn),    intent(in)            :: dynamics
        type(t_tracer), intent(inout), target :: tracers
        integer,        intent(in)            :: tr_num, stage, nsub
        real(kind=WP),  intent(in)            :: vel(2, mesh%nl-1, mesh%elem2D)
        real(kind=WP),  intent(in)            :: w (mesh%nl, mesh%nod2D)
        real(kind=WP),  intent(in)            :: wi(mesh%nl, mesh%nod2D)
        real(kind=WP),  intent(in)            :: we(mesh%nl, mesh%nod2D)
        type(t_partit), intent(in), optional  :: partit

        real(kind=WP), pointer :: ttf(:,:), ttfAB(:,:), fct_LO(:,:)
        real(kind=WP), pointer :: adv_flux_hor(:,:), adv_flux_ver(:,:)
        real(kind=WP_full), pointer :: dttf_h(:,:), dttf_v(:,:)
        real(kind=WP), pointer :: tr_xy(:,:,:)
        real(kind=WP), pointer :: lo_T(:,:), lo_fh(:,:), lo_fv(:,:), lo_top(:)
        real(kind=WP_full), allocatable :: hk(:,:), hk1(:,:)
        real(kind=WP) :: opth, optv, rn, dts
        integer       :: k, e, n, nz, enodes(2), el(2), nl1, nl2, nu1, nu2, nl12, nu12
        integer       :: nNodO, nNodL, nEdgeO, nElemO

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        ttf          => tracers%data(tr_num)%values
        ttfAB        => tracers%data(tr_num)%valuesAB
        opth         =  tracers%data(tr_num)%tra_adv_ph
        optv         =  tracers%data(tr_num)%tra_adv_pv
        fct_LO       => tracers%work%fct_LO
        adv_flux_ver => tracers%work%adv_flux_ver
        adv_flux_hor => tracers%work%adv_flux_hor
        tr_xy        => tracers%work%tr_xy
        dttf_h       => tracers%work%del_ttf_advhoriz
        dttf_v       => tracers%work%del_ttf_advvert

        if (stage < 1 .or. stage > 3) error stop 'do_oce_adv_tra_rk3: stage must be 1, 2 or 3'
        if (stage < 3) then
            ! stages 1-2: the high order of valuesAB with the full w, from zero, scattered
            call rk3_ho_fluxes(.true.)
            call oce_tra_adv_flux2dtracer(dt, dttf_h, dttf_v, adv_flux_hor, adv_flux_ver, mesh, partit=partit)
            return
        end if

        ! stage 3: a non-FCT tracer is today's operator on valuesAB (T**)
        if (trim(tracers%data(tr_num)%tra_adv_lim) /= 'FCT') then
            call do_oce_adv_tra(dt, vel, w, wi, we, tr_num, dynamics, tracers, mesh, partit)
            return
        end if
        if (nsub < 1) error stop 'do_oce_adv_tra_rk3: nsub must be >= 1'

        ! ---- FCT: the low order in nsub substeps of dt/nsub (Technical Details §5) ----
        if (.not. allocated(tracers%work%rk3_lo_T)) then
            allocate(tracers%work%rk3_lo_T (size(fct_LO,1), size(fct_LO,2)))
            allocate(tracers%work%rk3_lo_fh(size(adv_flux_hor,1), size(adv_flux_hor,2)))
            allocate(tracers%work%rk3_lo_fv(size(adv_flux_ver,1), size(adv_flux_ver,2)))
            allocate(tracers%work%rk3_lo_top(size(fct_LO,2)))
        end if
        lo_T   => tracers%work%rk3_lo_T
        lo_fh  => tracers%work%rk3_lo_fh
        lo_fv  => tracers%work%rk3_lo_fv
        lo_top => tracers%work%rk3_lo_top
        rn  = 1.0_WP/real(nsub, WP)
        dts = dt/real(nsub, WP)
        do n = 1, nNodL
            lo_T(:, n)  = ttf(:, n)              ! T^(0) = T^n, halo-valid
            lo_fv(:, n) = 0.0_WP
            lo_top(n)   = 0.0_WP
        end do
        do e = 1, nEdgeO
            lo_fh(:, e) = 0.0_WP
        end do
        ! the substep thicknesses h^(k-1) -> h^(k) (both hnode under linfs)
        allocate(hk(size(fct_LO,1), size(fct_LO,2)), hk1(size(fct_LO,1), size(fct_LO,2)))
        hk1 = mesh%hnode(:, 1:size(fct_LO,2))

        do k = 1, nsub
            hk = hk1
            if (k == nsub) then
                hk1 = mesh%hnode_new(:, 1:size(fct_LO,2))
            else
                hk1 = mesh%hnode(:, 1:size(fct_LO,2)) + (real(k, WP)/real(nsub, WP)) &
                    *(mesh%hnode_new(:, 1:size(fct_LO,2)) - mesh%hnode(:, 1:size(fct_LO,2)))
            end if
            ! explicit: horizontal upwind of T^(k-1) scattered into fct_LO, vertical upwind with w_e
            call adv_tra_hor_upw1(vel, lo_T, mesh, adv_flux_hor, o_init_zero=.true., partit=partit)
            do n = 1, nNodL
                do nz = 1, mesh%nl-1
                    fct_LO(nz,n) = 0.0_WP
                end do
            end do
            do e = 1, nEdgeO
                enodes = mesh%edges(:, e)
                el     = mesh%edge_tri(:, e)
                nl1 = mesh%nlevels(el(1))-1
                nu1 = mesh%ulevels(el(1))
                nl2 = 0; nu2 = 0
                if (el(2) /= el(1)) then
                    nl2 = mesh%nlevels(el(2))-1
                    nu2 = mesh%ulevels(el(2))
                end if
                nl12 = max(nl1, nl2)
                nu12 = nu1
                if (nu2 > 0) nu12 = min(nu1, nu2)
                do nz = nu12, nl12
                    fct_LO(nz, enodes(1)) = fct_LO(nz, enodes(1)) + adv_flux_hor(nz, e)
                    fct_LO(nz, enodes(2)) = fct_LO(nz, enodes(2)) - adv_flux_hor(nz, e)
                end do
            end do
            call adv_tra_ver_upw1(we, lo_T, mesh, adv_flux_ver, o_init_zero=.true., partit=partit)
            do n = 1, nNodO
                nu1 = mesh%ulevels_nod2D(n)
                nl1 = mesh%nlevels_nod2D(n)
                do nz = nu1, nl1-1
                    fct_LO(nz,n) = (lo_T(nz,n)*hk(nz,n) &
                                  + (fct_LO(nz,n) + (adv_flux_ver(nz,n)-adv_flux_ver(nz+1,n)))*dts/mesh%areasvol(n)) &
                                  / hk1(nz,n)
                end do
            end do
            ! implicit: w_i over dt/nsub, then the low-order vertical flux with the full w
            if (dynamics%use_wsplit) then
                call adv_tra_vert_impl(dts, wi, fct_LO, mesh, partit=partit, hnew=hk1)
                call adv_tra_ver_upw1(w, lo_T, mesh, adv_flux_ver, o_init_zero=.true., partit=partit)
            end if
            ! the substep means: low-order fluxes and the surface transport
            do e = 1, nEdgeO
                lo_fh(:, e) = lo_fh(:, e) + rn*adv_flux_hor(:, e)
            end do
            do n = 1, nNodO
                lo_fv(:, n) = lo_fv(:, n) + rn*adv_flux_ver(:, n)
                nz = mesh%ulevels_nod2D(n)
                lo_top(n) = lo_top(n) + rn*mesh%area(n)*(we(nz,n)*lo_T(nz,n) + wi(nz,n)*fct_LO(nz,n))
            end do
            ! T^(k) = the low-order solution, halo-valid for the next substep and the limiter
            do n = 1, nNodO
                lo_T(:, n) = fct_LO(:, n)
            end do
            if (is_multirank(partit)) call exchange_nod(lo_T, partit)
        end do
        do n = 1, nNodL
            fct_LO(:, n) = lo_T(:, n)
        end do
        do e = 1, nEdgeO
            adv_flux_hor(:, e) = lo_fh(:, e)
        end do
        do n = 1, nNodO
            adv_flux_ver(:, n) = lo_fv(:, n)
        end do

        ! antidiffusive flux = HO(T**, full w) - mean LO, the limiter, the scatter
        call rk3_ho_fluxes(.false.)
        call oce_tra_adv_fct(dt, ttf, fct_LO, adv_flux_hor, adv_flux_ver, &
                             tracers%work%fct_ttf_min, tracers%work%fct_ttf_max, &
                             tracers%work%fct_plus, tracers%work%fct_minus, mesh, partit=partit)
        call oce_tra_adv_flux2dtracer(dt, dttf_h, dttf_v, adv_flux_hor, adv_flux_ver, mesh, &
                                      use_lo=.true., ttf=ttf, lo=fct_LO, partit=partit)

    contains

        subroutine rk3_ho_fluxes(init_zero)
            ! the high-order fluxes of valuesAB with the FULL w: from zero (init_zero) or as
            ! high order minus the fluxes already in adv_flux_hor / adv_flux_ver
            logical, intent(in) :: init_zero
            select case (trim(tracers%data(tr_num)%tra_adv_hor))
            case ('MUSCL')
                call adv_tra_hor_muscl(vel, ttfAB, mesh, opth, adv_flux_hor, tr_xy, tracers%work%edge_up_dn_tri, &
                                       o_init_zero=init_zero, partit=partit)
            case ('UPW1')
                call adv_tra_hor_upw1(vel, ttfAB, mesh, adv_flux_hor, o_init_zero=init_zero, partit=partit)
            case default
                error stop 'do_oce_adv_tra_rk3: unknown tra_adv_hor (expected UPW1/MUSCL)'
            end select
            select case (trim(tracers%data(tr_num)%tra_adv_ver))
            case ('QR4C')
                call adv_tra_ver_qr4c(w, ttfAB, mesh, optv, adv_flux_ver, o_init_zero=init_zero, partit=partit)
            case ('UPW1')
                call adv_tra_ver_upw1(w, ttfAB, mesh, adv_flux_ver, o_init_zero=init_zero, partit=partit)
            case default
                error stop 'do_oce_adv_tra_rk3: unknown tra_adv_ver (expected UPW1/QR4C)'
            end select
        end subroutine rk3_ho_fluxes

    end subroutine do_oce_adv_tra_rk3

end module oce_adv_tra_driver
