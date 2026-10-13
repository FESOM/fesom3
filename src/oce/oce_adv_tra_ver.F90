module oce_adv_tra_ver
    ! Vertical tracer advection — interface fluxes, transcribed from FESOM2 v2.7.3
    ! oce_adv_tra_ver.F90:
    !   adv_tra_ver_upw1 (244-328)  1st-order upwind (explicit)
    !   adv_tra_ver_qr4c (332-434)  QR 4th-order centered (num_ord = 4th-order fraction)
    !   adv_tra_vert_impl (90-240)  1st-order upwind, IMPLICIT, with the implicit part w_i
    !                               of the split vertical velocity (use_wsplit); acts in
    !                               place on the FCT low-order solution, no flux output
    !
    ! The two explicit kernels return a flux given at the vertical interfaces of the
    ! scalar volumes, flux(nz,node) with nz = 1..nl interfaces (surface = ulevels_nod2D,
    ! zero at the bottom interface nlevels_nod2D); adv_tra_vert_impl updates ttf in place.
    ! o_init_zero=.true. zeroes the flux first; .false. SUBTRACTS the new contribution
    ! from the input flux (so an HO call after an LO call yields the antidiffusive flux).
    ! flux is NOT multiplied by dt — the driver scatters it as
    ! (flux(nz)-flux(nz+1))*dt/areasvol (oce_adv_tra_flux).
    !
    ! area(nz,n) is the geom-proven node control-volume area. Z_3d_n/zbar_3d_n are the
    ! ALE per-node mid-depth / interface-depth arrays; at the initial state (no cavity,
    ! full cells) the caller builds them from FESOM2 init_ale (oce_ale.F90:531-566) and
    ! the gate verifies them. QR4C's qc/qu/qd divide by (Z_3d_n(k)-Z_3d_n(k+1)), a
    ! RUNTIME divisor — FESOM2 does the identical division on byte-identical operands,
    ! so the -no-prec-div reciprocal matches (cf. LESSONS L7: the trap is a runtime vs
    ! LITERAL divisor mismatch, not a runtime divisor per se).
    !
    ! The explicit kernels were ported 1-rank (myDim_nod2D == nod2D) and later given the
    ! optional partit -> owned-node loop; adv_tra_vert_impl is owned-loop via owned_bounds
    ! from the start (the driver's exchange_nod(fct_LO) follows it). On pi the shallowest
    ! column is 4 layers, so neither the 1-layer ttf(0) read nor the QR4C 2-layer
    ! double-write (2nd-layer and bottom-1 both hitting interface nzmin+1) occurs; the
    ! statement order below still reproduces both faithfully for deeper-min meshes.
    use mod_precision,   only: WP
    use mod_mesh,        only: t_mesh
    use mod_partit,      only: t_partit
    use mod_part_bounds, only: owned_bounds
    implicit none
    private
    public :: adv_tra_ver_upw1, adv_tra_ver_qr4c, adv_tra_vert_impl

contains

    !===========================================================================
    subroutine adv_tra_ver_upw1(w, ttf, mesh, flux, o_init_zero, partit)
        ! 1st-order upwind explicit vertical flux (oce_adv_tra_ver.F90:244-328).
        ! M2.12b: optional partit -> loop over OWNED nodes (myDim_nod2D).
        type(t_mesh),  intent(in)    :: mesh
        real(kind=WP), intent(in)    :: ttf(mesh%nl-1, mesh%nod2D)
        real(kind=WP), intent(in)    :: w  (mesh%nl,   mesh%nod2D)
        real(kind=WP), intent(inout) :: flux(mesh%nl,  mesh%nod2D)
        logical, optional, intent(in) :: o_init_zero
        type(t_partit), intent(in), optional :: partit
        logical :: l_init_zero
        integer :: n, nz, nzmax, nzmin
        integer :: nNodO, nNodL, nEdgeO, nElemO

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        l_init_zero = .true.
        if (present(o_init_zero)) l_init_zero = o_init_zero
        if (l_init_zero) then
            do n = 1, nNodO
                do nz = 1, mesh%nl
                    flux(nz, n) = 0.0_WP
                end do
            end do
        end if

        do n = 1, nNodO
            nzmax = mesh%nlevels_nod2D(n)
            nzmin = mesh%ulevels_nod2D(n)
            ! vert. flux at surface layer
            nz = nzmin
            flux(nz,n) = -w(nz,n)*ttf(nz,n)*mesh%area(n) - flux(nz,n)
            ! vert. flux at bottom layer --> zero bottom flux
            nz = nzmax
            flux(nz,n) = 0.0_WP - flux(nz,n)
            ! vert. flux at remaining levels (upwind)
            do nz = nzmin+1, nzmax-1
                flux(nz,n) = -0.5*(                                            &
                              ttf(nz  ,n)*(w(nz,n)+abs(w(nz,n))) +             &
                              ttf(nz-1,n)*(w(nz,n)-abs(w(nz,n))))*mesh%area(n) - flux(nz,n)
            end do
        end do
    end subroutine adv_tra_ver_upw1

    !===========================================================================
    subroutine adv_tra_ver_qr4c(w, ttf, mesh, num_ord, flux, o_init_zero, partit)
        ! QR 4th-order centered vertical flux (oce_adv_tra_ver.F90:332-434). num_ord =
        ! fraction of the 4th-order (centered) contribution; (1-num_ord) is upwind-QR.
        ! M2.12b: optional partit -> loop over OWNED nodes (myDim_nod2D).
        type(t_mesh),  intent(in)    :: mesh
        real(kind=WP), intent(in)    :: num_ord
        real(kind=WP), intent(in)    :: ttf(mesh%nl-1, mesh%nod2D)
        real(kind=WP), intent(in)    :: w  (mesh%nl,   mesh%nod2D)
        real(kind=WP), intent(inout) :: flux(mesh%nl,  mesh%nod2D)
        logical, optional, intent(in) :: o_init_zero
        type(t_partit), intent(in), optional :: partit
        logical :: l_init_zero
        integer :: n, nz, nzmax, nzmin
        real(kind=WP) :: Tmean, Tmean1, Tmean2, qc, qu, qd
        integer :: nNodO, nNodL, nEdgeO, nElemO

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        l_init_zero = .true.
        if (present(o_init_zero)) l_init_zero = o_init_zero
        if (l_init_zero) then
            do n = 1, nNodO
                do nz = 1, mesh%nl
                    flux(nz, n) = 0.0_WP
                end do
            end do
        end if

        do n = 1, nNodO
            nzmax = mesh%nlevels_nod2D(n)
            nzmin = mesh%ulevels_nod2D(n)
            ! vert. flux at surface layer
            nz = nzmin
            flux(nz,n) = -ttf(nz,n)*w(nz,n)*mesh%area(n) - flux(nz,n)
            ! vert. flux 2nd layer --> centered differences
            nz = nzmin+1
            flux(nz,n) = -0.5_WP*(ttf(nz-1,n)+ttf(nz,n))*w(nz,n)*mesh%area(n) - flux(nz,n)
            ! vert. flux at bottom-1 layer --> centered differences
            nz = nzmax-1
            flux(nz,n) = -0.5_WP*(ttf(nz-1,n)+ttf(nz,n))*w(nz,n)*mesh%area(n) - flux(nz,n)
            ! vert. flux at bottom layer --> zero bottom flux
            nz = nzmax
            flux(nz,n) = 0.0_WP - flux(nz,n)
            ! vert. flux at remaining levels (4th order)
            do nz = nzmin+2, nzmax-2
                qc = (ttf(nz-1,n)-ttf(nz  ,n))/(mesh%Z_3d_n(nz-1,n)-mesh%Z_3d_n(nz  ,n))
                qu = (ttf(nz  ,n)-ttf(nz+1,n))/(mesh%Z_3d_n(nz  ,n)-mesh%Z_3d_n(nz+1,n))
                qd = (ttf(nz-2,n)-ttf(nz-1,n))/(mesh%Z_3d_n(nz-2,n)-mesh%Z_3d_n(nz-1,n))

                Tmean1 = ttf(nz  ,n)+(2*qc+qu)*(mesh%zbar_3d_n(nz,n)-mesh%Z_3d_n(nz  ,n))/3.0_WP
                Tmean2 = ttf(nz-1,n)+(2*qc+qd)*(mesh%zbar_3d_n(nz,n)-mesh%Z_3d_n(nz-1,n))/3.0_WP
                Tmean  = (w(nz,n)+abs(w(nz,n)))*Tmean1+(w(nz,n)-abs(w(nz,n)))*Tmean2
                flux(nz,n) = (-0.5_WP*(1.0_WP-num_ord)*Tmean - num_ord*(0.5_WP*(Tmean1+Tmean2))*w(nz,n))*mesh%area(n) - flux(nz,n)
            end do
        end do
    end subroutine adv_tra_ver_qr4c

    !===========================================================================
    subroutine adv_tra_vert_impl(dt, w, ttf, mesh, partit)
        ! Implicit (backward-Euler) 1st-order upwind vertical advection of the OWNED
        ! columns with the implicit part w_i of the split vertical velocity (use_wsplit,
        ! oce_wsplit / compute_Wvel_split), applied IN PLACE to the FCT low-order solution
        ! fct_LO. Port of FESOM2 v2.7.3 oce_adv_tra_ver.F90:90-240 (adv_tra_vert_impl);
        ! do_oce_adv_tra calls it after the explicit low-order step with w_e and before
        ! the low-order upwind flux is recomputed with the FULL w (FESOM2
        ! oce_adv_tra_driver.F90:282-292). WHY: the explicit step can only carry a vertical
        ! Courant number up to wsplit_maxcfl; the remainder w_i of the face velocity is
        ! taken implicitly here so that thin z* layers / strong convection do not force
        ! the time step.
        !
        ! Per column, v = dt*area(n)/areasvol(n) (f3: the 1-D area(n)/areasvol(n) rule of
        ! diff_ver_part_impl_ale, where FESOM2 forms v_adv per face from area(nz,n)/
        ! areasvol(nz,n)), h' = hnode_new:
        !   surface  a = 0              b = h' + w(nz)*v - min(0,w(nz+1))*v   c = -max(0,w(nz+1))*v
        !   interior a = min(0,w(nz))*v b = h' + max(0,w(nz))*v - min(0,w(nz+1))*v
        !                                                                   c = -max(0,w(nz+1))*v
        !   bottom   a = min(0,w(nz))*v b = h' + max(0,w(nz))*v              c = 0
        !   rhs      tr = -a*T(nz-1) - (b-h')*T(nz) - c*T(nz+1);  solve;  T = T + tr
        ! i.e. M*T^{n+1} = h'*T: upwind FLUX form on the NEW thickness, the thickness the
        ! explicit step has already advanced with the full w (the constancy proof is in
        ! test_wimpl_tra C2: for ANY split with w_e + w_i = w a uniform T stays uniform).
        ! Column sums of M are h' except the surface column, h' + w(nzmin)*v: the UNSIGNED
        ! surface transport, the implicit twin of the explicit -w*T*area of adv_tra_ver_upw1
        ! (L57: an implicit operator inherits the form of the explicit scheme it completes),
        ! so the solve conserves h'*T up to that surface flux. The off-diagonals are
        ! non-positive and the ROW margin is
        !     b - |a| - |c| = h' + (w_i(nz) - w_i(nz+1))*v = h - (w_e(nz) - w_e(nz+1))*v,
        ! the thickness the cell has LEFT after the explicit step with w_e. M is therefore
        ! diagonally dominant (Thomas sweep stable, pivots > 0) iff no cell was drained by
        ! the explicit step -- a PRECONDITION this kernel does not check (inherited from
        ! FESOM2): the cap bounds every FACE Courant number (CFL_z <= wsplit_maxcfl), and a
        ! cell with explicit outflow through both faces between much thicker neighbours
        ! can still lose up to ~2*maxcfl*h_min/h of its thickness. Under the precondition
        ! the COMBINED explicit+implicit step is the convex combination
        !     T^{n+1} = [(h - v*out_e)*T + v*in_e*T_donor^n + v*in_i*T_donor^{n+1}]
        !             / [(h - v*out_e) + v*in_e + v*in_i]
        ! (out/in = the upwind outflow/inflow face velocities of the cell, split into
        ! their explicit and implicit parts), i.e. bounded by T, the explicit donors' old
        ! and the implicit donors' new values. The implicit solve ALONE weights T* by
        ! h'/(h' + v*out_i) and is NOT a convex combination of T*; boundedness belongs to
        ! the combined step (test_wimpl_tra C4 asserts the precondition and the bound).
        ! FESOM2 also rebuilds zbar_n/Z_n (:142-150) but never uses them -- dropped.
        !
        ! PRECONDITION (not checked here, like the 1-layer trap the module header notes
        ! for adv_tra_ver_upw1): every owned column has at least 2 layers -- the surface
        ! row nzmin and the bottom row nzmax-1 would be the same row with one. read_mesh
        ! asserts it once at setup (assert_min_layers, test_minlayers trips it).
        ! M2.12c: OWNED node loop (FESOM2 :120 do n=1,myDim_nod2D); the TDMA is per column
        ! (no halo coupling); the driver's exchange_nod(fct_LO) follows.
        real(kind=WP), intent(in)    :: dt
        type(t_mesh),  intent(in)    :: mesh
        real(kind=WP), intent(in)    :: w  (mesh%nl,   mesh%nod2D)
        real(kind=WP), intent(inout) :: ttf(mesh%nl-1, mesh%nod2D)
        type(t_partit), intent(in), optional :: partit
        real(kind=WP) :: a(mesh%nl), b(mesh%nl), c(mesh%nl), tr(mesh%nl), cp(mesh%nl), tp(mesh%nl)
        real(kind=WP) :: v, m, hn
        integer :: n, nz, nzmax, nzmin
        integer :: nNodO, nNodL, nEdgeO, nElemO

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        do n = 1, nNodO
            nzmax = mesh%nlevels_nod2D(n)
            nzmin = mesh%ulevels_nod2D(n)
            v = dt*mesh%area(n)/mesh%areasvol(n)

            ! coefficients: surface layer
            nz = nzmin
            a(nz) = 0.0_WP
            b(nz) = mesh%hnode_new(nz,n) + w(nz,n)*v - min(0.0_WP, w(nz+1,n))*v
            c(nz) =                                  - max(0.0_WP, w(nz+1,n))*v
            ! interior layers
            do nz = nzmin+1, nzmax-2
                a(nz) =                        min(0.0_WP, w(nz,n))*v
                b(nz) = mesh%hnode_new(nz,n) + max(0.0_WP, w(nz,n))*v - min(0.0_WP, w(nz+1,n))*v
                c(nz) =                                              - max(0.0_WP, w(nz+1,n))*v
            end do
            ! bottom layer: zero bottom flux
            nz = nzmax-1
            a(nz) =                        min(0.0_WP, w(nz,n))*v
            b(nz) = mesh%hnode_new(nz,n) + max(0.0_WP, w(nz,n))*v
            c(nz) = 0.0_WP

            ! rhs = -(M - h')*T, the explicit upwind flux divergence of the current T
            nz = nzmin
            hn = mesh%hnode_new(nz,n)
            tr(nz) = -(b(nz)-hn)*ttf(nz,n) - c(nz)*ttf(nz+1,n)
            do nz = nzmin+1, nzmax-2
                hn = mesh%hnode_new(nz,n)
                tr(nz) = -a(nz)*ttf(nz-1,n) - (b(nz)-hn)*ttf(nz,n) - c(nz)*ttf(nz+1,n)
            end do
            nz = nzmax-1
            hn = mesh%hnode_new(nz,n)
            tr(nz) = -a(nz)*ttf(nz-1,n) - (b(nz)-hn)*ttf(nz,n)

            ! Thomas sweep (c-prime / t-prime), then back substitution
            nz = nzmin
            cp(nz) = c(nz)/b(nz)
            tp(nz) = tr(nz)/b(nz)
            do nz = nzmin+1, nzmax-1
                m = b(nz) - cp(nz-1)*a(nz)
                cp(nz) = c(nz)/m
                tp(nz) = (tr(nz) - tp(nz-1)*a(nz))/m
            end do
            tr(nzmax-1) = tp(nzmax-1)
            do nz = nzmax-2, nzmin, -1
                tr(nz) = tp(nz) - cp(nz)*tr(nz+1)
            end do

            ! update the tracer
            do nz = nzmin, nzmax-1
                ttf(nz,n) = ttf(nz,n) + tr(nz)
            end do
        end do
    end subroutine adv_tra_vert_impl

end module oce_adv_tra_ver
