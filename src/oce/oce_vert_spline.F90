module oce_vert_spline
    ! Vertical cubic-spline reconstruction of d/dz, and the single shared producer of the
    ! squared vertical velocity shear that PP, KPP's ri_iwmix and TKE all consume.
    !
    ! WHY. FESOM's mixing schemes each formed the shear as a two-point difference
    !
    !     shear2(nz) = [ (u(nz-1)-u(nz))^2 + (v(nz-1)-v(nz))^2 ] / (Z(nz-1)-Z(nz))^2
    !
    ! and used it AT THE LEVEL (interface) nz. But a two-point difference of cell-centred
    ! values is a second-order estimate of du/dz at the MIDPOINT of the interval between
    ! the two centres, and on a stretched grid the interface is not at that midpoint. The
    ! leading error term then scales with the thickness contrast, so the estimate drops to
    ! first order exactly where the grid stretches -- on core2 that is everything below
    ! ~100 m, where dz runs 10 m -> 250 m. The spline path removes that: it reconstructs a
    ! C^2 cubic through the layer-centre values and evaluates its derivative AT the true
    ! interface depth zbar, so no midpoint/interface mismatch remains.
    !
    ! This is FESOM's analogue of ROMS RI_SPLINES (ROMS/Nonlinear/lmd_vmix.F), which
    ! likewise replaces the finite-difference shear with a spline reconstruction and, when
    ! active, disables the Ri averaging options (ROMS/Include/globaldefs.h undefines
    ! RI_HORAVG / RI_VERAVG under RI_SPLINES). ROMS uses a cell-average parabolic spline;
    ! this uses a point-value cubic spline evaluated off-node, which is the natural fit for
    ! FESOM where Z_3d_n are layer-centre POINTS and zbar_3d_n are the interfaces.
    !
    ! N2_splines (mod_param_phys) applies the SAME operator to N^2 in oce_pressure_bv, so
    ! that every Richardson-type ratio (PP, KPP ri_iwmix, TKE Prandtl) and the GM/Redi
    ! neutral slope divide like by like. Splining only the shear would create the mirror
    ! image of the asymmetry the two flags exist to remove.
    !
    ! shear_splines (mod_param_phys) selects: .false. = the two-point form, reproducing the
    ! previous PP / ri_iwmix arithmetic BIT FOR BIT (dz_inv*dz_inv, same operand order);
    ! .true. = the spline. TKE previously divided by (Z(nz-1)-Z(nz))**2 in one operation
    ! instead of multiplying by dz_inv twice, so TKE changes in the last bits even at
    ! shear_splines=.false. -- same value, different rounding.
    !
    ! Purely vertical: no horizontal stencil, hence NO halo exchange (unlike smooth_nod,
    ! which costs an exchange_nod per sweep). Under ALE Z_3d_n moves every step, so the
    ! tridiagonal is rebuilt each call; it cannot be precomputed.
    use mod_precision,   only: WP
    use mod_mesh,        only: t_mesh
    use mod_dyn,         only: t_dyn
    use mod_partit,      only: t_partit
    use mod_part_bounds, only: owned_bounds
    use mod_param_phys,  only: shear_splines
    implicit none
    private
    public :: spline_dqdx, spline_eval, compute_shear2

contains

    pure subroutine spline_dqdx(n, x, q, xi, d)
        ! dq/dx only (the shear consumer). See spline_eval for the full contract.
        integer,       intent(in)  :: n
        real(kind=WP), intent(in)  :: x(n), q(n), xi(n-1)
        real(kind=WP), intent(out) :: d(n-1)
        real(kind=WP) :: qi(max(n-1,1))
        call spline_eval(n, x, q, xi, qi, d)
    end subroutine spline_dqdx

    pure subroutine spline_eval(n, x, q, xi, qi, d)
        ! q AND dq/dx at n-1 evaluation points, from n point values, via a natural cubic
        ! spline. The N^2 path needs both: the derivative of T and S at the interface, and
        ! their VALUE there to evaluate alpha/beta at the interface state.
        !
        !   x (1:n)    strictly INCREASING abscissae (pass DEPTH, so it increases downward)
        !   q (1:n)    values at x
        !   xi(1:n-1)  evaluation abscissae; xi(k) must lie in [x(k), x(k+1)]
        !   qi(1:n-1)  out, q at xi(k)      (cubic Hermite value: exact at the knots)
        !   d (1:n-1)  out, dq/dx at xi(k)
        !
        ! Step 1 solves the standard tridiagonal system for the knot derivatives g(1:n),
        !   h(k) g(k-1) + 2(h(k-1)+h(k)) g(k) + h(k-1) g(k+1)
        !       = 3 [ h(k) delta(k-1) + h(k-1) delta(k) ],    delta(k) = (q(k+1)-q(k))/h(k)
        ! closed with the natural end conditions 2g(1)+g(2) = 3 delta(1) and
        ! g(n-1)+2g(n) = 3 delta(n-1). Step 2 evaluates the cubic Hermite derivative on the
        ! interval, which is exact at the knots (t=0 -> g(k), t=1 -> g(k+1)) and reduces to
        ! delta(k) for a linear q. For n = 2 the system gives g(1) = g(2) = delta(1), so
        ! the routine degenerates to the plain two-point slope -- the correct limit.
        integer,       intent(in)  :: n
        real(kind=WP), intent(in)  :: x(n), q(n), xi(n-1)
        real(kind=WP), intent(out) :: qi(n-1), d(n-1)
        real(kind=WP) :: h(n-1), delta(n-1), g(n), c(n), r(n)
        real(kind=WP) :: b, den, t, t2, t3, dl, dr
        integer       :: k

        if (n < 2) then
            if (n >= 1) then
                qi = 0.0_WP; d = 0.0_WP
            end if
            return
        end if

        do k = 1, n-1
            h(k)     = x(k+1) - x(k)
            delta(k) = (q(k+1) - q(k))/h(k)
        end do

        ! --- Thomas forward sweep over the knot-derivative system -------------------
        b    = 2.0_WP
        c(1) = 1.0_WP/b
        r(1) = 3.0_WP*delta(1)/b
        do k = 2, n-1
            den  = 2.0_WP*(h(k-1) + h(k)) - h(k)*c(k-1)
            c(k) = h(k-1)/den
            r(k) = (3.0_WP*(h(k)*delta(k-1) + h(k-1)*delta(k)) - h(k)*r(k-1))/den
        end do
        den  = 2.0_WP - c(n-1)
        r(n) = (3.0_WP*delta(n-1) - r(n-1))/den

        ! --- back substitution ------------------------------------------------------
        g(n) = r(n)
        do k = n-1, 1, -1
            g(k) = r(k) - c(k)*g(k+1)
        end do

        ! --- monotonicity limiter (Fritsch & Carlson 1980 / Hyman 1983) -----------------
        ! A natural cubic spline overshoots at a sharp transition -- the mixed-layer base
        ! is the textbook case -- and an overshoot in T or S can flip the SIGN of the N^2 the
        ! interface sees, which the convective adjustment then acts on. Measured on pi
        ! without this: 0.9% -> 1.8% of levels sign-flipped over 20 steps, a feedback.
        ! Clamp each knot derivative into the monotone box: zero at a local extremum of the
        ! data, otherwise the sign of the local slope and |g| <= 3*min(|delta|) over the two
        ! adjacent intervals. With both knots of an interval in that box the Hermite cubic
        ! is monotone on it, so the derivative at the interface has the sign of the two-point
        ! difference or is zero. The clamp is INACTIVE wherever the unlimited spline is
        ! already monotone (g/delta ~ 1 for smooth data), so smooth profiles keep the full
        ! spline accuracy; for n = 2 it leaves g = delta untouched.
        do k = 1, n
            if (k == 1) then
                dl = delta(1);   dr = delta(1)
            else if (k == n) then
                dl = delta(n-1); dr = delta(n-1)
            else
                dl = delta(k-1); dr = delta(k)
            end if
            if (dl*dr <= 0.0_WP .or. g(k)*dr <= 0.0_WP) then
                g(k) = 0.0_WP
            else
                g(k) = sign(min(abs(g(k)), 3.0_WP*min(abs(dl), abs(dr))), dr)
            end if
        end do

        ! --- cubic Hermite value + derivative at xi ----------------------------------
        do k = 1, n-1
            t     = (xi(k) - x(k))/h(k)
            t2    = t*t
            t3    = t2*t
            qi(k) = q(k)  *(2.0_WP*t3 - 3.0_WP*t2 + 1.0_WP) &
                  + q(k+1)*(3.0_WP*t2 - 2.0_WP*t3) &
                  + h(k)*( g(k)*(t3 - 2.0_WP*t2 + t) + g(k+1)*(t3 - t2) )
            d(k)  = delta(k)*(6.0_WP*t - 6.0_WP*t2) &
                  + g(k)    *(1.0_WP - 4.0_WP*t + 3.0_WP*t2) &
                  + g(k+1)  *(3.0_WP*t2 - 2.0_WP*t)
        end do
    end subroutine spline_eval

    subroutine compute_shear2(dyn, mesh, partit)
        ! Fill dyn%work%shear2(nz,node) = |d(uvnode)/dz|^2 at LEVELS nzmin+1..nzmax-1,
        ! zero elsewhere (nzmin, nzmax and below-bottom stay 0 -- TKE reads the padded
        ! slice vshear2(nun:nln+1) and needs those zeros).
        !
        ! Runs 1..nNodL (owned + halo): PP consumes shear2 over nNodL, while ri_iwmix and
        ! TKE only need nNodO. uvnode is halo-valid from compute_vel_nodes.
        type(t_mesh),   intent(in)            :: mesh
        type(t_dyn),    intent(inout), target :: dyn
        type(t_partit), intent(in),    optional :: partit
        integer       :: node, nz, nzmin, nzmax, n, k
        integer       :: nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP) :: dz_inv
        real(kind=WP) :: xc(mesh%nl), xi(mesh%nl), uc(mesh%nl), vc(mesh%nl)
        real(kind=WP) :: du(mesh%nl), dv(mesh%nl)
        real(kind=WP), dimension(:,:),   pointer :: shear2
        real(kind=WP), dimension(:,:,:), pointer :: UVn

        shear2 => dyn%work%shear2
        UVn    => dyn%uvnode
        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)

        do node = 1, nNodL
            nzmin = mesh%ulevels_nod2D(node)
            nzmax = mesh%nlevels_nod2D(node)
            shear2(:, node) = 0.0_WP

            if (.not. shear_splines) then
                ! two-point form -- byte-identical to the previous PP / ri_iwmix inline code
                do nz = nzmin+1, nzmax-1
                    dz_inv = 1.0_WP/(mesh%Z_3d_n(nz-1,node) - mesh%Z_3d_n(nz,node))
                    shear2(nz,node) = ( (UVn(1,nz-1,node)-UVn(1,nz,node))**2 &
                                      + (UVn(2,nz-1,node)-UVn(2,nz,node))**2 )*dz_inv*dz_inv
                end do
                cycle
            end if

            ! spline path: layer centres nzmin..nzmax-1 are the knots, the interior
            ! interfaces nzmin+1..nzmax-1 are the evaluation points. DEPTH (-z) is used so
            ! the abscissae increase; the sign cancels in the square.
            n = nzmax - nzmin
            if (n < 2) cycle
            do k = 1, n
                xc(k) = -mesh%Z_3d_n(nzmin+k-1, node)
                uc(k) =  UVn(1, nzmin+k-1, node)
                vc(k) =  UVn(2, nzmin+k-1, node)
            end do
            do k = 1, n-1
                xi(k) = -mesh%zbar_3d_n(nzmin+k, node)
            end do
            call spline_dqdx(n, xc(1:n), uc(1:n), xi(1:n-1), du(1:n-1))
            call spline_dqdx(n, xc(1:n), vc(1:n), xi(1:n-1), dv(1:n-1))
            do k = 1, n-1
                shear2(nzmin+k, node) = du(k)*du(k) + dv(k)*dv(k)
            end do
        end do
    end subroutine compute_shear2

end module oce_vert_spline
