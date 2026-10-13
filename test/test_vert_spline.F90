program test_vert_spline
    ! Gate for oce_vert_spline::spline_dqdx -- the cubic-spline d/dz that replaces the
    ! two-point difference in the shear (FESOM's analogue of ROMS RI_SPLINES).
    !
    ! The whole point of the spline is that a two-point difference of cell-centred values
    ! is a second-order estimate of dq/dz at the MIDPOINT of the interval, while the mixing
    ! schemes use it at the INTERFACE. On a uniform grid those coincide and the spline buys
    ! nothing; on a stretched grid they do not. That is exactly what is checked here, on a
    ! grid stretched 5 m -> 250 m like core2.
    !
    ! For q = x^2 the two-point slope over [x_k, x_{k+1}] is x_k + x_{k+1} = 2*midpoint,
    ! i.e. EXACTLY the true derivative at the midpoint -- so its error at the interface is
    ! analytically 2*(midpoint - x_interface). That gives an exact expected value to test
    ! against rather than a hand-tuned tolerance.
    use mpi
    use mod_precision,    only: WP
    use oce_vert_spline,  only: spline_dqdx, spline_eval
    implicit none

    integer, parameter :: N = 48
    integer :: ierr, nfail, k
    real(kind=WP) :: xc(N), xi(N-1), q(N), d(N-1), dz(N-1), xmid(N-1)
    real(kind=WP) :: e_sp, e_2p, worst_sp, worst_2p, two_pt, expect
    real(kind=WP) :: x2(2), q2(2), xi2(1), d2(1)
    real(kind=WP) :: qi(N-1), qi2(1)

    nfail = 0
    call MPI_Init(ierr)

    ! ---- core2-like stretched column: dz 5 m -> 250 m, interfaces between centres ----
    call build_grid()
    call check_true('grid stretches by >20x', dz(N-1)/dz(1) > 20.0_WP)
    ! On this grid the interface sits (h(k+1)-h(k))/4 away from the midpoint of the
    ! centre-to-centre interval -- about 1% of the local spacing. Small, but it multiplies
    ! the local curvature, which is why it matters at the thermocline and not in the
    ! smoothly stratified interior.
    write(*,'(a,f6.2,a)') '  interface offset from midpoint: ', &
        100.0_WP*maxval(abs(xi - xmid)/dz), ' % of local dz'
    call check_true('interface is off-midpoint', maxval(abs(xi - xmid)/dz) > 0.005_WP)

    ! ---- A. linear q: the natural spline is EXACT (S''=0 matches the end condition) ----
    q = 3.0_WP*xc - 17.0_WP
    call spline_dqdx(N, xc, q, xi, d)
    worst_sp = maxval(abs(d - 3.0_WP))
    call check_true('linear: spline exact', worst_sp < 1.0e-10_WP)

    ! ---- B. quadratic q = x^2, dq/dx = 2x ---------------------------------------
    q = xc*xc
    call spline_dqdx(N, xc, q, xi, d)
    worst_sp = 0.0_WP; worst_2p = 0.0_WP
    do k = 6, N-6                              ! interior: away from the natural-BC ends
        two_pt   = (q(k+1) - q(k))/dz(k)       ! what the schemes used before
        e_sp     = abs(d(k)      - 2.0_WP*xi(k))
        e_2p     = abs(two_pt    - 2.0_WP*xi(k))
        expect   = abs(2.0_WP*(xmid(k) - xi(k)))
        call check_true('quadratic: two-point error is the midpoint offset', &
                        abs(e_2p - expect) < 1.0e-9_WP*max(1.0_WP, expect))
        worst_sp = max(worst_sp, e_sp)
        worst_2p = max(worst_2p, e_2p)
    end do
    write(*,'(a,es11.3,a,es11.3,a,f7.1,a)') '  quadratic interior: spline err ', worst_sp, &
        '   two-point err ', worst_2p, '   (', worst_2p/max(worst_sp,tiny(1.0_WP)), 'x)'
    call check_true('quadratic: spline beats two-point by >100x', worst_2p > 100.0_WP*worst_sp)

    ! ---- C. cubic q: interior still far better than the two-point form ----------
    q = xc*xc*xc
    call spline_dqdx(N, xc, q, xi, d)
    worst_sp = 0.0_WP; worst_2p = 0.0_WP
    do k = 6, N-6
        two_pt   = (q(k+1) - q(k))/dz(k)
        worst_sp = max(worst_sp, abs(d(k)   - 3.0_WP*xi(k)*xi(k)))
        worst_2p = max(worst_2p, abs(two_pt - 3.0_WP*xi(k)*xi(k)))
    end do
    write(*,'(a,es11.3,a,es11.3,a,f7.1,a)') '  cubic     interior: spline err ', worst_sp, &
        '   two-point err ', worst_2p, '   (', worst_2p/max(worst_sp,tiny(1.0_WP)), 'x)'
    call check_true('cubic: spline beats two-point by >10x', worst_2p > 10.0_WP*worst_sp)

    ! ---- D. two-knot column degenerates to the plain slope ----------------------
    ! compute_shear2 hits this wherever a column has exactly two wet layers.
    x2  = [ 2.5_WP, 12.5_WP ]
    q2  = [ 7.0_WP, -3.0_WP ]
    xi2 = [ 5.0_WP ]                            ! deliberately NOT the midpoint
    call spline_dqdx(2, x2, q2, xi2, d2)
    call check_true('n=2 reduces to the two-point slope', &
                    abs(d2(1) - (q2(2)-q2(1))/(x2(2)-x2(1))) < 1.0e-12_WP)

    ! ---- F. value evaluator (the N2 path needs T,S AT the interface) --------------
    q = 3.0_WP*xc - 17.0_WP
    call spline_eval(N, xc, q, xi, qi, d)
    call check_true('value: linear q interpolated exactly', &
                    maxval(abs(qi - (3.0_WP*xi - 17.0_WP))) < 1.0e-9_WP)
    call spline_eval(N, xc, q, xc(1:N-1), qi, d)          ! evaluate AT the knots
    call check_true('value: exact at the knots', maxval(abs(qi - q(1:N-1))) < 1.0e-12_WP)
    call spline_eval(2, x2, q2, xi2, qi2, d2)
    call check_true('value: n=2 is linear interpolation', &
                    abs(qi2(1) - (q2(1) + (q2(2)-q2(1))*(xi2(1)-x2(1))/(x2(2)-x2(1)))) < 1.0e-12_WP)

    ! ---- E. no NaN/Inf anywhere --------------------------------------------------
    q = sin(0.01_WP*xc)
    call spline_dqdx(N, xc, q, xi, d)
    call check_true('finite output', all(d == d) .and. all(abs(d) < huge(1.0_WP)))

    if (nfail == 0) then
        write(*,'(a)') 'test_vert_spline: OK'
    else
        write(*,'(a,i0,a)') 'test_vert_spline: ', nfail, ' FAILURE(S)'
    end if
    call MPI_Finalize(ierr)
    if (nfail /= 0) error stop 1

contains

    subroutine build_grid()
        ! Layer thicknesses stretched 5 m -> 250 m; xc = layer centres, xi = the interior
        ! interfaces between them (the depths the mixing schemes evaluate the shear at).
        real(kind=WP) :: h(N), edge(N+1), f
        integer :: i
        do i = 1, N
            f    = real(i-1, WP)/real(N-1, WP)
            h(i) = 5.0_WP + 245.0_WP*f*f
        end do
        edge(1) = 0.0_WP
        do i = 1, N
            edge(i+1) = edge(i) + h(i)
        end do
        do i = 1, N
            xc(i) = 0.5_WP*(edge(i) + edge(i+1))
        end do
        do i = 1, N-1
            dz(i)   = xc(i+1) - xc(i)
            xmid(i) = 0.5_WP*(xc(i) + xc(i+1))
            xi(i)   = edge(i+1)                 ! the shared interface
        end do
    end subroutine

    subroutine check_true(name, cond)
        character(len=*), intent(in) :: name
        logical,          intent(in) :: cond
        if (.not. cond) then
            nfail = nfail + 1
            write(*,'(a)') '  FAIL: '//name
        end if
    end subroutine

end program test_vert_spline
