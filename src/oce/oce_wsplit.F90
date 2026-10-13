module oce_wsplit
    ! Smooth, Courant-number-dependent split of the vertical velocity into an explicitly
    ! and an implicitly advected part (use_wsplit): w = w_e + w_i, w_i = f*w, w_e = w - w_i.
    !
    ! WHY. At high vertical Courant numbers (thin z* surface layers, cavities, strong
    ! convection, fine vertical grids) explicit vertical advection is unstable. FESOM2
    ! (oce_ale.F90:3217-3265, compute_Wvel_split) caps the explicit Courant number at
    ! wsplit_maxcfl and sends the excess to the implicit upstream solves (momentum TDMA,
    ! tracer vertical-diffusion TDMA, FCT adv_tra_vert_impl). Its split is a HARD switch:
    !     CFL_z <= C : w_e = w                   CFL_z > C : w_e = w*C/CFL_z
    ! so d(w_e)/d(CFL_z) jumps from 1 to 0 at the threshold and a face oscillating around
    ! it flips between high-order explicit and partly first-order implicit treatment.
    !
    ! THE FUNCTION (Shchepetkin 2015, Ocean Modelling 91, 38-69, Sec. 3.1 / Fig. 9; the same
    ! form is NEMO's wAimp in sshwzv.F90). With D = Cu_max - Cu_min, F = 4*Cu_max*D and
    ! Cu_cut = 2*Cu_max - Cu_min, the IMPLICIT share f of the face velocity is
    !     Cu <= Cu_min           f = 0                                   (fully explicit)
    !     Cu_min < Cu < Cu_cut   f = x^2/(F + x^2),   x = Cu - Cu_min   (smooth bend)
    !     Cu >= Cu_cut           f = (Cu - Cu_max)/Cu                   (Cu_e == Cu_max)
    ! Properties (all pinned by test/test_wsplit.F90 part W): C^1 everywhere -- at Cu_min
    ! f = f' = 0 on both sides; at Cu_cut both branches give f = D/Cu_cut and
    ! f' = Cu_max/Cu_cut^2. The explicit Courant number Cu_e = Cu*(1-f) is non-decreasing
    ! and saturates EXACTLY at Cu_max for Cu >= Cu_cut, so wsplit_maxcfl keeps its FESOM2
    ! meaning (the explicit CFL_z never exceeds it); f -> 1 as Cu -> inf. The argument is
    ! FESOM2's CFL_z (compute_CFLz: the face flux counted against BOTH adjacent cells),
    ! unchanged. The degenerate Cu_min = Cu_max (F = 0, Cu_cut = Cu_max) is FESOM2's hard
    ! switch in exact arithmetic; it is accepted by wsplit_check_params (used as the
    ! positive control of the smoothness test) but it is not a mode of its own.
    !
    ! Floating point: above Cu_cut, 1-f = Cu_max/Cu is formed by cancellation, so
    ! Cu*(1-f) carries an absolute error ~eps*Cu (relative ~eps*Cu/Cu_max); the consumers
    ! need no more than w_e + w_i == w to 1 ulp, which w_e = w - w_i guarantees.
    use mod_precision, only: WP
    implicit none
    private
    public :: wsplit_implicit_fraction, wsplit_check_params, wsplit_assert_params

contains

    elemental function wsplit_implicit_fraction(cfl, cmin, cmax) result(f)
        ! Implicit share f in [0,1] of the face velocity for the vertical Courant number
        ! cfl (FESOM2 CFL_z), parameters cmin = wsplit_mincfl, cmax = wsplit_maxcfl
        ! (validated by wsplit_check_params: cmax > 0, 0 <= cmin <= cmax).
        real(kind=WP), intent(in) :: cfl, cmin, cmax
        real(kind=WP) :: f
        real(kind=WP) :: x, ff, ccut

        ccut = 2.0_WP*cmax - cmin
        if (cfl <= cmin) then
            f = 0.0_WP
        else if (cfl < ccut) then
            x  = cfl - cmin
            ff = 4.0_WP*cmax*(cmax - cmin)
            f  = x*x/(ff + x*x)
        else
            f = (cfl - cmax)/cfl
        end if
    end function wsplit_implicit_fraction

    pure function wsplit_check_params(cmin, cmax) result(ok)
        ! .true. iff (wsplit_mincfl, wsplit_maxcfl) is admissible: a positive cap and a
        ! non-negative onset not above it (wsplit_assert_params error-stops on .false.).
        real(kind=WP), intent(in) :: cmin, cmax
        logical :: ok
        ok = (cmax > 0.0_WP .and. cmin >= 0.0_WP .and. cmin <= cmax)
    end function wsplit_check_params

    subroutine wsplit_assert_params(cmin, cmax)
        ! The SETUP-TIME guard, the one call every driver that enables use_wsplit makes
        ! once (t_dyn documents the obligation next to the parameters): error stop on an
        ! inadmissible pair. The kernels (compute_Wvel_split, adv_tra_vert_impl) assume it
        ! and validate nothing -- for (0.5, 0.3), say, Cu_cut = 0.1 < Cu_min and f would
        ! jump from 0 to 0.4 at Cu = 0.5, the discontinuity the function exists to remove.
        real(kind=WP), intent(in) :: cmin, cmax
        if (.not. wsplit_check_params(cmin, cmax)) &
            error stop 'wsplit_assert_params: need maxcfl > 0 and 0 <= mincfl <= maxcfl'
    end subroutine wsplit_assert_params

end module oce_wsplit
