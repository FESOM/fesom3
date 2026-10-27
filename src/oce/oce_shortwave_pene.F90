module oce_shortwave_pene
    ! Shortwave penetration into the ocean (M2.10c). Transcribed VERBATIM from FESOM2
    ! v2.7.3 oce_shortwave_pene.F90 (cal_shortwave_rad, 1-111) — Morel & Antoine 1994 /
    ! Sweeney et al. 2005 chlorophyll-dependent attenuation.
    !
    ! swsurf = (1-albw)*shortwave;  visible band = swsurf*0.54;  heat_flux += swsurf
    ! (the visible part is REMOVED from the surface non-solar heat_flux — heat_flux is
    ! +upward — and re-deposited over depth as sw_3d). The vertical profile is a
    ! two-exponential (v1/exp(z/sc1) + v2/exp(z/sc2)) with v1/v2/sc1/sc2 polynomials in
    ! c=log10(chl) (Sweeney 2005 App. A). swsurf converted W/m^2 -> K m/s by /vcpw.
    !
    ! BIT-IDENTITY: every literal verbatim (0.54_WP, the v1/v2/sc1/sc2 polynomial
    ! coefficients, 0.02_WP chl floor, 1.e-5_WP cutoff); zbar_3d_n is the per-node ALE
    ! depth (pressure-gate-proven). albw is FESOM2 ice%thermo%albw (0.066). No
    ! penetration under cavity (ulevels>1) or ice (use_ice .and. a_ice>0).
    !
    ! SCOPE (M2.10c): produces sw_3d (the in-water absorption profile the tracer solve
    ! consumes) + the heat_flux visible-removal. The surface heat_flux ITSELF (the
    ! air-sea obudget) is M3; here heat_flux is a prescribed input that the routine
    ! only ADDS the visible band back to.
    use mod_precision, only: WP, WP_full
    use mod_constants, only: vcpw
    use mod_mesh, only: t_mesh
    use mod_partit, only: t_partit
    use mod_part_bounds, only: owned_bounds
    implicit none
    private
    public :: cal_shortwave_rad

contains

    subroutine cal_shortwave_rad(use_ice, albw, shortwave, chl, a_ice, heat_flux, sw_3d, mesh, partit)
        logical,       intent(in)    :: use_ice
        real(kind=WP), intent(in)    :: albw
        real(kind=WP), intent(in)    :: shortwave(:), a_ice(:)
        real(kind=WP), intent(inout) :: chl(:)             ! floored at 0.02 in place
        real(kind=WP), intent(inout) :: heat_flux(:)       ! += visible swsurf
        real(kind=WP), intent(out)   :: sw_3d(:,:)         ! (nl, nNodL)
        type(t_mesh),  intent(in)    :: mesh
        ! M5d multi-rank: optional partit -> loop owned+halo nodes (nNodL = FESOM2
        ! myDim_nod2D+eDim_nod2D). ⚠️ in the LOCAL mesh mesh%nod2D = the GLOBAL count
        ! (read_mesh_local), so it CANNOT be the loop bound when partit is present; the
        ! arrays (shortwave/sw_3d/...) are nNodL-sized. Absent ⇒ 1-rank, nNodL = mesh%nod2D.
        type(t_partit), intent(in), optional :: partit
        integer :: n2, k, nzmax, nzmin, nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP) :: swsurf, aux, c, c2, c3, c4, c5, v1, v2, sc1, sc2

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)

        do n2 = 1, nNodL
            do k = 1, mesh%nl
                sw_3d(k, n2) = 0.0_WP
            end do
        end do

        do n2 = 1, nNodL
            if (mesh%ulevels_nod2D(n2) > 1) cycle                 ! no penetration in cavity
            if (use_ice .and. a_ice(n2) > 0._WP) cycle            ! no penetration under ice
            swsurf = (1.0_WP - albw)*shortwave(n2)
            swsurf = swsurf*0.54_WP                               ! visible (300-750nm)
            heat_flux(n2) = heat_flux(n2) + swsurf                ! remove visible from heat_flux
            if (chl(n2) < 0.02_WP) chl(n2) = 0.02_WP              ! floor chl
            c  = log10(chl(n2))
            c2 = c*c
            c3 = c2*c
            c4 = c3*c
            c5 = c4*c
            v1  = 0.008_WP*c + 0.132_WP*c2 + 0.038_WP*c3 - 0.017_WP*c4 - 0.007_WP*c5
            v2  = 0.679_WP - v1
            v1  = 0.321_WP + v1
            sc1 = 1.54_WP  - 0.197_WP*c + 0.166_WP*c2 - 0.252_WP*c3 - 0.055_WP*c4 + 0.042_WP*c5
            sc2 = 7.925_WP - 6.644_WP*c + 3.662_WP*c2 - 1.815_WP*c3 - 0.218_WP*c4 + 0.502_WP*c5
            swsurf = swsurf/vcpw                                  ! W/m^2 -> K m/s
            nzmax = mesh%nlevels_nod2D(n2)
            nzmin = mesh%ulevels_nod2D(n2)
            sw_3d(nzmin, n2) = swsurf
            do k = nzmin+1, nzmax
#if defined(USE_SINGLE_PRECISION)
                ! single precision: the attenuation exp(z/sc) leaves the real32 normal range
                ! (exp(-88) ~ 5e-39) above the aux < 1e-5 cut-off; Intel's expf then traps
                ! (-fpe0, FTZ on) on pi at step 1. Evaluated in WP_full and rounded once; in
                ! double precision this is the expression below.
                aux = real(v1*exp(real(mesh%zbar_3d_n(k,n2)/sc1, WP_full)) &
                         + v2*exp(real(mesh%zbar_3d_n(k,n2)/sc2, WP_full)), WP)
#else
                aux = (v1*exp(mesh%zbar_3d_n(k,n2)/sc1) + v2*exp(mesh%zbar_3d_n(k,n2)/sc2))
#endif
                sw_3d(k, n2) = swsurf*aux
                if (aux < 1.e-5_WP .or. k == nzmax) then
                    sw_3d(k, n2) = 0.0_WP
                    exit
                end if
            end do
        end do
    end subroutine cal_shortwave_rad

end module oce_shortwave_pene
