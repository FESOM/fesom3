module mod_step_oce_rk3
    ! The RK3 ocean step (FESOM3_RK3=1), the alternative to step_oce's AB2 sequence. Plan:
    ! docs/plans/2026-10-28-rk3.md, Technical Details §2; the scheme and the knobs are documented
    ! in src/oce/oce_rk3.F90. step_oce hands over to step_oce_rk3 right after its first-call
    ! guards, with its own argument list.
    use mod_precision,      only: WP
    use mod_mesh,           only: t_mesh
    use mod_dyn,            only: t_dyn
    use mod_tracer,         only: t_tracer
    use mod_partit,         only: t_partit
    use oce_rk3,            only: rk3_die
    implicit none
    private
    public :: step_oce_rk3

contains

    subroutine step_oce_rk3(n, dt, dynamics, tracers, mesh, &
                            Ki, heat_flux, water_flux, virtual_salt, relax_salt, &
                            real_salt_flux, is_nonlinfs, stress_surf, partit, stress_node_surf)
        integer,        intent(in)            :: n
        real(kind=WP),  intent(in)            :: dt
        type(t_dyn),    intent(inout), target :: dynamics
        type(t_tracer), intent(inout), target :: tracers
        type(t_mesh),   intent(inout), target :: mesh
        real(kind=WP),  intent(in) :: Ki(mesh%nl-1, mesh%nod2D)
        real(kind=WP),  intent(in) :: heat_flux(mesh%nod2D), water_flux(mesh%nod2D)
        real(kind=WP),  intent(in) :: virtual_salt(mesh%nod2D), relax_salt(mesh%nod2D)
        real(kind=WP),  intent(in) :: real_salt_flux(mesh%nod2D), is_nonlinfs
        real(kind=WP),  intent(in) :: stress_surf(2, mesh%elem2D)
        type(t_partit), intent(in), optional :: partit
        real(kind=WP),  intent(in), optional :: stress_node_surf(2, mesh%nod2D)

        call rk3_die('step_oce_rk3: not yet implemented', partit)
    end subroutine step_oce_rk3

end module mod_step_oce_rk3
