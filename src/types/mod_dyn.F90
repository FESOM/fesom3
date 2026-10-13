module mod_dyn
    ! Ocean dynamics type (decision D5: EVOLVING; prognostic state split from
    ! work/scratch). Structure from tracer_dwarf MOD_DYN (== FESOM2 v2.7.3 MOD_DYN),
    ! trimmed to v1 scope: split-explicit SSH (se_*), backscatter/UKE, energy
    ! diagnostics (ke_*) and iceberg arrays are dropped. fer_uv/fer_w are declared
    ! (GM, M4) but not yet serialized.
    !
    ! Aux 3D WORK fields (density/N^2/Kv/Av/hpressure/sw_alpha-beta/GM slopes, D5)
    ! are added to t_dyn_work at M2.1 with verified FESOM2 names.
    use mod_precision, only: WP
    use, intrinsic :: iso_fortran_env, only: int32
    use mod_binary_arrays, only: write_bin_array, read_bin_array
    implicit none
    save

    type t_solverinfo
        integer       :: ident   = 1
        integer       :: maxiter = 2000
        integer       :: restart = 15
        integer       :: fillin  = 3
        integer       :: lutype  = 2
        real(kind=WP) :: droptol = 1.e-8_WP
        real(kind=WP) :: soltol  = 1.e-5_WP
        real(kind=WP), allocatable :: rr(:), zz(:), pp(:), App(:)
    contains
        procedure :: write_si => write_t_solverinfo
        procedure :: read_si  => read_t_solverinfo
    end type t_solverinfo

    type t_dyn_work
        real(kind=WP), allocatable, dimension(:,:,:) :: uvnode_rhs
        real(kind=WP), allocatable, dimension(:,:)   :: u_c, v_c
        ! Aux 3D fields recomputed each step from T/S by the EOS/pressure pass
        ! (M2.1 oce_pressure_bv); NOT serialized — they are diagnostics of the
        ! prognostic tracer state, restored by recomputation, not from the restart.
        ! FESOM2 o_ARRAYS names. density_ref = density_0 unless use_density_ref.
        real(kind=WP), allocatable, dimension(:,:)   :: density_m_rho0  ! (nl-1,nod2D) in-situ density - density_ref (PGF)
        real(kind=WP), allocatable, dimension(:,:)   :: density_ref     ! (nl-1,nod2D) reference density
        real(kind=WP), allocatable, dimension(:,:)   :: hpressure       ! (nl,  nod2D) hydrostatic pressure
        real(kind=WP), allocatable, dimension(:,:)   :: bvfreq          ! (nl,  nod2D) N^2 (squared Brunt-Vaisala)
        ! N^2 as pressure_bv computed it, BEFORE the horizontal smooth_nod pass. Consumers
        ! that must see the true local stratification (mo_convect's static-instability ramp)
        ! read this; GM/Redi + PP/KPP/TKE keep reading the smoothed bvfreq above.
        real(kind=WP), allocatable, dimension(:,:)   :: bvfreq_raw      ! (nl,  nod2D) N^2 pre-smoothing
        ! Squared vertical velocity shear |d(uvnode)/dz|^2 at LEVELS, produced once per
        ! step by oce_vert_spline::compute_shear2 and shared by PP, KPP's ri_iwmix and
        ! TKE (which each formed it separately before, with slightly different rounding).
        real(kind=WP), allocatable, dimension(:,:)   :: shear2          ! (nl,  nod2D) |du/dz|^2
        ! Relative vorticity curl(u) at nodes, for the vector-invariant momentum
        ! advection (momadv_opt==1, oce_dyn_vinv). Recomputed from UV at the start of
        ! every momentum_adv_vinv call, so it is NOT serialized -- adding it to
        ! write/read_t_dyn_work would change the restart layout for no benefit.
        real(kind=WP), allocatable, dimension(:,:)   :: vorticity       ! (nl-1,nod2D) curl(u)
        real(kind=WP), allocatable, dimension(:,:)   :: pgf_x, pgf_y    ! (nl-1,elem2D) PGF (M2.2 oce_pgf), from hpressure
        ! M2.8 PP vertical mixing coefficients (oce_ale_mixing_pp). FESOM2 o_ARRAYS
        ! names Kv/Av; recomputed each step from N^2 + uvnode shear, NOT serialized.
        real(kind=WP), allocatable, dimension(:,:)   :: Kv              ! (nl,  nod2D) vertical diffusivity (tracers)
        real(kind=WP), allocatable, dimension(:,:)   :: Av              ! (nl,  elem2D) vertical viscosity (momentum)
        ! M4 GM/Redi aux fields (recomputed each step from T/S; NOT serialized). FESOM2
        ! o_ARRAYS names. Allocated only when Fer_GM/Redi is on (the GM-off step never
        ! touches them). sw_alpha/sw_beta -> sigma_xy; init_Redi_GM -> fer_K/fer_c/fer_scal;
        ! fer_solve_Gamma -> fer_gamma. fer_uv/fer_w (the bolus velocities) ride t_dyn.
        real(kind=WP), allocatable, dimension(:,:)   :: sw_alpha, sw_beta ! (nl-1, nod2D) EOS expansion coeffs
        real(kind=WP), allocatable, dimension(:,:,:) :: sigma_xy         ! (2, nl-1, nod2D) density gradient
        real(kind=WP), allocatable, dimension(:,:)   :: fer_K            ! (nl,  nod2D) GM diffusivity
        real(kind=WP), allocatable, dimension(:)     :: fer_c, fer_scal  ! (nod2D) gravity-wave c^2 / scaling
        real(kind=WP), allocatable, dimension(:,:,:) :: fer_gamma        ! (2, nl, nod2D) GM streamfunction
        ! M4d Redi: compute_neutral_slope outputs + the Redi diffusivity (init_Redi_GM).
        real(kind=WP), allocatable, dimension(:,:,:) :: neutral_slope, slope_tapered ! (3, nl-1, nod2D)
        real(kind=WP), allocatable, dimension(:,:)   :: fer_tapfac       ! (nl-1, nod2D) Redi/GM slope taper
        real(kind=WP), allocatable, dimension(:,:)   :: Ki               ! (nl-1, nod2D) Redi isopycnal diffusivity
        ! M5 KPP vertical mixing (oce_mixing_KPP). Recomputed each step; NOT serialized.
        ! Allocated only when KPP (mix_scheme_nmb==1); the PP step never touches them. Names
        ! match FESOM2 o_mixing_KPP_mod module arrays. Kv_double(:,:,1)=T channel (-> Kv),
        ! (:,:,2)=S channel; viscA_kpp = the node momentum viscosity (FESOM2's local viscA)
        ! averaged node->elem into Av. dbsfc is filled by pressure_bv; sw_3d by cal_shortwave_rad
        ! (M5c; zero + unused when use_sw_pene=.false.).
        real(kind=WP), allocatable, dimension(:,:,:) :: Kv_double        ! (nl, nod2D, num_tracers) T/S diffusivity
        real(kind=WP), allocatable, dimension(:,:)   :: viscA_kpp        ! (nl, nod2D) node momentum viscosity (-> Av)
        real(kind=WP), allocatable, dimension(:,:,:) :: blmc             ! (nl, nod2D, 3) BL mixing coeffs (mom/T/S)
        real(kind=WP), allocatable, dimension(:,:)   :: ghats            ! (nl-1, nod2D) nonlocal counter-gradient flux
        real(kind=WP), allocatable, dimension(:,:)   :: dkm1             ! (nod2D, 3) kbl-1 diffusivities (mom/T/S)
        real(kind=WP), allocatable, dimension(:,:)   :: dbsfc            ! (nl, nod2D) buoyancy diff wrt surface (pressure_bv)
        real(kind=WP), allocatable, dimension(:,:)   :: dVsq             ! (nl, nod2D) surface-referenced velocity shear
        real(kind=WP), allocatable, dimension(:,:)   :: sw_3d            ! (nl, nod2D) penetrating shortwave (M5c)
        real(kind=WP), allocatable, dimension(:)     :: hbl, bfsfc       ! (nod2D) OBL depth / surface buoyancy forcing
        real(kind=WP), allocatable, dimension(:)     :: stable, caseA    ! (nod2D) stable flag / caseA flag
        real(kind=WP), allocatable, dimension(:)     :: ustar, Bo        ! (nod2D) friction velocity / surface buoyancy flux
        integer,       allocatable, dimension(:)     :: kbl              ! (nod2D) first level below the OBL
        ! M7 TKE vertical mixing (oce_mixing_tke). Allocated only when TKE (mix_scheme_nmb==5);
        ! the KPP/PP step never touches them. tke is the FIRST stateful mixing field — a prognostic
        ! tke(nl,node) carried step->step (tke_old -> integrate_tke -> tke_new in the same slab);
        ! tke restart serialization is the M8 item (NOT in write_t_dyn_work). tke_Av/tke_Kv are
        ! FRESH overwrites each step (node KappaM/KappaH); tke_Kv -> Kv (nodes), tke_Av -> Av
        ! (node->elem 3-vertex mean). FESOM2 g_cvmix_tke module-array names.
        real(kind=WP), allocatable, dimension(:,:)   :: tke              ! (nl, nod2D) prognostic TKE [m2/s2]
        real(kind=WP), allocatable, dimension(:,:)   :: tke_Av           ! (nl, nod2D) node KappaM (-> Av)
        real(kind=WP), allocatable, dimension(:,:)   :: tke_Kv           ! (nl, nod2D) node KappaH (-> Kv)
    contains
        procedure :: write_dw => write_t_dyn_work
        procedure :: read_dw  => read_t_dyn_work
    end type t_dyn_work

    type t_dyn
        ! ---- prognostic / state ----
        real(kind=WP), allocatable, dimension(:,:,:)   :: uv, uv_rhs, fer_uv  ! (2,nl-1,elem2D)
        real(kind=WP), allocatable, dimension(:,:,:,:) :: uv_rhsAB            ! (AB_order-1,2,nl-1,elem2D) — FESOM2 order (ab_lvl,comp,nz,elem)
        real(kind=WP), allocatable, dimension(:,:,:)   :: uvnode              ! (2,nl-1,nod2D)
        real(kind=WP), allocatable, dimension(:,:)     :: w, w_e, w_i, w_old, cfl_z, fer_w ! (nl,nod2D)
        real(kind=WP), allocatable, dimension(:)       :: eta_n, d_eta, ssh_rhs, ssh_rhs_old ! (nod2D)
        integer :: AB_order = 2

        ! ---- run config (rides in the type, D6) ----
        logical       :: check_opt_visc = .true.
        integer       :: opt_visc       = 5
        real(kind=WP) :: visc_gamma0    = 0.03_WP
        real(kind=WP) :: visc_gamma1    = 0.1_WP
        real(kind=WP) :: visc_gamma2    = 0.285_WP
        ! harmonic (Laplacian) viscosity coefficients for opt_visc=7 (FESOM2 MOD_DYN:
        ! visc_gamma{0,1}_h). Default 0 -> pure biharmonic (the pi/reduced-M2 config).
        real(kind=WP) :: visc_gamma0_h  = 0.0_WP
        real(kind=WP) :: visc_gamma1_h  = 0.0_WP
        logical       :: use_ivertvisc  = .true.
        integer       :: momadv_opt     = 2
        logical       :: use_freeslip   = .false.
        ! Smooth Courant-dependent w = w_e + w_i split (oce_wsplit documents the function;
        ! compute_Wvel_split applies it). Admissible iff wsplit_check_params (maxcfl > 0,
        ! 0 <= mincfl <= maxcfl). OBLIGATION: every driver that sets use_wsplit=.true.
        ! calls wsplit_assert_params(wsplit_mincfl, wsplit_maxcfl) once at setup (the
        ! kernels validate nothing). The default mincfl is the ABSOLUTE 0.5 (= 0.5*maxcfl
        ! only at the default maxcfl): a driver that lowers maxcfl must set mincfl too
        ! (the env-hook drivers default it to 0.5*maxcfl).
        ! Plan: docs/plans/completed/2026-10-02-wsplit-smooth.md.
        logical       :: use_wsplit     = .false.
        real(kind=WP) :: wsplit_mincfl  = 0.5_WP
        real(kind=WP) :: wsplit_maxcfl  = 1.0_WP

        ! ---- sub-objects ----
        type(t_solverinfo) :: solverinfo
        type(t_dyn_work)   :: work
    contains
        procedure :: write_unformatted => write_t_dyn
        procedure :: read_unformatted  => read_t_dyn
        generic   :: write(unformatted) => write_unformatted
        generic   :: read(unformatted)  => read_unformatted
    end type t_dyn

contains

    subroutine write_t_solverinfo(si, unit)
        class(t_solverinfo), intent(in) :: si
        integer,             intent(in) :: unit
        integer :: iostat
        character(len=1024) :: iomsg
        write(unit, iostat=iostat, iomsg=iomsg) si%ident, si%maxiter, si%restart, &
            si%fillin, si%lutype
        write(unit, iostat=iostat, iomsg=iomsg) si%droptol, si%soltol
        call write_bin_array(si%rr,  unit, iostat, iomsg)
        call write_bin_array(si%zz,  unit, iostat, iomsg)
        call write_bin_array(si%pp,  unit, iostat, iomsg)
        call write_bin_array(si%App, unit, iostat, iomsg)
    end subroutine write_t_solverinfo

    subroutine read_t_solverinfo(si, unit)
        class(t_solverinfo), intent(inout) :: si
        integer,             intent(in)    :: unit
        integer :: iostat
        character(len=1024) :: iomsg
        read(unit, iostat=iostat, iomsg=iomsg) si%ident, si%maxiter, si%restart, &
            si%fillin, si%lutype
        read(unit, iostat=iostat, iomsg=iomsg) si%droptol, si%soltol
        call read_bin_array(si%rr,  unit, iostat, iomsg)
        call read_bin_array(si%zz,  unit, iostat, iomsg)
        call read_bin_array(si%pp,  unit, iostat, iomsg)
        call read_bin_array(si%App, unit, iostat, iomsg)
    end subroutine read_t_solverinfo

    subroutine write_t_dyn_work(dw, unit)
        class(t_dyn_work), intent(in) :: dw
        integer,           intent(in) :: unit
        integer :: iostat
        character(len=1024) :: iomsg
        call write_bin_array(dw%uvnode_rhs, unit, iostat, iomsg)
        call write_bin_array(dw%u_c,        unit, iostat, iomsg)
        call write_bin_array(dw%v_c,        unit, iostat, iomsg)
    end subroutine write_t_dyn_work

    subroutine read_t_dyn_work(dw, unit)
        class(t_dyn_work), intent(inout) :: dw
        integer,           intent(in)    :: unit
        integer :: iostat
        character(len=1024) :: iomsg
        call read_bin_array(dw%uvnode_rhs, unit, iostat, iomsg)
        call read_bin_array(dw%u_c,        unit, iostat, iomsg)
        call read_bin_array(dw%v_c,        unit, iostat, iomsg)
    end subroutine read_t_dyn_work

    subroutine write_t_dyn(dyn, unit, iostat, iomsg)
        class(t_dyn), intent(in)    :: dyn
        integer,      intent(in)    :: unit
        integer,      intent(out)   :: iostat
        character(*), intent(inout) :: iomsg
        write(unit, iostat=iostat, iomsg=iomsg) dyn%AB_order, dyn%opt_visc, &
            dyn%momadv_opt
        write(unit, iostat=iostat, iomsg=iomsg) dyn%check_opt_visc, &
            dyn%use_ivertvisc, dyn%use_freeslip, dyn%use_wsplit
        write(unit, iostat=iostat, iomsg=iomsg) dyn%visc_gamma0, dyn%visc_gamma1, &
            dyn%visc_gamma2, dyn%visc_gamma0_h, dyn%visc_gamma1_h, dyn%wsplit_maxcfl, &
            dyn%wsplit_mincfl
        call dyn%solverinfo%write_si(unit)
        call dyn%work%write_dw(unit)
        call write_bin_array(dyn%uv,          unit, iostat, iomsg)
        call write_bin_array(dyn%uv_rhs,      unit, iostat, iomsg)
        call write_bin_array(dyn%uv_rhsAB,    unit, iostat, iomsg)
        call write_bin_array(dyn%uvnode,      unit, iostat, iomsg)
        call write_bin_array(dyn%w,           unit, iostat, iomsg)
        call write_bin_array(dyn%w_e,         unit, iostat, iomsg)
        call write_bin_array(dyn%w_i,         unit, iostat, iomsg)
        call write_bin_array(dyn%w_old,       unit, iostat, iomsg)
        call write_bin_array(dyn%cfl_z,       unit, iostat, iomsg)
        call write_bin_array(dyn%eta_n,       unit, iostat, iomsg)
        call write_bin_array(dyn%d_eta,       unit, iostat, iomsg)
        call write_bin_array(dyn%ssh_rhs,     unit, iostat, iomsg)
        call write_bin_array(dyn%ssh_rhs_old, unit, iostat, iomsg)
    end subroutine write_t_dyn

    subroutine read_t_dyn(dyn, unit, iostat, iomsg)
        class(t_dyn), intent(inout) :: dyn
        integer,      intent(in)    :: unit
        integer,      intent(out)   :: iostat
        character(*), intent(inout) :: iomsg
        read(unit, iostat=iostat, iomsg=iomsg) dyn%AB_order, dyn%opt_visc, &
            dyn%momadv_opt
        read(unit, iostat=iostat, iomsg=iomsg) dyn%check_opt_visc, &
            dyn%use_ivertvisc, dyn%use_freeslip, dyn%use_wsplit
        read(unit, iostat=iostat, iomsg=iomsg) dyn%visc_gamma0, dyn%visc_gamma1, &
            dyn%visc_gamma2, dyn%visc_gamma0_h, dyn%visc_gamma1_h, dyn%wsplit_maxcfl, &
            dyn%wsplit_mincfl
        call dyn%solverinfo%read_si(unit)
        call dyn%work%read_dw(unit)
        call read_bin_array(dyn%uv,          unit, iostat, iomsg)
        call read_bin_array(dyn%uv_rhs,      unit, iostat, iomsg)
        call read_bin_array(dyn%uv_rhsAB,    unit, iostat, iomsg)
        call read_bin_array(dyn%uvnode,      unit, iostat, iomsg)
        call read_bin_array(dyn%w,           unit, iostat, iomsg)
        call read_bin_array(dyn%w_e,         unit, iostat, iomsg)
        call read_bin_array(dyn%w_i,         unit, iostat, iomsg)
        call read_bin_array(dyn%w_old,       unit, iostat, iomsg)
        call read_bin_array(dyn%cfl_z,       unit, iostat, iomsg)
        call read_bin_array(dyn%eta_n,       unit, iostat, iomsg)
        call read_bin_array(dyn%d_eta,       unit, iostat, iomsg)
        call read_bin_array(dyn%ssh_rhs,     unit, iostat, iomsg)
        call read_bin_array(dyn%ssh_rhs_old, unit, iostat, iomsg)
    end subroutine read_t_dyn

end module mod_dyn
