module mod_param_phys
    ! Write-once physics parameters (decision D4). Declared with FESOM2 defaults,
    ! populated once by read_param_phys() from namelist.oce, read-only thereafter.
    !
    ! Defaults transcribed from FESOM2 v2.7.3 src/oce_modules.F90 (module o_PARAM,
    ! line cites below). The &oce_dyn namelist group is transcribed in full from a
    ! real run dir (port2/fesom2/work_linfs_pp/namelist.oce) — every group member
    ! must be declared or the namelist read fails. Time-stepping/AB2 constants are
    ! o_PARAM defaults not exposed in the namelist.
    !
    ! Note: `epsilon` shadows the Fortran intrinsic of the same name, exactly as in
    ! FESOM2 o_PARAM:92 — kept faithful so AB2 kernels transcribe verbatim.
    use mod_precision, only: WP
    implicit none
    public
    save

    ! --- equation of state / general ---
    integer       :: state_equation = 1            ! oce_modules.F90:21 (1=full EOS, 0=linear)

    ! --- &oce_dyn: basic dynamics ---
    real(kind=WP) :: C_d        = 0.0025_WP        ! oce_modules.F90:23 bottom drag
    real(kind=WP) :: A_ver      = 0.001_WP         ! oce_modules.F90:26 background vert. visc.
    real(kind=WP) :: scale_area = 2.0e8_WP         ! oce_modules.F90:29 ref element area

    ! --- &oce_dyn: salt plume ---
    logical       :: SPP        = .false.          ! oce_modules.F90:96

    ! --- &oce_dyn: Gent-McWilliams ---
    logical       :: Fer_GM           = .false.    ! oce_modules.F90:38
    real(kind=WP) :: K_GM_max         = 3000.0_WP  ! oce_modules.F90:39
    real(kind=WP) :: K_GM_min         = 2.0_WP     ! oce_modules.F90:40
    integer       :: K_GM_bvref       = 2          ! oce_modules.F90:41
    real(kind=WP) :: K_GM_rampmax     = 40.0_WP    ! oce_modules.F90:43
    real(kind=WP) :: K_GM_rampmin     = 30.0_WP    ! oce_modules.F90:44
    real(kind=WP) :: K_GM_resscalorder = 2.0_WP    ! oce_modules.F90:42
    real(kind=WP) :: K_GM_cm          = 1.0_WP     ! oce_modules.F90:45
    real(kind=WP) :: K_GM_cmin        = 0.5_WP     ! oce_modules.F90:46
    logical       :: K_GM_Ktaper      = .false.    ! oce_modules.F90:47

    ! --- &oce_dyn: GM scaling ---
    logical       :: scaling_Ferreira   = .true.   ! oce_modules.F90:49
    logical       :: scaling_Rossby     = .false.  ! oce_modules.F90:50
    logical       :: scaling_resolution = .true.   ! oce_modules.F90:51
    logical       :: scaling_FESOM14    = .false.  ! oce_modules.F90:52
    logical       :: scaling_GMzexp     = .false.  ! oce_modules.F90:54
    real(kind=WP) :: GMzexp_zref        = 500.0_WP ! oce_modules.F90:55
    real(kind=WP) :: GMzexp_smin        = 0.1_WP   ! oce_modules.F90:56
    logical       :: scaling_GINsea     = .false.  ! oce_modules.F90:59
    real(kind=WP) :: GINsea_fac         = 2.0_WP   ! oce_modules.F90:60

    ! --- &oce_dyn: Redi ---
    logical       :: Redi        = .false.         ! oce_modules.F90:62
    logical       :: Redi_Ktaper = .false.         ! oce_modules.F90:63
    real(kind=WP) :: Redi_Kmax   = 3000.0_WP       ! oce_modules.F90:64
    real(kind=WP) :: Redi_Kmin   = 2.0_WP          ! oce_modules.F90:65

    ! --- &oce_dyn: ODM95 / LDD97 tapering ---
    logical       :: scaling_ODM95 = .true.        ! oce_modules.F90:67
    real(kind=WP) :: ODM95_Scr     = 1.0e-2_WP     ! oce_modules.F90:68
    real(kind=WP) :: ODM95_Sd      = 1.0e-3_WP     ! oce_modules.F90:69
    logical       :: scaling_LDD97 = .false.       ! oce_modules.F90:71
    real(kind=WP) :: LDD97_c       = 2.0_WP        ! oce_modules.F90:72
    real(kind=WP) :: LDD97_rmin    = 15.0e3_WP     ! oce_modules.F90:73
    real(kind=WP) :: LDD97_rmax    = 100.0e3_WP    ! oce_modules.F90:74

    ! --- &oce_dyn: vertical mixing scheme ---
    real(kind=WP) :: visc_sh_limit = 5.0e-3_WP     ! oce_modules.F90:76
    real(kind=WP) :: diff_sh_limit = 5.0e-3_WP     ! oce_modules.F90:77 (KPP shear diff, tracer_phys)
    character(25) :: mix_scheme    = 'KPP'         ! oce_modules.F90:81 'KPP','PP','TKE'
    real(kind=WP) :: Ricr          = 0.3_WP        ! oce_modules.F90:83
    real(kind=WP) :: concv         = 1.6_WP        ! oce_modules.F90:84
    logical       :: use_global_tides = .false.    ! oce_modules.F90:87

    namelist /oce_dyn/ C_d, A_ver, scale_area, SPP, &
        Fer_GM, K_GM_max, K_GM_min, K_GM_bvref, K_GM_rampmax, K_GM_rampmin, &
        K_GM_resscalorder, K_GM_cm, K_GM_cmin, K_GM_Ktaper, &
        scaling_Ferreira, scaling_Rossby, scaling_resolution, scaling_FESOM14, &
        scaling_GMzexp, GMzexp_zref, GMzexp_smin, scaling_GINsea, GINsea_fac, &
        scaling_ODM95, ODM95_Scr, ODM95_Sd, scaling_LDD97, LDD97_c, LDD97_rmin, LDD97_rmax, &
        Redi, Redi_Ktaper, Redi_Kmax, Redi_Kmin, &
        visc_sh_limit, mix_scheme, Ricr, concv, use_global_tides

    ! --- o_PARAM defaults not in &oce_dyn (used by M2 kernels) ---
    integer       :: mix_scheme_nmb = 1            ! oce_modules.F90:82 (set from mix_scheme)
    real(kind=WP) :: kappa        = 0.4_WP         ! oce_modules.F90:24 von Karman
    real(kind=WP) :: mix_coeff_PP = 0.01_WP        ! oce_modules.F90:25 PP mixing coef
    real(kind=WP) :: K_hor        = 10.0_WP        ! oce_modules.F90:27
    real(kind=WP) :: K_ver        = 0.00001_WP     ! oce_modules.F90:28
    real(kind=WP) :: alpha   = 1.0_WP              ! oce_modules.F90:90 elevation implicitness
    real(kind=WP) :: theta   = 1.0_WP              ! oce_modules.F90:90 divergence implicitness
    real(kind=WP) :: epsilon = 0.1_WP              ! oce_modules.F90:92 AB2 offset (shadows intrinsic, faithful)

    ! --- N2 smoothing (oce_modules.F90:104-106) ---
    ! FESOM2's N2smth_v (a thickness-weighted 1-2-1 vertical filter on N^2) is NOT ported:
    ! it defaulted .false., was never switched on by any driver, and N2_splines supersedes
    ! it -- a spline RECONSTRUCTS the derivative at the interface where the schemes need it,
    ! where the filter only smeared the two-point estimate over neighbouring levels.
    logical       :: N2smth_h    = .true.
    integer       :: N2smth_hidx = 1

    ! --- vertical shear reconstruction (oce_vert_spline) ---
    ! .false. = two-point difference of uvnode across the level pair, the historical form
    !           (bit-identical for PP and KPP's ri_iwmix).
    ! .true.  = cubic-spline d(uvnode)/dz evaluated AT the interface depth -- FESOM's
    !           analogue of ROMS RI_SPLINES. Restores the order of accuracy that the
    !           two-point form loses where the grid stretches (core2: dz 10 m -> 250 m).
    logical       :: shear_splines = .false.
    ! .true. = N^2 from the same cubic spline: N2 = g*(beta*dS/dx - alpha*dT/dx) at the
    !          interface, x = depth, alpha/beta (McDougall 1987, the sw_alpha_beta
    !          polynomial GM/Redi already uses for sigma_xy) at the spline-interpolated
    !          interface T,S. The chain rule IS the exact derivative of rho(T(z),S(z),p)
    !          at fixed p; the two-point difference is the approximation to it. Pair with
    !          shear_splines so Ri and the neutral slope divide like by like.
    logical       :: N2_splines    = .false.
    ! Upwind blending of the face relative-vorticity reconstruction in the vector-
    ! invariant momentum advection (momadv_opt==1, oce_dyn_vinv). Range [0,1]:
    !   0 = the plain 3-vertex average (bit-identical default; no upwind code executes)
    !   1 = fully upwind: the face value comes from the 1-2 inflow neighbours, weighted
    !       by inflow strength (d+|d| normalised).
    ! Energy-neutral for ANY value (u.[(f+zeta) x u] == 0 identically); acts on the
    ! vorticity/enstrophy dynamics only. VALUE-based env FESOM3_RVO_UPWIND.
    real(kind=WP) :: rvo_upwind    = 0.0_WP

    ! --- &tracer_phys: PP background diffusivity + mo_convect enhancements ---
    ! Members of the FESOM2 &tracer_phys namelist group (oce_modules.F90 cites below)
    ! read by M2.8 oce_mixing_pp (Kv0_const) and M2.8b mo_convect (the use_* / *_kv set).
    ! pi does NOT set any of these -> the FESOM2 defaults hold (Kv0_const=.true. routes
    ! the simple Kv = mix_coeff_PP*factor^3 + K_ver background; use_momix=.true. but it
    ! needs forcing/ice not present pre-forcing, so the M2.8b gate FORCES use_momix=.false.
    ! — momix is the deferred forcing-coupled path, gated at M2.10/M2.11).
    logical       :: Kv0_const     = .true.        ! oce_modules.F90:78 (const vs lat/depth Kv0)
    logical       :: use_momix     = .true.        ! oce_modules.F90:146 Monin-Obukhov (TB04)
    real(kind=WP) :: momix_lat     = -50.0_WP      ! oce_modules.F90:147 apply mo where lat<momix_lat
    real(kind=WP) :: momix_kv      = 0.01_WP       ! oce_modules.F90:148 mixing within MO length
    logical       :: use_instabmix = .true.        ! oce_modules.F90:152 convective adjustment
    real(kind=WP) :: instabmix_kv  = 0.1_WP        ! oce_modules.F90:153 convective Kv/Av amplitude
    ! Width of the convective ramp (= ROMS lmd_bvfcon). MUST be negative: the convective
    ! enhancement ramps 0 -> instabmix_kv as N^2 falls from 0 to instabmix_n2ref, and
    ! saturates below it (see oce_mo_conv). ROMS uses -2.0e-5 in an LMD/KPP model where the
    ! boundary-layer scheme already handles convection; mo_convect is the ONLY convection
    ! treatment under PP, so the default here is narrower, chosen so that an ordinary ocean
    ! instability (N^2 ~ -1e-6) is treated in full rather than at <1% of instabmix_kv.
    real(kind=WP) :: instabmix_n2ref = -1.0e-6_WP  ! [s^-2] convective ramp width
    logical       :: use_windmix   = .false.       ! oce_modules.F90:156 enhanced near-surface wind mixing
    real(kind=WP) :: windmix_kv    = 1.0e-3_WP     ! oce_modules.F90:157
    integer       :: windmix_nl    = 2             ! oce_modules.F90:158 # near-surface levels

    ! --- M5c: KPP nonlocal counter-gradient flux ---
    ! (use_sw_pene lives in mod_config — the g_config analog — NOT here; it is read by the KPP
    !  bldepth and the sw_3d tracer term from the single mod_config source.)
    ! use_kpp_nonlclflx (oce_modules.F90:168, &tracer_phys): adds the KPP nonlocal counter-gradient
    !   flux term. ⚠️ DEFAULTS .false. AND is ABSENT from work_core -> the ghats term is DEAD in the
    !   production CORE2 config (like MLD1_ind/K_hor in M4). The oracle guards it with this flag
    !   BEFORE the mix_scheme_nmb==1 test (oce_ale_tracer.F90:892). The term is transcribed (faithful
    !   port) but only fires when explicitly enabled; gated by the KPP_NONLCL=1 variant.
    logical       :: use_kpp_nonlclflx = .false.   ! oce_modules.F90:168 (&tracer_phys)
    ! Reference SSS for the KPP nonlocal salinity flux (oce_modules.F90:36-37, work_core
    ! namelist.tra: ref_sss_local=.true. -> rsss = local surface salinity; ref_sss=34 dead).
    real(kind=WP) :: ref_sss           = 34.7_WP   ! oce_modules.F90:37
    logical       :: ref_sss_local     = .false.   ! oce_modules.F90:36

    ! --- M7: TKE (cvmix_TKE) vertical mixing — &param_tke namelist DOUBLES (L38) ---
    ! Baked from the work_*_tke namelist.cvmix &param_tke group (init_cvmix_tke reads it,
    ! gen_modules_cvmix_tke.F90:166-169,313-322). mix_scheme='cvmix_TKE' -> mix_scheme_nmb==5
    ! -> tke_only=.true. (pure standalone TKE; IDEMIX uncoupled, nmb/=56). tke_mxl_choice==2 is
    ! the only ported mixing-length branch (Blanke-Delecluse). use_ubound/lbound_dirichlet keep
    ! the code defaults (.false. -> the Neumann BC). ⚠️ tke_cd: the namelist value 3.75 OVERRIDES
    ! the gen_modules code default 1.0 (the feedback_namelist_over_codedefault trap); it IS used
    ! at the Neumann (cd*forc_tke_surf**(3./2.))/dzt(1) surface term despite the "Dirichlet" name.
    real(kind=WP) :: tke_c_k                  = 0.1_WP    ! TKE parameter c_k
    real(kind=WP) :: tke_c_eps                = 0.7_WP    ! dissipation parameter c_eps
    real(kind=WP) :: tke_alpha                = 30.0_WP   ! diffusion-of-TKE parameter alpha_tke
    real(kind=WP) :: tke_mxl_min              = 1.0e-8_WP ! minimum mixing length [m]
    real(kind=WP) :: tke_kappaM_min           = 0.0_WP    ! minimum Kappa_M [m2/s]
    real(kind=WP) :: tke_kappaM_max           = 100.0_WP  ! maximum Kappa_M [m2/s]
    real(kind=WP) :: tke_cd                   = 3.75_WP   ! surface BC coeff (namelist override of 1.0)
    real(kind=WP) :: tke_surf_min             = 1.0e-4_WP ! minimum surface TKE [m2/s2]
    real(kind=WP) :: tke_min                  = 1.0e-6_WP ! minimum interior TKE [m2/s2]
    integer       :: tke_mxl_choice           = 2         ! 2 = Blanke-Delecluse (only ported branch)
    logical       :: tke_only                 = .true.    ! standalone TKE (nmb/=56 -> IDEMIX off)
    logical       :: tke_use_ubound_dirichlet = .false.   ! .false. -> Neumann surface BC
    logical       :: tke_use_lbound_dirichlet = .false.   ! .false. -> Neumann bottom BC
    logical       :: tke_dolangmuir           = .false.   ! Langmuir term (l_lc); dead here

contains

    subroutine read_param_phys(nml_path, ierr)
        ! Read namelist.oce &oce_dyn once. ierr=0 success; >0 error.
        character(len=*), intent(in)  :: nml_path
        integer,          intent(out) :: ierr
        integer :: u, ios

        ierr = 0
        open(newunit=u, file=trim(nml_path), status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(*,'(a)') 'read_param_phys: cannot open '//trim(nml_path)
            ierr = 1; return
        end if

        rewind(u)
        read(u, nml=oce_dyn, iostat=ios)
        if (ios > 0) then
            write(*,'(a)') 'read_param_phys: malformed namelist group &oce_dyn'
            ierr = 2
        end if
        close(u)
    end subroutine read_param_phys

end module mod_param_phys
