module mod_ice_thermo
    ! M3d: sea-ice thermodynamics. Faithful transcription of FESOM2 v2.7.3
    ! ice_thermo_oce.F90 — cut_off (hmin/Armin clamp) + thermodynamics (the per-node
    ! growth/melt driver) + therm_ice (0-layer Semtner/Hibler-1984 ice-class growth) +
    ! budget (thick-ice 5-iter Newton-Raphson surface temperature) + obudget (open-ocean
    ! growth rate + evaporation) + flooding (snow->ice conversion) + TFrez (Millero-1978
    ! freezing point). Standard path (NO icepack/meltponds/cavity/oasis-yac/oifs).
    !
    ! ATMOSPHERIC FORCING / FLUX DIAGNOSTICS: FESOM2's thermodynamics reads/writes these via
    ! the GLOBAL modules g_forcing_arrays / g_forcing_param / o_arrays. FESOM3 has NO global
    ! mutable forcing state (the explicit-dataflow architecture), so they are bundled into
    ! t_atmflux and passed as one argument: the per-node atmospheric INPUTS (shortwave/
    ! longwave/Tair/shum/prec/runoff/wind/Ch-Ce_atm_oce), the thermo flux OUTPUTS
    ! (evaporation/ice_sublimation/flice/real_salt_flux/fw_ice-snw/hf_Q*), and the scalar
    ! config (Ch/Ce_atm_ice/ref_sss/ref_sss_local/use_virt_salt/l_snow). The GATED outputs
    ! (a_ice/m_ice/m_snow/flx_h/flx_fw/t_skin) all live in t_ice. M3e (oce_fluxes) consumes
    ! the t_atmflux ocean fluxes (real_salt_flux/fw_ice/fw_snw/evaporation + ice%flx_h/fw).
    !
    ! CONFIG (the driver overrides these to the CORE2 namelist values, like M3b/M3c):
    !   - &ice_therm params are read by the oracle as namelist DOUBLES (con=2.1656_WP,
    !     consn=0.31_WP, hmin/Armin=0.01_WP, emiss=0.97_WP, albsn/snm/i/im, albw=0.1_WP,
    !     Sice=4.0_WP, h0/h0_s=0.5_WP, c_melt=0.5_WP) — NOT the t_ice_thermo single->WP
    !     defaults; the driver sets them as _WP doubles. h_ml=2.5_WP (not in namelist, suffixed
    !     double both sides), cc=rhowat*4190 / cl=rhoice*3.34e5 (ice_init recomputes to the
    !     exactly-representable values = the single->WP defaults), rho*/inv_rho*/clh*/tmelt/
    !     boltzmann/cpair (NOT in namelist -> MOD_ICE single->WP defaults match on both sides).
    !   - Ch_atm_ice=Ce_atm_ice=0.00175_WP (namelist.forcing doubles, NOT 1.75e-3 single->WP);
    !     ref_sss_local=.true. (=> rsss=S_oc; ref_sss value dead); use_virt_salt=.true. (linfs
    !     => the (rsss-Sice)/rsss freshwater branch); l_snow=.true. (=> rain=prec_rain,
    !     snow=prec_snow, evap_in=0, prec arrays not written back); cd_oce_ice=0.0055_WP
    !     (the M3b override, read by ustar = sqrt(((u_ice-u_w)^2+(v_ice-v_w)^2)*cd_oce_ice)).
    !
    ! M2.12 optional-partit pattern: partit absent OR npes==1 -> the proven 1-rank path
    ! VERBATIM (owned_bounds returns global counts, is_multirank=.false.); present+npes>1 ->
    ! owned/halo loop bounds + the FESOM2 exchange (exchange_nod(ustar) after the friction-
    ! velocity loop, which loops owned-only). The per-node arithmetic is UNCHANGED so codegen
    ! — and the byte-match — is preserved. (Multi-rank ice = M3f; M3d is the CORE2 1-rank gate.)
    !
    ! BIT-IDENTITY NOTES (the L9 transitive-gate pattern; every operand already byte-pinned):
    !  - therm_ice/budget/obudget/flooding/TFrez are all SCALAR per-node arithmetic (no array
    !    loops to vectorise), so every runtime divide is scalar on both sides — the L29
    !    vectorised-vs-scalar divide trap cannot arise here; -no-prec-div reciprocals agree
    !    whenever the operands agree (they do: prescribed forcing + M3a/b/c-proven ice state +
    !    do_ic3d-proven srfoce_temp/salt + geometry-proven geo_coord_nod2D).
    !  - the pointers mirror the oracle's ithermp access pattern to keep codegen identical.
    use mod_precision,   only: WP
    use mod_mesh,        only: t_mesh
    use mod_ice,         only: t_ice, t_ice_thermo
    use mod_partit,      only: t_partit
    use mod_part_bounds, only: owned_bounds, is_multirank
    use mod_halo,        only: exchange_nod
    implicit none
    private
    public :: t_atmflux, cut_off, thermodynamics

    !___________________________________________________________________________
    ! atmospheric forcing + thermo flux bundle (FESOM2 g_forcing_arrays / g_forcing_param /
    ! o_arrays analog). Scalar-config defaults = the FESOM2 module defaults (single->WP); the
    ! driver overrides the ones the CORE2 namelist sets (Ch/Ce_atm_ice, ref_sss_local).
    type t_atmflux
        ! atmospheric INPUTS (prescribed by the driver / forcing read)
        real(kind=WP), allocatable :: shortwave(:), longwave(:), Tair(:), shum(:)
        real(kind=WP), allocatable :: prec_rain(:), prec_snow(:), runoff(:)
        real(kind=WP), allocatable :: u_wind(:), v_wind(:)
        real(kind=WP), allocatable :: Ch_atm_oce_arr(:), Ce_atm_oce_arr(:)
        ! thermo flux OUTPUTS (diagnostics + ocean fluxes; not gated, consumed by M3e)
        real(kind=WP), allocatable :: evaporation(:), ice_sublimation(:)
        real(kind=WP), allocatable :: flice(:), real_salt_flux(:)
        real(kind=WP), allocatable :: fw_ice(:), fw_snw(:)
        real(kind=WP), allocatable :: hf_Qlat(:), hf_Qsen(:), hf_Qradtot(:)
        real(kind=WP), allocatable :: hf_Qswr(:), hf_Qlwr(:), hf_Qlwrout(:)
        ! M3e oce_fluxes coupling-out (ice_oce_coupling.F90 analogs of o_ARRAYS). The
        ! surface-flux OUTPUTS heat_flux/water_flux/virtual_salt/relax_salt are exactly the
        ! step_oce surface boundary conditions the ocean step consumes (and heat_flux_in is
        ! the pre-SW-penetration copy); stress_node_surf is the total (ice+atm) momentum
        ! stress on nodes; stress_surf (2,elem2D) is produced by oce_fluxes_mom into a
        ! separate caller-owned array (it is the dynamics input), not stored here. The atm-
        ! ocean stress stress_atmoce_x/y and the SSS-restoring climatology Ssurf are INPUTS
        ! (FESOM2's bulk-formula / do_ic3d products); prescribed analytically in the M3e gate.
        real(kind=WP), allocatable :: heat_flux(:), water_flux(:), heat_flux_in(:)
        real(kind=WP), allocatable :: virtual_salt(:), relax_salt(:)
        real(kind=WP), allocatable :: stress_node_surf(:,:)
        real(kind=WP), allocatable :: stress_atmoce_x(:), stress_atmoce_y(:)
        real(kind=WP), allocatable :: Ssurf(:)
        ! scalar config (g_forcing_param / o_param / g_sbf analogs)
        real(kind=WP) :: Ch_atm_ice = 1.75e-3, Ce_atm_ice = 1.75e-3
        real(kind=WP) :: ref_sss = 34.7
        ! surface salinity restoring coefficient [m/s]: FESOM2 o_PARAM default (oce_modules.F90:31);
        ! CORE2 namelist.tra sets 1.929e-06 (the M3e driver re-sets that _WP double).
        real(kind=WP) :: surf_relax_S = 10.0_WP/(60*3600.0_WP*24)
        logical       :: ref_sss_local = .false.
        logical       :: use_virt_salt = .true.
        logical       :: l_snow        = .true.
    end type t_atmflux

contains

    !___________________________________________________________________________
    ! cut_off (FESOM2 ice_thermo_oce.F90:70). Clamp a_ice into [0,1] and zero a/m_ice/m_snow
    ! together when concentration or ice mass falls below 1e-9. Loops owned+halo (nNodL).
    !
    ! FESOM2 deletes that ice and snow without any flux. Behind the ice FCT, which limits a_ice
    ! and m_ice separately, this hits open-water nodes ahead of a moving ice edge (a_ice ~ 0,
    ! m_ice 1e-5..2e-4 m): on CORE2 ~65 km^3 of ice a year, -2.3e11 psu m^3 of ocean + ice salt
    ! under zstar. CONSERVING CUT_OFF (ice%thermo%cutoff_conserve, FESOM3_ICE_CUTOFF_CONSERVE=1,
    ! not in FESOM2): record what is deleted, dm_ice_cut = m_ice(before) - m_ice(after) and
    ! dm_snow_cut the same for m_snow [m], 0 exactly at a node that is not cut; thermodynamics
    ! returns them to the ocean in this step. A negative FCT undershoot (m ~ -1e-17) that the
    ! m_ice test zeroes is recorded with its sign, so the budget closes exactly. Off: the arrays
    ! are not allocated and only FESOM2's loop runs.
    subroutine cut_off(ice, mesh, partit)
        type(t_ice),    intent(inout), target :: ice
        type(t_mesh),   intent(in)            :: mesh
        type(t_partit), intent(in), optional  :: partit
        integer :: n, nNodO, nNodL, nEdgeO, nElemO
        real(kind=WP), dimension(:), pointer  :: a_ice, m_ice, m_snow
        real(kind=WP), dimension(:), pointer  :: dm_ice_cut, dm_snow_cut

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        a_ice  => ice%data(1)%values(:)
        m_ice  => ice%data(2)%values(:)
        m_snow => ice%data(3)%values(:)

        ! conserving cut_off: the ice and snow before the cut
        if (ice%thermo%cutoff_conserve) then
            dm_ice_cut  => ice%thermo%dm_ice_cut(:)
            dm_snow_cut => ice%thermo%dm_snow_cut(:)
            do n = 1, nNodL
                dm_ice_cut(n)  = m_ice(n)
                dm_snow_cut(n) = m_snow(n)
            end do
        end if

        do n = 1, nNodL
            ! upper cutoff: a_ice
            if (a_ice(n) > 1.0_WP)   a_ice(n) = 1.0_WP
            ! lower cutoff: a_ice
            if (a_ice(n) < .1e-8_WP) then
                a_ice(n)  = 0.0_WP
                m_ice(n)  = 0.0_WP
                m_snow(n) = 0.0_WP
            end if
            ! lower cutoff: m_ice
            if (m_ice(n) < .1e-8_WP) then
                m_ice(n)  = 0.0_WP
                m_snow(n) = 0.0_WP
                a_ice(n)  = 0.0_WP
            end if
        end do

        ! conserving cut_off: the deleted ice and snow (before - after)
        if (ice%thermo%cutoff_conserve) then
            do n = 1, nNodL
                dm_ice_cut(n)  = dm_ice_cut(n)  - m_ice(n)
                dm_snow_cut(n) = dm_snow_cut(n) - m_snow(n)
            end do
        end if
    end subroutine cut_off

    !___________________________________________________________________________
    ! thermodynamics (FESOM2 ice_thermo_oce.F90:148). For every surface node extract the
    ! inputs, call therm_ice, and write the prognostic ice state + air-sea fluxes back.
    !
    ! CONSERVING CUT_OFF (ice%thermo%cutoff_conserve, FESOM3_ICE_CUTOFF_CONSERVE=1; not in
    ! FESOM2). The ice and snow that cut_off deleted at a node in this step (dm_ice_cut,
    ! dm_snow_cut) melt at that node in this step, as if therm_ice had melted them. After
    ! therm_ice, with the melt rates dhcut = dm_ice_cut/ice_dt and dhsncut = dm_snow_cut/ice_dt:
    !   fwice_c = dhcut*rhoice*inv_rhowat          (use_virt_salt: *(rsss-Sice)/rsss, therm_ice's form)
    !   fwsnw_c = dhsncut*rhosno*inv_rhowat
    !   fw = fw + fwice_c + fwsnw_c;  fwice = fwice + fwice_c;  fwsnw = fwsnw + fwsnw_c
    !   rsf = rsf + fwice_c*Sice                   (.not. use_virt_salt only, as therm_ice)
    !   thdgr = thdgr - dhcut;  thdgrsn = thdgrsn - dhsncut   (negative growth)
    ! So the ocean's salt gains Sice*rhoice*inv_rhowat*dm_ice_cut per unit area, what the ice
    ! lost, and oce_fluxes' zstar freshwater balance (-thdgr*rhoice*inv_rhowat - thdgrsn*rhosno*
    ! inv_rhowat) sees the cut ice and snow exactly as it sees thermodynamic melt: ocean + ice
    ! salt closes to roundoff under zstar. The latent heat is NOT taken from the ocean (FESOM2's
    ! cut_off never accounted heat): on CORE2 ~3.5e-6 K a year of ocean warming equivalent.
    subroutine thermodynamics(ice, mesh, atm, partit)
        type(t_ice),     intent(inout), target :: ice
        type(t_mesh),    intent(in),    target :: mesh
        type(t_atmflux), intent(inout), target :: atm
        type(t_partit),  intent(in),    optional :: partit
        integer :: i, nNodO, nNodL, nEdgeO, nElemO
        logical :: lmr
        real(kind=WP) :: h,hsn,A,fsh,flo,Ta,qa,rain,snow,runo,rsss,rsf,evap_in
        real(kind=WP) :: ug,ustar,T_oc,S_oc,h_ml,t,ch,ce,ch_i,ce_i,fw,fwice,fwsnw,ehf,evap
        real(kind=WP) :: ithdgr, ithdgrsn, ithdgra, iflice, hflatow, hfsenow, hflwrdout
        real(kind=WP) :: subli, hfswrow, hflwrow, hfradow, lid_clo, geolon, geolat
        real(kind=WP) :: dhcut, dhsncut, fwice_c, fwsnw_c   ! conserving cut_off
        !_______________________________________________________________________
        real(kind=WP), dimension(:), pointer :: u_ice, v_ice, a_ice, m_ice, m_snow
        real(kind=WP), dimension(:), pointer :: a_ice_old, m_ice_old, m_snow_old
        real(kind=WP), dimension(:), pointer :: thdgr, thdgrsn, thdgra, thdgr_old, t_skin, ustar_aux
        real(kind=WP), dimension(:), pointer :: u_w, v_w, T_oc_array, S_oc_array
        real(kind=WP), dimension(:), pointer :: net_heat_flux, fresh_wa_flux

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        lmr = is_multirank(partit)

        u_ice         => ice%uice(:)
        v_ice         => ice%vice(:)
        a_ice         => ice%data(1)%values(:)
        m_ice         => ice%data(2)%values(:)
        m_snow        => ice%data(3)%values(:)
        a_ice_old     => ice%data(1)%values_old(:)
        m_ice_old     => ice%data(2)%values_old(:)
        m_snow_old    => ice%data(3)%values_old(:)
        thdgr         => ice%thermo%thdgr
        thdgrsn       => ice%thermo%thdgrsn
        thdgra        => ice%thermo%thdgra
        thdgr_old     => ice%thermo%thdgr_old
        t_skin        => ice%thermo%t_skin
        ustar_aux     => ice%thermo%ustar
        u_w           => ice%srfoce_u(:)
        v_w           => ice%srfoce_v(:)
        T_oc_array    => ice%srfoce_temp(:)
        S_oc_array    => ice%srfoce_salt(:)
        net_heat_flux => ice%flx_h(:)
        fresh_wa_flux => ice%flx_fw(:)

        !_______________________________________________________________________
        rsss = atm%ref_sss

        ! Friction velocity (owned only), then broadcast to the halo.
        do i = 1, nNodO
            ustar = 0.0_WP
            if (mesh%ulevels_nod2D(i) > 1) cycle
            ustar = ((u_ice(i)-u_w(i))**2 + (v_ice(i)-v_w(i))**2)
            ustar_aux(i) = sqrt(ustar*ice%cd_oce_ice)
        end do
        if (lmr) call exchange_nod(ustar_aux, partit)

        !_______________________________________________________________________
        do i = 1, nNodL
            ! if there is a cavity no sea ice thermodynamics is applied
            if (mesh%ulevels_nod2D(i) > 1) cycle

            ! prepare inputs for ice thermodynamics step
            h   = m_ice(i)
            hsn = m_snow(i)
            A   = a_ice(i)
            fsh = atm%shortwave(i)
            flo = atm%longwave(i)
            Ta  = atm%Tair(i)
            qa  = atm%shum(i)

            if (.not. atm%l_snow) then
                if (Ta >= 0.0_WP) then
                    rain = atm%prec_rain(i)
                    snow = 0.0_WP
                else
                    rain = 0.0_WP
                    snow = atm%prec_rain(i)
                end if
                evap_in = atm%evaporation(i)   ! evap_in: positive up
            else
                rain    = atm%prec_rain(i)
                snow    = atm%prec_snow(i)
                evap_in = 0.0_WP
            end if
            runo = atm%runoff(i)
            ug   = sqrt(atm%u_wind(i)**2 + atm%v_wind(i)**2)
            ustar= ustar_aux(i)
            T_oc = T_oc_array(i)
            S_oc = S_oc_array(i)
            if (atm%ref_sss_local) rsss = S_oc
            t    = t_skin(i)
            ch   = atm%Ch_atm_oce_arr(i)
            ce   = atm%Ce_atm_oce_arr(i)
            ch_i = atm%Ch_atm_ice
            ce_i = atm%Ce_atm_ice
            h_ml = ice%thermo%h_ml
            fw    = 0.0_WP
            fwice = 0.0_WP
            fwsnw = 0.0_WP
            ithdgra = 0.0_WP
            ehf   = 0.0_WP
            geolon = mesh%geo_coord_nod2D(1, i)
            geolat = mesh%geo_coord_nod2D(2, i)

            if (geolat > 0) then   ! TODO 2 separate pars for each hemisphere
                lid_clo = ice%thermo%h0
            else
                lid_clo = ice%thermo%h0_s
            end if

            ! do ice thermodynamics
            call therm_ice(ice%thermo, h, hsn, A, fsh, flo, Ta, qa, rain, snow, runo, rsss, &
                           ug, ustar, T_oc, S_oc, h_ml, t, ice%ice_dt, ch, ce, ch_i, ce_i,    &
                           evap_in, fw, fwice, fwsnw, ehf, evap, rsf, ithdgr, ithdgrsn, ithdgra, iflice, &
                           hflatow, hfsenow, hflwrdout, hfswrow, hflwrow, hfradow, lid_clo, geolon, geolat, subli, &
                           atm%use_virt_salt)

            ! conserving cut_off: the ice and snow cut_off deleted here melt in this step
            ! (the rule in the header above; latent heat not returned)
            if (ice%thermo%cutoff_conserve) then
                dhcut   = ice%thermo%dm_ice_cut(i)/ice%ice_dt
                dhsncut = ice%thermo%dm_snow_cut(i)/ice%ice_dt
                if (.not. atm%use_virt_salt) then
                    fwice_c = dhcut*ice%thermo%rhoice*ice%thermo%inv_rhowat
                    fwsnw_c = dhsncut*ice%thermo%rhosno*ice%thermo%inv_rhowat
                    rsf     = rsf + fwice_c*ice%thermo%Sice
                else
                    fwice_c = dhcut*ice%thermo%rhoice*ice%thermo%inv_rhowat*(rsss-ice%thermo%Sice)/rsss
                    fwsnw_c = dhsncut*ice%thermo%rhosno*ice%thermo%inv_rhowat
                end if
                fw       = fw + fwice_c + fwsnw_c
                fwice    = fwice + fwice_c
                fwsnw    = fwsnw + fwsnw_c
                ithdgr   = ithdgr - dhcut
                ithdgrsn = ithdgrsn - dhsncut
            end if

            ! write ice thermodyn. results into arrays. backup of old values
            m_ice_old(i)  = m_ice(i)
            m_snow_old(i) = m_snow(i)
            a_ice_old(i)  = a_ice(i)
            thdgr_old(i)  = thdgr(i)

            ! new values
            m_ice(i)  = h
            m_snow(i) = hsn
            a_ice(i)  = A

            t_skin(i)        = t
            fresh_wa_flux(i) = fw      ! positive down
            net_heat_flux(i) = ehf     ! positive down
            atm%evaporation(i)     = evap     ! negative up
            atm%ice_sublimation(i) = subli

            thdgr(i)     = ithdgr
            thdgrsn(i)   = ithdgrsn
            thdgra(i)    = ithdgra
            atm%flice(i) = iflice

            atm%fw_ice(i)     = - fwice     ! freshwater flux from ice
            atm%fw_snw(i)     = - fwsnw     ! freshwater flux from snow
            atm%hf_Qlat(i)    = - hflatow   ! latent heat flux
            atm%hf_Qsen(i)    = - hfsenow   ! sensible heat flux
            atm%hf_Qradtot(i) = - hfradow   ! total radiation heat flux
            atm%hf_Qswr(i)    = - hfswrow   ! shortwave radiation heat flux incoming
            atm%hf_Qlwr(i)    = - hflwrow   ! longwave radiation heatflux incoming
            atm%hf_Qlwrout(i) = - hflwrdout ! longwave radiation heat flux outgoing

            ! real salt flux due to salinity contained in the sea ice 4-5 psu
            atm%real_salt_flux(i) = rsf

            ! if snow file is not given snow computed from prec_rain --> but prec_snow
            ! array needs to be filled so the freshwater balancing adds up
            if (.not. atm%l_snow) then
                atm%prec_rain(i) = rain
                atm%prec_snow(i) = snow
            end if
        end do
    end subroutine thermodynamics

    !___________________________________________________________________________
    ! therm_ice (FESOM2 ice_thermo_oce.F90:367). 0-layer ice thermodynamic growth model.
    ! Inputs h/hsn/A (ice/snow mass, concentration) + atmosphere + ocean; outputs the updated
    ! h/hsn/A/t plus the freshwater (fw/fwice/fwsnw), heat (ehf), salt (rsf) and growth-rate
    ! diagnostics. use_virt_salt passed in (FESOM2 reads it from g_forcing_param).
    subroutine therm_ice(ithermp, h, hsn, A, fsh, flo, Ta, qa, rain, snow, runo, rsss, &
                         ug, ustar, T_oc, S_oc, H_ML, t, ice_dt, ch, ce, ch_i, ce_i,    &
                         evap_in, fw, fwice, fwsnw, ehf, evap, rsf, dhgrowth, dhsngrowth, dAgrowth, iflice, &
                         hflatow, hfsenow, hflwrdout, hfswrow, hflwrow, hfradow, lid_clo, geolon, geolat, subli, &
                         use_virt_salt)
        type(t_ice_thermo), intent(in), target :: ithermp
        integer k
        real(kind=WP)  h,hsn,A,Aold,fsh,flo,Ta,qa,rain,snow,runo,rsss,evap_in
        real(kind=WP)  ug,ustar,T_oc,S_oc,H_ML,t,ice_dt,ch,ce,ch_i,ce_i,fw,fwice,fwsnw,ehf
        real(kind=WP)  dhgrowth,dhsngrowth,dAgrowth,ahf,prec,subli,subli_i,rsf
        real(kind=WP)  rhow,show,rhice,shice,sh,snthick,thick,thact
        real(kind=WP)  rh,rA,qhst,sn,hsntmp,o2ihf,evap
        real(kind=WP)  iflice, hflatow, hfsenow, hflwrdout, hfswrow, hflwrow, hfradow
        real(kind=WP)  lid_clo, geolon, geolat
        logical        use_virt_salt
        !_______________________________________________________________________
        logical      , pointer :: snowdist, new_iclasses
        integer      , pointer :: iclasses, open_water_albedo
        real(kind=WP), pointer :: hmin, Sice, Armin, cc, cl, con, consn, rhosno, rhoice, &
                                  inv_rhowat, inv_rhosno, c_melt, h_cutoff
        real(kind=WP), pointer, dimension (:) :: hpdf
        snowdist          => ithermp%snowdist
        new_iclasses      => ithermp%new_iclasses
        iclasses          => ithermp%iclasses
        open_water_albedo => ithermp%open_water_albedo
        hmin       => ithermp%hmin
        Armin      => ithermp%Armin
        Sice       => ithermp%Sice
        cc         => ithermp%cc
        cl         => ithermp%cl
        con        => ithermp%con
        consn      => ithermp%consn
        rhosno     => ithermp%rhosno
        rhoice     => ithermp%rhoice
        inv_rhowat => ithermp%inv_rhowat
        inv_rhosno => ithermp%inv_rhosno
        c_melt     => ithermp%c_melt
        h_cutoff   => ithermp%h_cutoff
        hpdf       => ithermp%hpdf

        !_______________________________________________________________________
        ! Store ice thickness at start of growth routine
        dhgrowth=h

        ! effective ice/snow thickness on the ice-covered part (0-layer Semtner 1976)
        snthick=hsn*(con/consn)/max(A,Armin)  ! Effective snow thickness
        thick=h/max(A,Armin)                  ! Effective ice thickness
        if (snowdist) thick=snthick+thick     ! Effective ice and snow thickness

        ! Growth rate for ice in open ocean
        rhow=0.0_WP
        evap=0.0_WP
        call obudget(ithermp, qa,fsh,flo,T_oc,ug,ta,ch,ce,geolon, geolat, rhow, evap, &
                     hflatow, hfsenow, hflwrdout, hfswrow, hflwrow, hfradow)
        hflatow  = hflatow  *(1.0_WP-A)   ! latent heatflux
        hfsenow  = hfsenow  *(1.0_WP-A)   ! sensible heatflux
        hfradow  = hfradow  *(1.0_WP-A)   ! total radiation hfswrow+hflwrow+hflwrdout
        hfswrow  = hfswrow  *(1.0_WP-A)   ! incoming shortwave radiation
        hflwrow  = hflwrow  *(1.0_WP-A)   ! incoming longwave radiation
        hflwrdout= hflwrdout*(1.0_WP-A)   ! outgoing long wave radiation

        ! growth rate of ice in ice covered part (Hibler 1984, 7-level thickness distribution)
        rhice=0.0_WP
        subli=0.0_WP
        if (thick.gt.hmin) then
            do k=1,iclasses
                thact = real((2*k-1),WP)*thick/real(iclasses,WP) ! Thicknesses of actual ice class
                if(new_iclasses) thact=h_cutoff/2.*thact       ! h_cutoff is variable
                if(.not. snowdist) thact=thact+snthick         ! snow same on every class if snowdist
                call budget(ithermp, thact, hsn,t,Ta,qa,fsh,flo,ug,S_oc,ch_i,ce_i,shice,subli_i)
                !Thick ice K-class growth rate
                if(new_iclasses) then
                    rhice=rhice+shice*hpdf(k)
                    subli=subli+subli_i*hpdf(k)
                 else
                    rhice=rhice+shice
                    subli=subli+subli_i
                 end if
            end do
            if(.not. new_iclasses) then
                rhice=rhice/real(iclasses,WP)      ! Add to average heat flux
                subli=subli/real(iclasses,WP)
            end if
        end if

        ! Convert growth rates [m ice/sec] into growth per time step DT.
        rhow=rhow*ice_dt
        rhice=rhice*ice_dt

        ! Multiply ice growth of open water and ice with the areal fractions of grid cell
        show =rhow*(1.0_WP-A)
        shice=rhice*A
        sh   =show+shice

        ! Store atmospheric heat flux, average over grid cell [W/m**2]
        ahf=-cl*sh/ice_dt

        ! precipitation (into the ocean)
        prec=rain+runo+snow*(1.0_WP-A)              ! m water/s

        ! snow fall above ice
        hsn=hsn+snow*ice_dt*A*1000.0_WP*inv_rhosno  ! Add snow fall to temporary snow thickness
        dhsngrowth=hsn                              ! Store snow thickness after snow fall

        evap=evap*(1.0_WP-A)                        ! m water/s
        subli=subli*A

        ! If there is atmospheric melting, first melt any snow present.
        hsntmp= -min(sh,0.0_WP)*rhoice*inv_rhosno
        hsntmp=min(hsntmp,hsn)                      ! Do not melt more snow than available
        hsn=hsn-hsntmp                              ! Update snow thickness after atm snow melt

        ! Negative atmospheric heat flux left after melting of snow
        rh=sh+hsntmp*rhosno/rhoice
        h=max(h,0.0_WP)

        ! Ocean-to-ice heat flux as a function of temperature difference and friction velocity
        o2ihf= (T_oc-TFrez(S_oc))*0.006_WP*ustar*cc*A  &
            +(T_oc-Tfrez(S_oc))*H_ML/ice_dt*cc*(1.0_WP-A)      ! [W/m2]
        rh=rh-o2ihf*ice_dt/cl
        qhst=h+rh                                          ! [m]

        ! Melt snow if there is any ML heat content left (qhst<0).
        sn=hsn+min(qhst,0.0_WP)*rhoice*inv_rhosno
        sn=max(sn,0.0_WP)                          ! New temporary snow thickness >= 0

        ! Update snow and ice depth
        hsn=sn
        h=max(qhst,0.0_WP)
        if (h.lt.1E-6_WP) h=0._WP                   ! Avoid very small ice thicknesses

        ! heat and fresh water fluxes
        dhgrowth=h-dhgrowth        ! Change in ice thickness due to thermodynamic effects
        dhsngrowth=hsn-dhsngrowth  ! Change in snow thickness due to thermodynamic melting

        dhgrowth=dhgrowth/ice_dt       ! Conversion: 'per time step' -> 'per second'
        dhsngrowth=dhsngrowth/ice_dt   ! Conversion: 'per time step' -> 'per second'

        ehf = ahf + cl*(dhgrowth+(rhosno/rhoice)*dhsngrowth)

        ! (prec+runoff)+evap - freezing(+melting) ice&snow
        if (.not. use_virt_salt) then
            fwice = - dhgrowth*rhoice*inv_rhowat
            fwsnw = - dhsngrowth*rhosno*inv_rhowat
            fw= prec + evap + fwice + fwsnw
            rsf= fwice*Sice
        else
            fwice = - dhgrowth*rhoice*inv_rhowat*(rsss-Sice)/rsss
            fwsnw = - dhsngrowth*rhosno*inv_rhowat
            fw= prec + evap + fwice + fwsnw
            rsf = 0.0_WP
        end if

        ! Changes in compactnesses (equation 16 of Hibler 1979)
        rh=-min(h,-rh)   ! Make sure we do not melt more ice than is available
        rA= rhow - o2ihf*ice_dt/cl
        Aold = A
        A=A + c_melt*min(rh,0.0_WP)*A/max(h,hmin) + max(rA,0.0_WP)*(1._WP-A)/lid_clo
        !meaning:           melting                         freezing

        A=min(A,h*1.e6_WP)         ! A -> 0 for h -> 0
        A=min(max(A,0.0_WP),1._WP) ! A >= 0, A <= 1
        dAgrowth = (A-Aold)/ice_dt

        ! Flooding (snow to ice conversion)
        iflice=h
        call flooding(ithermp, h, hsn)
        iflice=(h-iflice)/ice_dt

        ! to maintain salt conservation for the current model version
        if (.not. use_virt_salt) then
            rsf=rsf-iflice*rhoice*inv_rhowat*Sice
        else
            fw=fw+iflice*rhoice*inv_rhowat*Sice/rsss
        end if

        evap=evap+subli

    end subroutine therm_ice

    !___________________________________________________________________________
    ! budget (FESOM2 ice_thermo_oce.F90:657). Thick-ice growth rate [m ice/sec] via a
    ! 5-iteration Newton-Raphson for the ice surface temperature t (modified in place).
    subroutine budget(ithermp, hice,hsn,t,ta,qa,fsh,flo,ug,S_oc,ch_i,ce_i,fh,subli)
        type(t_ice_thermo), intent(in), target :: ithermp
        integer iter, imax      ! Number of iterations
        real(kind=WP)  hice,hsn,t,ta,qa,fsh,flo,ug,S_oc,ch_i,ce_i,fh
        real(kind=WP)  hfsen,hfrad,hflat,hftot,subli
        real(kind=WP)  alb             ! Albedo of sea ice
        real(kind=WP)  q1, q2          ! coefficients for saturated specific humidity
        real(kind=WP)  A1,A2,A3,B,C, d1, d2, d3
        !_______________________________________________________________________
        real(kind=WP), pointer :: boltzmann, emiss_ice, tmelt, cl, clhi, con, cpair, &
                                  inv_rhowat, inv_rhoair, rhoair, albim, albi, albsn, albsnm
        boltzmann  => ithermp%boltzmann
        emiss_ice  => ithermp%emiss_ice
        tmelt      => ithermp%tmelt
        cl         => ithermp%cl
        clhi       => ithermp%clhi
        con        => ithermp%con
        cpair      => ithermp%cpair
        inv_rhowat => ithermp%inv_rhowat
        inv_rhoair => ithermp%inv_rhoair
        rhoair     => ithermp%rhoair
        albim      => ithermp%albim
        albi       => ithermp%albi
        albsn      => ithermp%albsn
        albsnm     => ithermp%albsnm

        !_______________________________________________________________________
        q1   = 11637800.0_WP
        q2   = -5897.8_WP
        imax = 5

        ! set albedo (ice/snow, freezing/melting distinguished)
        if (t<0.0_WP) then ! --> freezing condition
            if (hsn.gt.0.0_WP) then ! --> snow cover present
                alb=albsn
            else                    ! --> no snow cover
                alb=albi
            endif
        else               ! --> melting condition
            if (hsn.gt.0.0_WP) then ! --> snow cover present
                alb=albsnm
            else                    ! --> no snow cover
                alb=albim
            endif
        endif

        d1=rhoair*cpair*Ch_i
        d2=rhoair*Ce_i
        d3=d2*clhi

        ! total incoming atmospheric heat flux
        A1=(1.0_WP-alb)*fsh + flo + d1*ug*ta + d3*ug*qa
        ! Newton-Raphson to get temperature at the top of the ice layer
        do iter=1,imax
            B=q1*inv_rhoair*exp(q2/(t+tmelt))       ! (saturated) specific humidity over ice
            A2=-d1*ug*t-d3*ug*B &
                -emiss_ice*boltzmann*((t+tmelt)**4) ! sensible+latent heat and outward radiation
            A3=-d3*ug*B*q2/((t+tmelt)**2)           ! gradient coefficient for the latent heat part
            C=con/hice                              ! gradient coefficient for downward conductivity
            A3=A3+C+d1*ug &                         ! gradient coefficient for sensible heat+radiation
                +4.0_WP*emiss_ice*boltzmann*((t+tmelt)**3)
            C=C*(TFrez(S_oc)-t)                     ! downward conductivity term
            t=t+(A1+A2+C)/A3                        ! NEW ICE TEMPERATURE
        end do
        t=min(0.0_WP,t)

        ! heat fluxes [W/m**2]
        hfrad= (1.0_WP-alb)*fsh &               ! absorbed short wave radiation
            +flo &                              ! long wave radiation coming in
            -emiss_ice*boltzmann*((t+tmelt)**4) ! long wave radiation going out

        hfsen=d1*ug*(ta-t)                    ! sensible heat
        subli=d2*ug*(qa-B)                    ! sublimation
        hflat=clhi*subli                      ! latent heat

        hftot=hfrad+hfsen+hflat               ! total heat

        fh= -hftot/cl                         ! growth rate [m ice/sec]
        subli=subli*inv_rhowat                ! negative upward

        return
    end subroutine budget

    !___________________________________________________________________________
    ! obudget (FESOM2 ice_thermo_oce.F90:777). Open-ocean ice growth rate + evaporation.
    ! The open_water_albedo>0 solar-zenith block is DEAD here (the gated config sets
    ! open_water_albedo=0 -> albw stays the namelist constant); daynew/timenew are local
    ! placeholders for that dead branch (FESOM2 reads them from g_clock).
    subroutine obudget(ithermp, qa,fsh,flo,t,ug,ta,ch,ce,geolon, &
                       geolat, fh, evap, hflatow, hfsenow, hflwrdout, hfswrow, &
                       hflwrow, hfradow)
        type(t_ice_thermo), intent(in), target :: ithermp
        real(kind=WP) qa,t,ta,fsh,flo,ug,ch,ce,fh,evap
        real(kind=WP) hfsenow, hfswrow, hflwrow, hfradow, hflatow, hftotow, hflwrdout,b
        real(kind=WP) q1, q2            ! coefficients for saturated specific humidity
        real(kind=WP) c1, c4, c5, coszen, geolon, geolat
        logical :: standard_saturation_shum_formula = .true.
        integer :: daynew = 1           ! dead-branch placeholder (open_water_albedo=0)
        real(kind=WP) :: timenew = 0.0_WP
        !_______________________________________________________________________
        integer, pointer :: open_water_albedo
        real(kind=WP), pointer :: boltzmann, emiss_wat, inv_rhowat, inv_rhoair, rhoair, &
                                  tmelt, cl, clhw, cpair, albw
        boltzmann  => ithermp%boltzmann
        emiss_wat  => ithermp%emiss_wat
        inv_rhowat => ithermp%inv_rhowat
        inv_rhoair => ithermp%inv_rhoair
        rhoair     => ithermp%rhoair
        tmelt      => ithermp%tmelt
        cl         => ithermp%cl
        clhw       => ithermp%clhw
        cpair      => ithermp%cpair
        albw       => ithermp%albw
        open_water_albedo => ithermp%open_water_albedo

        !_______________________________________________________________________
        c1 = 3.8e-3_WP
        c4 = 17.27_WP
        c5 = 237.3_WP
        q1 = 640380._WP
        q2 = -5107.4_WP
        if(open_water_albedo > 0)then
            coszen=compute_solar_zenith_angle(daynew, timenew/3600., geolon, geolat)
            if(open_water_albedo > 1)then
               albw=albw_briegleb(coszen)
            else
               albw=albw_taylor(coszen)
            endif
         endif

        ! (saturated) surface specific humidity
        if(standard_saturation_shum_formula) then
            b=c1*exp(c4*t/(t+c5))                      ! a standard one
        else
            b=0.98_WP*q1*inv_rhoair*exp(q2/(t+tmelt))  ! LY2004 NCAR version
        end if

        ! radiation heat fluxes [W/m**2]
        hfswrow  = (1.0_WP-albw)*fsh
        hflwrow  = flo                                 ! long wave radiation coming in
        hflwrdout= -emiss_wat*boltzmann*((t+tmelt)**4) ! long wave radiation going out
        hfradow  = hfswrow + hflwrow + hflwrdout

        ! sensible heat flux [W/m**2]
        hfsenow  = rhoair*cpair*ch*ug*(ta-t)           ! sensible heat

        ! latent heat flux [W/m**2]
        evap     = rhoair*ce*ug*(qa-b)                 ! evaporation kg/m2/s
        hflatow  = clhw*evap                           ! latent heat W/m2

        ! total heat flux [W/m**2]
        hftotow  = hfradow+hfsenow+hflatow             ! total heat W/m2

        fh= -hftotow/cl                                ! growth rate [m ice/sec]
        evap=evap*inv_rhowat                           ! evaporation rate [m water/s], negative up

        return
    end subroutine obudget

    !___________________________________________________________________________
    ! flooding (FESOM2 ice_thermo_oce.F90:872). Snow-to-ice conversion (Archimedes).
    subroutine flooding(ithermp, h, hsn)
        type(t_ice_thermo), intent(in), target :: ithermp
        real(kind=WP) h,hsn,hdraft,hflood
        real(kind=WP), pointer :: inv_rhowat, inv_rhosno, rhoice, rhosno
        inv_rhowat => ithermp%inv_rhowat
        inv_rhosno => ithermp%inv_rhosno
        rhoice     => ithermp%rhoice
        rhosno     => ithermp%rhosno

        hdraft=(rhosno*hsn+h*rhoice)*inv_rhowat ! Archimedes: displaced water
        hflood=hdraft-min(hdraft,h)             ! Increase in mean ice thickness due to flooding
        h=h+hflood                              ! Add converted snow to ice volume
        hsn=hsn-hflood*rhoice*inv_rhosno        ! Subtract snow from snow layer

        return
    end subroutine flooding

    !___________________________________________________________________________
    ! TFrez (FESOM2 ice_thermo_oce.F90:899). Millero (1978) / UNESCO water freezing point.
    function TFrez(S)
        real(kind=WP) :: S, TFrez
        TFrez= -0.0575_WP*S+1.7105e-3_WP *sqrt(S**3)-2.155e-4_WP *S*S
    end function TFrez

    !___________________________________________________________________________
    ! compute_solar_zenith_angle (FESOM2 ice_thermo_oce.F90:911). Only reached when
    ! open_water_albedo>0 (dead in the gated config); ported for linkage.
    function compute_solar_zenith_angle(day_of_year, hour_utc, longitude, latitude) result(cos_zenith)
        integer, intent(in)        :: day_of_year
        real(kind=WP), intent(in)  :: hour_utc
        real(kind=WP), intent(in)  :: longitude, latitude
        real(kind=WP)              :: cos_zenith
        real(kind=WP), parameter   :: PI = 3.141592653589793_WP
        real(kind=WP), parameter   :: DEG_TO_RAD = PI / 180.0_WP
        real(kind=WP), parameter   :: DAYS_PER_YEAR = 365.25_WP
        real(kind=WP)              :: solar_declination, hour_angle
        real(kind=WP)              :: solar_fraction

        solar_fraction = 2.0_WP * PI * day_of_year / DAYS_PER_YEAR
        solar_declination = 0.006918_WP - 0.399912_WP * cos(solar_fraction) &
                          + 0.070257_WP * sin(solar_fraction) &
                          - 0.006758_WP * cos(2.0_WP * solar_fraction) &
                          + 0.000907_WP * sin(2.0_WP * solar_fraction) &
                          - 0.002697_WP * cos(3.0_WP * solar_fraction) &
                          + 0.001480_WP * sin(3.0_WP * solar_fraction)
        hour_angle = (hour_utc - 12.0_WP) * 15.0_WP * DEG_TO_RAD + longitude
        cos_zenith = sin(latitude) * sin(solar_declination) &
                   + cos(latitude) * cos(solar_declination) * cos(hour_angle)
        if (cos_zenith < 0.0_WP) cos_zenith = 0.0_WP
        return
    end function compute_solar_zenith_angle

    !___________________________________________________________________________
    ! albw_taylor (FESOM2 ice_thermo_oce.F90:979). Taylor et al. (1996) open-water albedo.
    function albw_taylor(coszen)
        real(kind=WP), intent(in)  :: coszen
        real(kind=WP) :: albw_taylor
        albw_taylor = 0.037_WP/(1.1_WP * coszen**1.4_WP + 0.15_WP)
        return
    end function albw_taylor

    !___________________________________________________________________________
    ! albw_briegleb (FESOM2 ice_thermo_oce.F90:995). Briegleb et al. (1986) albedo.
    function albw_briegleb(coszen)
        real(kind=WP), intent(in)  :: coszen
        real(kind=WP)              :: albw_briegleb
        albw_briegleb = 0.026_WP/(1.1_WP * coszen**1.7_WP + 0.065_WP) + 0.15_WP * (coszen-1._WP)**2 * (coszen-0.5_WP)
        return
    end function albw_briegleb

end module mod_ice_thermo
