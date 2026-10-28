program fesom_forcingdump
    ! M2.10a forcing-read byte-gate driver. Loads pi 1-rank (same rotation as the
    ! FESOM2 oracle), reads the 8 CORE2 atmospheric forcing fields via
    ! mod_forcing_read (netCDF + spatial bilinear + linear time interp + g2r wind
    ! rotation), maps atmdata -> the physical node arrays exactly as FESOM2
    ! gen_forcing_couple::update_atm_forcing, and dumps them (FADVHDMP). The FESOM2
    ! oracle (port2/fesom2/src/fesom_forcing_dump.F90, via tools/run_forcing_gate.sh)
    ! drives the REAL sbc_do; tools/pressure_diff.py compares for max|delta|=0.
    !
    ! Time pinning (shared with the oracle shim): the coefficients are built at the
    ! COLD-START rdate (clock-init time: day 1, sec 0, NO half-step), the data is
    ! evaluated at the per-step rdate (day 1, sec 43200, dt 2400, WITH -dt/2 shift).
    ! Both on the noleap Julian scale (julday(1948,1,1,noleap)=365*1948=710820).
    use mpi
    use, intrinsic :: iso_fortran_env, only: real64
    use mod_precision,    only: WP, MP, WP_full
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init, par_ex
    use mod_mesh_read,    only: read_mesh
    use mod_mesh_areas,   only: compute_geometry
    use mod_forcing_read
    use mod_forcing_bulk, only: forcing_bulk_ncar, forcing_wind_stress, forcing_stress_surf
    use oce_shortwave_pene, only: cal_shortwave_rad
    use mod_advhor_dump,  only: advhor_dump_open, advhor_dump_close, wr_r1, wr_r2
    implicit none
    character(len=*), parameter :: DPATH = '/home/a/a270088/port2/fesom2/test/input/global/'
    character(len=512) :: mesh_dir, out_path
    type(t_partit) :: partit
    type(t_mesh)   :: mesh
    type(t_atm_forcing) :: frc
    integer :: nsw, u, fld, nn, year
    real(WP) :: dt
    real(WP_full) :: rdate_cold, rdate_eval
    real(WP), allocatable :: u_wind(:), v_wind(:), Tair(:), shum(:), &
                             shortwave(:), longwave(:), prec_rain(:), prec_snow(:)
    real(WP), allocatable :: sst(:), u_w(:), v_w(:), cd_oce(:), ch_oce(:), ce_oce(:), &
                             stress_ax(:), stress_ay(:), stress_surf(:,:)
    real(WP), allocatable :: chl(:), heat_flux_sw(:), a_ice(:), sw_3d(:,:)
    real(WP) :: lon, lat, zbar_srf, zbar_bot
    integer :: nzmin, nzmax

    call get_environment_variable('FESOM3_MESH_DIR', mesh_dir)
    if (len_trim(mesh_dir) == 0) &
        mesh_dir = '/home/a/a270088/port2/fesom2/tests/data/MESHES/pi'
    call get_environment_variable('FESOM3_FORCING_OUT', out_path)
    if (len_trim(out_path) == 0) out_path = 'forcing_f3.bin'

    call par_init(partit)
    if (partit%npes /= 1) then
        if (partit%mype == 0) write(*,'(a)') 'fesom_forcingdump: requires 1 rank'
        call par_ex(partit%MPI_COMM_FESOM, partit%mype); error stop 1
    end if

    call read_mesh(mesh, partit, trim(mesh_dir), 50.0_WP, 15.0_WP, -90.0_WP, 360.0_WP, &
                   force_rotation=.true., n_cw_swaps=nsw)
    call compute_geometry(mesh, partit, cartesian=.false.)
    ! 1-rank gate: kernels loop over mesh%nod2D (== myDim+eDim at 1-rank; the dump
    ! drivers do not call set_partition, so partit%myDim_nod2D stays 0 — see L25).
    nn = mesh%nod2D

    !--- forcing config (pi work_pi/namelist.forcing &nam_sbc) -------------------
    frc%nfld = 8;  frc%nnod = nn
    frc%iyear = 1948; frc%imm = 1; frc%idd = 1; frc%freq = 1; frc%tmid = 1
    frc%ic_cyclic = .true.; frc%rotated_grid = .true.
    frc%i_xwind = 1; frc%i_ywind = 2
    frc%f(1)%file_base = DPATH//'u_10.';        frc%f(1)%varname = 'U_10_MOD'
    frc%f(2)%file_base = DPATH//'v_10.';        frc%f(2)%varname = 'V_10_MOD'
    frc%f(3)%file_base = DPATH//'q_10.';        frc%f(3)%varname = 'Q_10_MOD'
    frc%f(4)%file_base = DPATH//'ncar_rad.';    frc%f(4)%varname = 'SWDN_MOD'
    frc%f(5)%file_base = DPATH//'ncar_rad.';    frc%f(5)%varname = 'LWDN_MOD'
    frc%f(6)%file_base = DPATH//'t_10.';        frc%f(6)%varname = 'T_10_MOD'
    frc%f(7)%file_base = DPATH//'ncar_precip.'; frc%f(7)%varname = 'RAIN'
    frc%f(8)%file_base = DPATH//'ncar_precip.'; frc%f(8)%varname = 'SNOW'

    year = 1948; dt = 2400.0_WP
    ! cold-start rdate (nc_sbc_ini:643-644 — NO half-step; clock-init day 1, sec 0)
    rdate_cold = real(forcing_julday(year,1,1,'noleap'),WP) + real(1-1,WP) + 0.0_WP/86400._WP
    ! per-step rdate (sbc_do:1527-1528 — half-step; pinned day 1, sec 43200, dt 2400)
    rdate_eval = real(forcing_julday(year,1,1,'noleap'),WP) + real(1-1,WP) &
               + 43200._WP/86400._WP - dt/86400._WP/2._WP

    call forcing_alloc(frc)
    do fld = 1, frc%nfld
        call forcing_read_grid(frc, fld, year)
        call forcing_build_bilin(frc, fld, mesh, partit)
    end do
    do fld = 1, frc%nfld
        call forcing_getcoeffld(frc, fld, year, rdate_cold, mesh, partit)
    end do
    call forcing_rotate_wind(frc, mesh, partit)
    call forcing_timeinterp(frc, rdate_eval, partit)

    !--- map atmdata -> physical arrays (update_atm_forcing 681-694) -------------
    allocate(u_wind(nn), v_wind(nn), Tair(nn), shum(nn), shortwave(nn), &
             longwave(nn), prec_rain(nn), prec_snow(nn))
    u_wind    = frc%atmdata(1, 1:nn)
    v_wind    = frc%atmdata(2, 1:nn)
    shum      = frc%atmdata(3, 1:nn)
    shortwave = frc%atmdata(4, 1:nn)
    longwave  = frc%atmdata(5, 1:nn)
    Tair      = frc%atmdata(6, 1:nn) - 273.15_WP
    prec_rain = frc%atmdata(7, 1:nn) / 1000._WP
    prec_snow = frc%atmdata(8, 1:nn) / 1000._WP

    !--- M2.10b: prescribe SST + surface ocean velocity (identical analytic formula
    !    shared with the oracle shim; the M2.5/M2.8 prescribe-the-unsourced-input
    !    pattern), then bulk transfer coeffs -> wind stress -> node->elem stress_surf.
    allocate(sst(nn), u_w(nn), v_w(nn), cd_oce(nn), ch_oce(nn), ce_oce(nn), &
             stress_ax(nn), stress_ay(nn), stress_surf(2, mesh%elem2D))
    do fld = 1, nn
        lon = mesh%geo_coord_nod2D(1, fld)
        lat = mesh%geo_coord_nod2D(2, fld)
        sst(fld) = -1.0_WP + 20.0_WP*cos(lat)
        u_w(fld) =  0.15_WP*sin(lon)*cos(lat)
        v_w(fld) = -0.10_WP*cos(lon)*cos(lat)
    end do
    call forcing_bulk_ncar(10.0_WP, 10.0_WP, 10.0_WP, Tair, shum, u_wind, v_wind, &
                           sst, u_w, v_w, cd_oce, ch_oce, ce_oce, mesh)
    call forcing_wind_stress(0.0_WP, u_wind, v_wind, u_w, v_w, cd_oce, stress_ax, stress_ay, mesh)
    call forcing_stress_surf(stress_ax, stress_ay, stress_surf, mesh)

    write(*,'(a,i0,a,i0)') 'fesom_forcingdump: nod2D=', mesh%nod2D, ' nn=', nn
    write(*,'(a,3f12.6)') '  node 1 u_wind/Tair/shortwave = ', u_wind(1), Tair(1), shortwave(1)
    write(*,'(a,3es13.5)') '  node 1 Cd/Ch/Ce = ', cd_oce(1), ch_oce(1), ce_oce(1)
    write(*,'(a,3es13.5)') '  node 1 stress_atmoce_x/y, elem1 stress_surf_x = ', &
                           stress_ax(1), stress_ay(1), stress_surf(1,1)

    !--- M2.10c: per-node ALE depths zbar_3d_n (init_ale, pressure-gate-proven) +
    !    prescribed chl/input heat_flux/a_ice=0, then shortwave penetration -> sw_3d
    !    + the heat_flux visible-removal. shortwave is the LIVE M2.10a read output.
    allocate(mesh%zbar_3d_n(mesh%nl, mesh%nod2D)); mesh%zbar_3d_n = 0.0_MP
    do fld = 1, mesh%nod2D
        nzmin = mesh%ulevels_nod2D(fld); nzmax = mesh%nlevels_nod2D(fld)
        zbar_srf = mesh%zbar(nzmin); zbar_bot = mesh%zbar(nzmax)
        mesh%zbar_3d_n(1:nzmin-1, fld)       = mesh%zbar(1:nzmin-1)
        mesh%zbar_3d_n(nzmin, fld)           = zbar_srf
        mesh%zbar_3d_n(nzmin+1:nzmax-1, fld) = mesh%zbar(nzmin+1:nzmax-1)
        mesh%zbar_3d_n(nzmax, fld)           = zbar_bot
    end do
    allocate(chl(nn), heat_flux_sw(nn), a_ice(nn), sw_3d(mesh%nl, mesh%nod2D))
    do fld = 1, nn
        lon = mesh%geo_coord_nod2D(1, fld); lat = mesh%geo_coord_nod2D(2, fld)
        chl(fld)          = 0.3_WP*cos(lat)*cos(lat)   ! 0..0.3; <0.02 near poles -> floor fires
        heat_flux_sw(fld) = 30.0_WP*cos(lat)*sin(lon)  ! input non-solar heat flux
        a_ice(fld)        = 0.0_WP
    end do
    call cal_shortwave_rad(.true., 0.066_WP, shortwave, chl, a_ice, heat_flux_sw, sw_3d, mesh)
    write(*,'(a,2es13.5,a,es13.5)') '  node 1 chl(floored)/heat_flux_sw = ', chl(1), &
                           heat_flux_sw(1), '  max|sw_3d|=', maxval(abs(sw_3d))

    call advhor_dump_open(u, trim(out_path), mesh%nod2D, mesh%elem2D, mesh%edge2D, mesh%nl)
    call wr_r1(u, 'u_wind',    u_wind(1:mesh%nod2D))
    call wr_r1(u, 'v_wind',    v_wind(1:mesh%nod2D))
    call wr_r1(u, 'Tair',      Tair(1:mesh%nod2D))
    call wr_r1(u, 'shum',      shum(1:mesh%nod2D))
    call wr_r1(u, 'shortwave', shortwave(1:mesh%nod2D))
    call wr_r1(u, 'longwave',  longwave(1:mesh%nod2D))
    call wr_r1(u, 'prec_rain', prec_rain(1:mesh%nod2D))
    call wr_r1(u, 'prec_snow', prec_snow(1:mesh%nod2D))
    ! M2.10b bulk: prescribed SST/surf-vel + Cd/Ch/Ce + wind stress + stress_surf
    call wr_r1(u, 'sst',             sst(1:mesh%nod2D))
    call wr_r1(u, 'srfoce_u',        u_w(1:mesh%nod2D))
    call wr_r1(u, 'srfoce_v',        v_w(1:mesh%nod2D))
    call wr_r1(u, 'cd_atm_oce',      cd_oce(1:mesh%nod2D))
    call wr_r1(u, 'ch_atm_oce',      ch_oce(1:mesh%nod2D))
    call wr_r1(u, 'ce_atm_oce',      ce_oce(1:mesh%nod2D))
    call wr_r1(u, 'stress_atmoce_x', stress_ax(1:mesh%nod2D))
    call wr_r1(u, 'stress_atmoce_y', stress_ay(1:mesh%nod2D))
    call wr_r2(u, 'stress_surf',     stress_surf(1:2, 1:mesh%elem2D))
    ! M2.10c shortwave penetration: floored chl + modified heat_flux + sw_3d profile
    call wr_r1(u, 'chl',          chl(1:mesh%nod2D))
    call wr_r1(u, 'heat_flux_sw', heat_flux_sw(1:mesh%nod2D))
    call wr_r2(u, 'sw_3d',        sw_3d(1:mesh%nl, 1:mesh%nod2D))
    call advhor_dump_close(u)
    write(*,'(a,a)') 'fesom_forcingdump: wrote ', trim(out_path)

    call par_ex(partit%MPI_COMM_FESOM, partit%mype)
end program fesom_forcingdump
