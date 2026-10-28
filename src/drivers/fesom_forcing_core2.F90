program fesom_forcing_core2
    ! M3f-3a CORE2 atmospheric-forcing READ byte-gate. Drives the M2.10a mod_forcing_read
    ! (netCDF + spatial bilinear + linear time-interp + g2r wind rotation — already byte-proven
    ! on pi against these EXACT CORE2 NCAR stub files) on the CORE2 1-rank mesh, advancing the
    ! per-step rdate exactly as FESOM2 gen_surface_forcing::sbc_do (g_clock), and SELF-CHECKS
    ! the 8 raw atm fields (shortwave/longwave/Tair/shum/prec_rain/prec_snow/u_wind/v_wind) per
    ! step against the M3f-2 oracle atm-forcing dump (FESOM_ATMFLUX_DUMP -> FESOM3_ATMFLUX_FILE).
    ! No new oracle: the dump already holds the oracle's post-read atm forcing every step.
    !
    ! Clock/rdate (g_clock + sbc_do:1527): fesom.clock '0 1 1948' => timenew=0,daynew=1,
    ! yearnew=1948; clock() at the TOP of each step adds dt (1800 s), so at step n
    ! timenew=n*dt, daynew=1 (n<48). cold-start getcoeffld uses rdate_cold (NO half-step,
    ! nc_sbc_ini:643); per-step timeinterp uses rdate(n)=julday(1948,1,1,noleap)+(daynew-1)
    ! +timenew/86400 - dt/2/86400 (WITH the -dt/2 half-step, sbc_do:1528). For a short run
    ! (n<48, within forcing day 1) NO getcoeffld re-trigger fires (rdate stays below
    ! nc_time(t_indx_p1)), so the cold-start coefficients apply every step — byte-faithful;
    ! the multi-day re-trigger (sbc_do:1561) is an M3f-3 (longer-run) deferral.
    !
    !   FESOM3_MESH_DIR     mesh dir   (default CORE2)
    !   FESOM3_FORCING_DIR  NCAR stubs (default port2 test/input/global)
    !   FESOM3_ATMFLUX_FILE oracle atm-forcing dump (REQUIRED; from fesom_atmflux_dump)
    !   FESOM3_NSTEPS       steps      (default 3)
    use mpi
    use, intrinsic :: iso_fortran_env, only: int32, real64
    use mod_precision,    only: WP, WP_full
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init, par_ex
    use mod_mesh_read,    only: read_mesh
    use mod_mesh_areas,   only: compute_geometry
    use mod_forcing_read
    implicit none

    real(kind=WP), parameter :: dt = 86400.0_WP / real(48, WP)   ! CORE2 dt = 1800 s

    character(len=512) :: mesh_dir, forcing_dir, atm_file, env
    type(t_partit) :: partit
    type(t_mesh)   :: mesh
    type(t_atm_forcing) :: frc
    integer :: nsw, fld, nn, year, n, nsteps, ios, env_len, atm_unit
    real(WP) :: timenew
    real(WP_full) :: rdate_cold, rdate
    real(WP), allocatable :: u_wind(:), v_wind(:), Tair(:), shum(:), &
                             shortwave(:), longwave(:), prec_rain(:), prec_snow(:)
    real(WP), allocatable :: rd(:)
    integer(int32) :: s, fnn, fne
    real(real64) :: dmax(8)

    call get_environment_variable('FESOM3_MESH_DIR', mesh_dir)
    if (len_trim(mesh_dir) == 0) &
        mesh_dir = '/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2'
    call get_environment_variable('FESOM3_FORCING_DIR', forcing_dir)
    if (len_trim(forcing_dir) == 0) &
        forcing_dir = '/home/a/a270088/port2/fesom2/test/input/global'
    call get_environment_variable('FESOM3_ATMFLUX_FILE', atm_file)
    if (len_trim(atm_file) == 0) then
        write(*,'(a)') 'fesom_forcing_core2: FESOM3_ATMFLUX_FILE is REQUIRED'; error stop 1
    end if
    nsteps = 3
    call get_environment_variable('FESOM3_NSTEPS', env, length=env_len, status=ios)
    if (ios == 0 .and. env_len > 0) read(env, *, iostat=ios) nsteps

    call par_init(partit)
    if (partit%npes /= 1) then
        if (partit%mype == 0) write(*,'(a)') 'fesom_forcing_core2: requires 1 rank'
        call par_ex(partit%MPI_COMM_FESOM, partit%mype); error stop 1
    end if

    call read_mesh(mesh, partit, trim(mesh_dir), 50.0_WP, 15.0_WP, -90.0_WP, 360.0_WP, &
                   force_rotation=.true., n_cw_swaps=nsw)
    call compute_geometry(mesh, partit, cartesian=.false.)
    nn = mesh%nod2D
    write(*,'(a,i0,a,i0)') 'fesom_forcing_core2: nod2D=', mesh%nod2D, ' CW swaps=', nsw

    !--- forcing config (CORE2 namelist.forcing &nam_sbc, the 8 NCAR fields) ------
    frc%nfld = 8;  frc%nnod = nn
    frc%iyear = 1948; frc%imm = 1; frc%idd = 1; frc%freq = 1; frc%tmid = 1
    frc%ic_cyclic = .true.; frc%rotated_grid = .true.
    frc%i_xwind = 1; frc%i_ywind = 2
    frc%f(1)%file_base = trim(forcing_dir)//'/u_10.';        frc%f(1)%varname = 'U_10_MOD'
    frc%f(2)%file_base = trim(forcing_dir)//'/v_10.';        frc%f(2)%varname = 'V_10_MOD'
    frc%f(3)%file_base = trim(forcing_dir)//'/q_10.';        frc%f(3)%varname = 'Q_10_MOD'
    frc%f(4)%file_base = trim(forcing_dir)//'/ncar_rad.';    frc%f(4)%varname = 'SWDN_MOD'
    frc%f(5)%file_base = trim(forcing_dir)//'/ncar_rad.';    frc%f(5)%varname = 'LWDN_MOD'
    frc%f(6)%file_base = trim(forcing_dir)//'/t_10.';        frc%f(6)%varname = 'T_10_MOD'
    frc%f(7)%file_base = trim(forcing_dir)//'/ncar_precip.'; frc%f(7)%varname = 'RAIN'
    frc%f(8)%file_base = trim(forcing_dir)//'/ncar_precip.'; frc%f(8)%varname = 'SNOW'

    year = 1948
    rdate_cold = real(forcing_julday(year,1,1,'noleap'),WP)   ! daynew=1, timenew=0, NO half-step

    call forcing_alloc(frc)
    do fld = 1, frc%nfld
        call forcing_read_grid(frc, fld, year)
        call forcing_build_bilin(frc, fld, mesh, partit)
    end do
    do fld = 1, frc%nfld
        call forcing_getcoeffld(frc, fld, year, rdate_cold, mesh, partit)
    end do
    call forcing_rotate_wind(frc, mesh, partit)

    allocate(u_wind(nn), v_wind(nn), Tair(nn), shum(nn), shortwave(nn), &
             longwave(nn), prec_rain(nn), prec_snow(nn), rd(nn))

    open(newunit=atm_unit, file=trim(atm_file), status='old', form='unformatted', &
         access='stream', action='read')
    write(*,'(a,a)') 'fesom_forcing_core2: self-check vs ', trim(atm_file)

    do n = 1, nsteps
        timenew = real(n,WP)*dt                       ! daynew stays 1 for n<48
        rdate   = real(forcing_julday(year,1,1,'noleap'),WP) &
                + real(1-1,WP) + timenew/86400._WP - dt/86400._WP/2._WP
        call forcing_timeinterp(frc, rdate, partit)
        u_wind    = frc%atmdata(1, 1:nn)
        v_wind    = frc%atmdata(2, 1:nn)
        shum      = frc%atmdata(3, 1:nn)
        shortwave = frc%atmdata(4, 1:nn)
        longwave  = frc%atmdata(5, 1:nn)
        Tair      = frc%atmdata(6, 1:nn) - 273.15_WP
        prec_rain = frc%atmdata(7, 1:nn) / 1000._WP
        prec_snow = frc%atmdata(8, 1:nn) / 1000._WP

        ! oracle dump record order (fesom_atmflux_dump): shortwave,longwave,Tair,shum,
        ! prec_rain,prec_snow,runoff,u_wind,v_wind,Ch,Ce,satmoce_x,satmoce_y,satmice_x,
        ! satmice_y,Ssurf. Compare the 8 we compute; skip runoff + the 6 bulk/clim arrays.
        read(atm_unit) s, fnn, fne
        call cmp(shortwave, dmax(1))    ! shortwave
        call cmp(longwave,  dmax(2))    ! longwave
        call cmp(Tair,      dmax(3))    ! Tair
        call cmp(shum,      dmax(4))    ! shum
        call cmp(prec_rain, dmax(5))    ! prec_rain
        call cmp(prec_snow, dmax(6))    ! prec_snow
        read(atm_unit) rd               ! runoff (skip)
        call cmp(u_wind,    dmax(7))    ! u_wind
        call cmp(v_wind,    dmax(8))    ! v_wind
        read(atm_unit) rd; read(atm_unit) rd   ! Ch, Ce (skip)
        read(atm_unit) rd; read(atm_unit) rd   ! stress_atmoce_x/y (skip)
        read(atm_unit) rd; read(atm_unit) rd   ! stress_atmice_x/y (skip)
        read(atm_unit) rd                       ! Ssurf (skip)
        write(*,'(a,i0,a,8es11.3)') 'fesom_forcing_core2: [step ', n, &
            '] max|d(sw,lw,Tair,shum,rain,snow,uw,vw)| = ', dmax
    end do
    close(atm_unit)

    write(*,'(a)') 'fesom_forcing_core2: done.'
    call par_ex(partit%MPI_COMM_FESOM, partit%mype)

contains

    ! read the next dumped nod2D array and set d = max|native - oracle|.
    subroutine cmp(native, d)
        real(WP),     intent(in)  :: native(:)
        real(real64), intent(out) :: d
        read(atm_unit) rd
        d = maxval(abs(native - rd))
    end subroutine cmp

end program fesom_forcing_core2
