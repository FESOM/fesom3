module mod_forcing_read
    ! Surface-forcing READ + space/time interpolation (M2.10a). Transcribed from
    ! FESOM2 v2.7.3 gen_surface_forcing.F90 (MODULE g_sbf):
    !   julday              (1850-1889)  Numerical-Recipes Julian Day Number
    !   calendar_date       (1892-1935)  inverse (only the year is used)
    !   binarysearch        (1962-2006)  bisection -> largest index with arr(i)<=val
    !   nc_readTimeGrid     ( 223- 538)  read lon/lat/time axes; time-axis transform
    !                                     to days-since-0001; periodic-lon halo; lat flip
    !   nc_sbc_ini          (1047-1468)  build per-node bilinear indices; cold-start
    !                                     getcoeffld; g2r wind-coef rotation
    !   getcoeffld          ( 724-1020)  read 2 bracketing real(4) slices; spatial
    !                                     bilinear -> data1/data2; coef_a/coef_b
    !   data_timeinterp     (1022-1045)  atmdata = rdate*coef_a + coef_b
    !
    ! BIT-IDENTITY NOTES:
    ! * The time interp is kept as the EXACT two-stage form coef_a=(data2-data1)/dt;
    !   coef_b=data1-coef_a*nc_time(t_indx); atmdata=rdate*coef_a+coef_b. nc_time and
    !   rdate live on the Julian-Day scale (~710820 for the pi CORE2 noleap year 1948
    !   = 365*1948; ~2.43e6 for JRA gregorian), so rdate*coef_a and coef_b nearly
    !   cancel — NEVER refactor to data1+coef_a*(rdate-nc_time) (the WP rounding of
    !   the large coef_b differs). FESOM2 getcoeffld:1015-1016 + data_timeinterp:1041.
    ! * The coefficients are built at the COLD-START rdate (no half-step shift,
    !   nc_sbc_ini:643-644) and evaluated at the per-step rdate (with the -dt/2
    !   half-step shift, sbc_do:1527-1528). Both passed in by the caller.
    ! * The raw slices stay real(4) (on-disk type); they promote to WP inside the
    !   bilinear weight expression exactly as FESOM2 (real(4) sbcdata * real(WP) wgt).
    ! * julday for 'noleap'/'none'/'365_days' = 365*yyyy (the else branch, :1887).
    ! * The g2r wind rotation acts on the interpolation COEFFICIENTS (coef_a, coef_b)
    !   separately — rotating the two linear coefficients then evaluating a*t+b equals
    !   rotating the evaluated vector (nc_sbc_ini:703-707).
    !
    ! SCOPE (M2.10a): the READ only — produces atmdata for the enabled fields. The
    ! bulk transfer coefficients + wind stress (M2.10b) and SW penetration (M2.10c)
    ! consume these; heat/water flux air-sea budget is M3 (ice thermo). The async
    ! prefetch reader stack + the SSS/runoff/chl climatology reads are NOT ported
    ! (perf / separate fields).
    use, intrinsic :: iso_fortran_env, only: real32, real64
    use mod_precision, only: WP
    use mod_constants,  only: rad
    use mod_io_netcdf
    use mod_mesh_rotate, only: vector_g2r
    use mod_mesh,   only: t_mesh
    use mod_partit, only: t_partit
    use mod_timer,  only: timer_start, timer_stop, TMR_FRC_OPEN, TMR_FRC_READ, TMR_FRC_INTRP2
    implicit none
    private

    ! PERF toggle (FESOM3_FORCING_PERSIST, default ON). ON  = FESOM2-style persistent handle +
    ! double-buffer (read one new slice/crossing). OFF = legacy open + read both + close every
    ! crossing (kept for A/B measurement and as a safety valve). Read once, lazily, in getcoeffld.
    logical, save :: frc_use_cache = .true.
    logical, save :: frc_cache_init = .false.
    public :: t_atm_forcing, forcing_alloc, forcing_read_grid, forcing_build_bilin, &
              forcing_getcoeffld, forcing_rotate_wind, forcing_timeinterp, &
              forcing_julday, forcing_binarysearch, forcing_sbc_do

    integer, parameter, public :: FRC_MAXFLD = 16

    type :: t_ffile
        character(len=512) :: file_base = ''   ! e.g. '<path>/u_10.'  (year + '.nc' appended)
        character(len=64)  :: varname  = ''    ! e.g. 'U_10_MOD'
        character(len=64)  :: calendar = 'none'
        integer            :: nlon = 0, nlat = 0, ntime = 0   ! nlon INCLUDES the +2 lon halo
        integer            :: year_orig = 0
        real(WP), allocatable :: nc_lon(:), nc_lat(:), nc_time(:)
        integer            :: flip_lat = 0
        ! M8b: persisted time bracket from the previous forcing_getcoeffld (FESOM2 flfi_type
        ! t_indx/t_indx_p1). forcing_sbc_do's crossing test reads them; nc_Ntime == ntime.
        integer            :: t_indx = 1, t_indx_p1 = 1
        ! PERF (2026-06-28): the JRA55-do files are netCDF-4/HDF5, DEFLATE-compressed, chunked one
        ! (lat,lon) slice per chunk. So every per-crossing nc_open re-parses HDF5 metadata and every
        ! slice read re-inflates ~800 KB on EVERY rank. FESOM3 used to nc_open + read BOTH brackets +
        ! nc_close every crossing on every rank — measured ~172 ms/crossing (microbench). FESOM2's
        ! getcoeffld instead keeps the file open and double-buffers (reads only the ONE genuinely-new
        ! slice). We mirror that here:
        !   ncid/ncid_year: persistent read handle, reopened only when the year (=> filename) changes.
        !   sbc_a/sbc_b + ia_indx/ib_indx: two slice buffers tagged with the time index they hold; on a
        !     crossing the buffer already holding t_indx is reused (slot1), only t_indx_p1 is read fresh
        !     (slot2). Byte-exact: the reused bytes are exactly what a re-read would deserialize.
        integer            :: ncid = -1, ncid_year = -1
        real(real32), allocatable :: sbc_a(:,:), sbc_b(:,:)
        integer            :: ia_indx = -1, ib_indx = -1
    end type t_ffile

    type :: t_atm_forcing
        integer :: nfld   = 0
        integer :: nnod   = 0            ! myDim_nod2D + eDim_nod2D
        integer :: iyear  = 1948, imm = 1, idd = 1   ! time-axis reference (nm_nc_*)
        integer :: freq   = 1            ! nm_nc_freq
        integer :: tmid   = 1            ! nm_nc_tmid
        logical :: ic_cyclic = .true.
        logical :: rotated_grid = .true.
        integer :: i_xwind = 0, i_ywind = 0   ! wind-pair field indices for g2r (0 = none)
        type(t_ffile)         :: f(FRC_MAXFLD)
        integer,  allocatable :: idx_i(:,:), idx_j(:,:)   ! (nfld, nnod) bilinear indices
        real(WP), allocatable :: coef_a(:,:), coef_b(:,:) ! (nfld, nnod) time-interp coeffs
        real(WP), allocatable :: atmdata(:,:)             ! (nfld, nnod) interpolated data
    end type t_atm_forcing

contains

    ! ---- Numerical-Recipes Julian Day (FESOM2 julday, 1850-1889) -----------------
    integer function forcing_julday(yyyy, mm, dd, calendar) result(jd)
        integer, intent(in) :: yyyy, mm, dd
        character(len=*), intent(in) :: calendar
        integer, parameter :: IGREG = 15 + 31*(10 + 12*1582)
        integer :: ja, jm, jy
        if ((trim(calendar) == 'julian')              .or. &
            (trim(calendar) == 'gregorian')           .or. &
            (trim(calendar) == 'proleptic_gregorian') .or. &
            (trim(calendar) == 'standard')) then
            jy = yyyy
            if (jy == 0) error stop 'julday: there is no year zero'
            if (jy < 0) jy = jy + 1
            if (mm > 2) then
                jm = mm + 1
            else
                jy = jy - 1
                jm = mm + 13
            end if
            jd = int(365.25_WP*jy) + int(30.6001_WP*jm) + dd + 1720995
            if (dd + 31*(mm + 12*yyyy) >= IGREG) then
                ja = int(0.01_WP*jy)
                jd = jd + 2 - ja + int(0.25_WP*ja)
            end if
        else
            jd = 365*yyyy
        end if
    end function forcing_julday

    ! Inverse: only the YEAR is consumed (FESOM2 calendar_date, 1892-1935).
    integer function forcing_calyear(julian, calendar) result(yyyy)
        integer, intent(in) :: julian
        character(len=*), intent(in) :: calendar
        integer, parameter :: IGREG = 2299161
        integer :: ja, jb, jc, jd, je, mm
        real(WP) :: x
        if ((trim(calendar) == 'julian')              .or. &
            (trim(calendar) == 'gregorian')           .or. &
            (trim(calendar) == 'proleptic_gregorian') .or. &
            (trim(calendar) == 'standard')) then
            if (julian >= IGREG) then
                x = ((julian - 1867216) - 0.25_WP)/36524.25_WP
                ja = julian + 1 + int(x) - int(0.25*x)
            else
                ja = julian
            end if
            jb = ja + 1524
            jc = int(6680 + ((jb - 2439870) - 122.1_WP)/365.25_WP)
            jd = int(365*jc + (0.25_WP*jc))
            je = int((jb - jd)/30.6001_WP)
            mm = je - 1
            if (mm > 12) mm = mm - 12
            yyyy = jc - 4715
            if (mm > 2) yyyy = yyyy - 1
            if (yyyy <= 0) yyyy = yyyy - 1
        else
            yyyy = int((real(julian) + 1.e-12_WP)/365._WP)
        end if
    end function forcing_calyear

    ! ---- bisection: largest index with array(ind) <= value (FESOM2, 1962-2006) ---
    subroutine forcing_binarysearch(length, array, value, ind)
        integer,  intent(in)  :: length
        real(WP), intent(in)  :: array(length)
        real(WP), intent(in)  :: value
        integer,  intent(out) :: ind
        integer :: left, middle, right
        real(WP), parameter :: d = 1e-9_WP
        left = 1
        right = length
        do
            if (left > right) exit
            middle = nint((left + right)/2.0_WP)
            if (abs(array(middle) - value) <= d) then
                ind = middle
                return
            else if (array(middle) > value) then
                right = middle - 1
            else
                left = middle + 1
            end if
        end do
        ind = right
    end subroutine forcing_binarysearch

    subroutine forcing_alloc(frc)
        type(t_atm_forcing), intent(inout) :: frc
        allocate(frc%idx_i(frc%nfld, frc%nnod), frc%idx_j(frc%nfld, frc%nnod))
        allocate(frc%coef_a(frc%nfld, frc%nnod), frc%coef_b(frc%nfld, frc%nnod))
        allocate(frc%atmdata(frc%nfld, frc%nnod))
    end subroutine forcing_alloc

    ! ---- read lon/lat/time axes + transform time axis (FESOM2 nc_readTimeGrid) ----
    subroutine forcing_read_grid(frc, fld, year)
        type(t_atm_forcing), intent(inout) :: frc
        integer, intent(in) :: fld, year
        character(len=512) :: fname
        character(len=4) :: cyear
        integer :: ncid, nlon0, i, yyyy
        real(real64), allocatable :: tmp(:)
        write(cyear, '(I4)') year
        fname = trim(frc%f(fld)%file_base)//trim(adjustl(cyear))//'.nc'
        ncid = nc_open_read(fname)
        nlon0 = nc_dimlen(ncid, ['LON ','lon '])              ! file lon count (no halo)
        frc%f(fld)%nlat  = nc_dimlen(ncid, ['LAT ','lat '])
        frc%f(fld)%ntime = nc_dimlen(ncid, ['TIME','time'])
        frc%f(fld)%nlon  = nlon0 + 2                            ! +2 periodic-lon halo (353)
        ! M8c year-file rollover (forcing_sbc_do) re-reads the grid for the NEXT year, so the axis
        ! arrays may already be allocated (and ntime can differ across a leap boundary: 2920 vs 2928);
        ! free them first. Cold start: not allocated => the guards are no-ops.
        if (allocated(frc%f(fld)%nc_lon))  deallocate(frc%f(fld)%nc_lon)
        if (allocated(frc%f(fld)%nc_lat))  deallocate(frc%f(fld)%nc_lat)
        if (allocated(frc%f(fld)%nc_time)) deallocate(frc%f(fld)%nc_time)
        allocate(frc%f(fld)%nc_lon(frc%f(fld)%nlon), frc%f(fld)%nc_lat(frc%f(fld)%nlat), &
                 frc%f(fld)%nc_time(frc%f(fld)%ntime))
        ! lat (no halo)
        call nc_get_axis_dp(ncid, ['LAT ','lat '], frc%f(fld)%nc_lat)
        ! lon into interior 2:nlon-1, then periodic halo (377-379)
        allocate(tmp(nlon0))
        call nc_get_axis_dp(ncid, ['LON ','lon '], tmp)
        frc%f(fld)%nc_lon(2:frc%f(fld)%nlon-1) = tmp
        frc%f(fld)%nc_lon(1)            = frc%f(fld)%nc_lon(frc%f(fld)%nlon-1)
        frc%f(fld)%nc_lon(frc%f(fld)%nlon) = frc%f(fld)%nc_lon(2)
        deallocate(tmp)
        ! time axis + calendar
        call nc_get_axis_dp(ncid, ['TIME','time'], frc%f(fld)%nc_time)
        frc%f(fld)%calendar = lc(nc_get_att_text(ncid, ['TIME','time'], 'calendar'))
        if (len_trim(frc%f(fld)%calendar) == 0) frc%f(fld)%calendar = 'none'
        call nc_close(ncid)

        ! transfer time-axis to days since 0001 (473-503)
        frc%f(fld)%nc_time = frc%f(fld)%nc_time / frc%freq
        frc%f(fld)%nc_time = frc%f(fld)%nc_time + &
            forcing_julday(frc%iyear, frc%imm, frc%idd, frc%f(fld)%calendar)
        yyyy = forcing_calyear(int(frc%f(fld)%nc_time(1)), frc%f(fld)%calendar)
        frc%f(fld)%year_orig = yyyy
        frc%f(fld)%nc_time = frc%f(fld)%nc_time - forcing_julday(yyyy, 1, 1, frc%f(fld)%calendar)
        frc%f(fld)%nc_time = frc%f(fld)%nc_time + forcing_julday(year, 1, 1, frc%f(fld)%calendar)
        ! mid-point shift (505-512): pi has tmid=1 -> skipped
        if (frc%tmid /= 1 .and. frc%f(fld)%ntime > 1) then
            do i = 1, frc%f(fld)%ntime - 1
                frc%f(fld)%nc_time(i) = (frc%f(fld)%nc_time(i+1) + frc%f(fld)%nc_time(i))/2.0_WP
            end do
            frc%f(fld)%nc_time(frc%f(fld)%ntime) = frc%f(fld)%nc_time(frc%f(fld)%ntime) + &
                (frc%f(fld)%nc_time(frc%f(fld)%ntime) - frc%f(fld)%nc_time(frc%f(fld)%ntime-1))/2.0
        end if
        ! lat flip if descending (519-526)
        frc%f(fld)%flip_lat = 0
        if (frc%f(fld)%nlat > 1) then
            if (frc%f(fld)%nc_lat(1) > frc%f(fld)%nc_lat(frc%f(fld)%nlat)) then
                frc%f(fld)%flip_lat = 1
                frc%f(fld)%nc_lat = frc%f(fld)%nc_lat(frc%f(fld)%nlat:1:-1)
            end if
        end if
        ! cyclic lon: shift the two halo columns by +-360 (534-537)
        if (frc%ic_cyclic) then
            frc%f(fld)%nc_lon(1)            = frc%f(fld)%nc_lon(1)            - 360._WP
            frc%f(fld)%nc_lon(frc%f(fld)%nlon) = frc%f(fld)%nc_lon(frc%f(fld)%nlon) + 360._WP
        end if
    end subroutine forcing_read_grid

    ! ---- per-node bilinear source indices (FESOM2 nc_sbc_ini, 652-675) ----------
    subroutine forcing_build_bilin(frc, fld, mesh, partit)
        type(t_atm_forcing), intent(inout) :: frc
        integer, intent(in) :: fld
        type(t_mesh),   intent(in), target :: mesh
        type(t_partit), intent(in), target :: partit
        integer :: i
        real(WP) :: x, y
        do i = 1, frc%nnod
            x = mesh%geo_coord_nod2D(1, i)/rad
            if (x < 0) x = x + 360._WP
            y = mesh%geo_coord_nod2D(2, i)/rad
            if (x < frc%f(fld)%nc_lon(frc%f(fld)%nlon) .and. x >= frc%f(fld)%nc_lon(1)) then
                call forcing_binarysearch(frc%f(fld)%nlon, frc%f(fld)%nc_lon, x, frc%idx_i(fld, i))
            else
                if (x < frc%f(fld)%nc_lon(1)) then
                    frc%idx_i(fld, i) = -1
                else
                    frc%idx_i(fld, i) = 0
                end if
            end if
            if (y < frc%f(fld)%nc_lat(frc%f(fld)%nlat) .and. y >= frc%f(fld)%nc_lat(1)) then
                call forcing_binarysearch(frc%f(fld)%nlat, frc%f(fld)%nc_lat, y, frc%idx_j(fld, i))
            else
                if (y < frc%f(fld)%nc_lat(1)) then
                    frc%idx_j(fld, i) = -1
                else
                    frc%idx_j(fld, i) = 0
                end if
            end if
        end do
    end subroutine forcing_build_bilin

    ! ---- read the bracketing slice(s), spatial bilinear, build coef (FESOM2 getcoeffld) --
    ! PERF (2026-06-28): the JRA55-do files are DEFLATE-compressed netCDF-4/HDF5 chunked one (lat,lon)
    ! slice/chunk, so an nc_open re-parses HDF5 metadata and a slice read re-inflates ~800 KB — on
    ! EVERY rank. We mirror FESOM2: keep the file open (persistent handle, reopen only on year change)
    ! and double-buffer, so only the genuinely-new slice (t_indx_p1) is read each crossing; whichever
    ! buffer already holds t_indx is reused (FESOM2 getcoeffld:866-898; slot2 is always read fresh).
    ! Byte-exact: the reused bytes are exactly what re-reading the same record would deserialize.
    subroutine forcing_getcoeffld(frc, fld, year, rdate, mesh, partit)
        type(t_atm_forcing), intent(inout), target :: frc
        integer,  intent(in) :: fld, year
        real(WP), intent(in) :: rdate
        type(t_mesh),   intent(in), target :: mesh
        type(t_partit), intent(in), target :: partit
        character(len=512) :: fname
        character(len=4) :: cyear
        character(len=8) :: env
        integer :: ncid, nlon, nlat, ntime, t_indx, t_indx_p1, ii, i, j, ip1, jp1, extrp, ln, ios
        real(WP) :: delta_t, x, y, x1, x2, y1, y2, denom, data1, data2
        real(real32), allocatable :: raw(:,:)
        real(real32), pointer     :: sbc1(:,:), sbc2(:,:)
        logical :: cache1

        ! one-time toggle read (FESOM3_FORCING_PERSIST: default ON = persistent handle + double-buffer;
        ! '0' restores the legacy open + read-both + close every crossing, for A/B and as a safety valve)
        if (.not. frc_cache_init) then
            call get_environment_variable('FESOM3_FORCING_PERSIST', env, length=ln, status=ios)
            if (ios == 0 .and. ln > 0) frc_use_cache = (trim(env) /= '0')
            frc_cache_init = .true.
        end if

        nlon = frc%f(fld)%nlon; nlat = frc%f(fld)%nlat; ntime = frc%f(fld)%ntime
        write(cyear, '(I4)') year
        fname = trim(frc%f(fld)%file_base)//trim(adjustl(cyear))//'.nc'

        call forcing_binarysearch(ntime, frc%f(fld)%nc_time, rdate, t_indx)
        if (t_indx < ntime .and. t_indx > 0) then
            t_indx_p1 = t_indx + 1
            delta_t   = frc%f(fld)%nc_time(t_indx_p1) - frc%f(fld)%nc_time(t_indx)
        else if (t_indx > 0) then            ! no extrapolation into the future
            t_indx = ntime; t_indx_p1 = t_indx; delta_t = 1.0_WP
        else                                 ! t_indx < 1: no extrapolation back in time
            t_indx = 1; t_indx_p1 = t_indx; delta_t = 1.0_WP
        end if
        ! M8b: persist the resolved bracket so forcing_sbc_do's crossing test can read it next
        ! step (FESOM2 getcoeffld stores via the sbc_flfi%t_indx/%t_indx_p1 pointers).
        frc%f(fld)%t_indx    = t_indx
        frc%f(fld)%t_indx_p1 = t_indx_p1

        ! ensure the persistent double buffers exist (nlon/nlat are year-invariant for the grids)
        if (.not. allocated(frc%f(fld)%sbc_a)) then
            allocate(frc%f(fld)%sbc_a(nlon, nlat), frc%f(fld)%sbc_b(nlon, nlat))
            frc%f(fld)%ia_indx = -1; frc%f(fld)%ib_indx = -1
        end if

        ! double-buffer slot selection (FESOM2 getcoeffld:866-898): reuse whichever buffer already
        ! holds t_indx as slot1; slot2 (t_indx_p1) is ALWAYS read fresh. Toggle off => always read both
        ! (the `frc_use_cache .and.` guards force the else branch), recycling only the buffer storage.
        cache1 = .false.
        if (frc_use_cache .and. frc%f(fld)%ia_indx == t_indx) then
            cache1 = .true.;  sbc1 => frc%f(fld)%sbc_a; sbc2 => frc%f(fld)%sbc_b
            frc%f(fld)%ib_indx = t_indx_p1
        else if (frc_use_cache .and. frc%f(fld)%ib_indx == t_indx) then
            cache1 = .true.;  sbc1 => frc%f(fld)%sbc_b; sbc2 => frc%f(fld)%sbc_a
            frc%f(fld)%ia_indx = t_indx_p1
        else
            cache1 = .false.; sbc1 => frc%f(fld)%sbc_a; sbc2 => frc%f(fld)%sbc_b
            frc%f(fld)%ia_indx = t_indx;  frc%f(fld)%ib_indx = t_indx_p1
        end if

        ! open: persistent handle (reopen only on year/file change) or a fresh open per call (toggle off)
        call timer_start(TMR_FRC_OPEN)
        if (frc_use_cache) then
            if (frc%f(fld)%ncid < 0 .or. frc%f(fld)%ncid_year /= year) then
                if (frc%f(fld)%ncid >= 0) call nc_close(frc%f(fld)%ncid)
                frc%f(fld)%ncid = nc_open_read(fname)
                frc%f(fld)%ncid_year = year
            end if
            ncid = frc%f(fld)%ncid
        else
            ncid = nc_open_read(fname)
        end if
        call timer_stop(TMR_FRC_OPEN)

        ! read the genuinely-new slice(s) into the halo'd buffers (interior 2:nlon-1; halo mirror)
        call timer_start(TMR_FRC_READ)
        allocate(raw(nlon-2, nlat))
        if (.not. cache1) then
            call nc_get_slice_r4(ncid, [frc%f(fld)%varname], t_indx, raw)
            sbc1(2:nlon-1, 1:nlat) = raw
            sbc1(1,    1:nlat) = sbc1(nlon-1, 1:nlat)
            sbc1(nlon, 1:nlat) = sbc1(2,      1:nlat)
        end if
        call nc_get_slice_r4(ncid, [frc%f(fld)%varname], t_indx_p1, raw)
        sbc2(2:nlon-1, 1:nlat) = raw
        sbc2(1,    1:nlat) = sbc2(nlon-1, 1:nlat)
        sbc2(nlon, 1:nlat) = sbc2(2,      1:nlat)
        deallocate(raw)
        if (.not. frc_use_cache) call nc_close(ncid)
        call timer_stop(TMR_FRC_READ)

        call timer_start(TMR_FRC_INTRP2)
        do ii = 1, frc%nnod
            i = frc%idx_i(fld, ii); j = frc%idx_j(fld, ii)
            ip1 = i + 1; jp1 = j + 1
            x = mesh%geo_coord_nod2D(1, ii)/rad
            if (x < 0.0_WP) x = x + 360._WP
            y = mesh%geo_coord_nod2D(2, ii)/rad
            extrp = 0
            if (i ==  0) then; i = nlon; ip1 = i; extrp = extrp + 1; end if
            if (i == -1) then; i = 1;    ip1 = i; extrp = extrp + 1; end if
            if (j ==  0) then; j = nlat; jp1 = j; extrp = extrp + 2; end if
            if (j == -1) then; j = 1;    jp1 = j; extrp = extrp + 2; end if
            x1 = frc%f(fld)%nc_lon(i);  x2 = frc%f(fld)%nc_lon(ip1)
            y1 = frc%f(fld)%nc_lat(j);  y2 = frc%f(fld)%nc_lat(jp1)
            if (extrp == 0) then
                denom = (x2 - x1)*(y2 - y1)
                data1 = ( sbc1(i,j)*(x2-x)*(y2-y) + sbc1(ip1,j)*(x-x1)*(y2-y) + &
                          sbc1(i,jp1)*(x2-x)*(y-y1) + sbc1(ip1,jp1)*(x-x1)*(y-y1) ) / denom
                data2 = ( sbc2(i,j)*(x2-x)*(y2-y) + sbc2(ip1,j)*(x-x1)*(y2-y) + &
                          sbc2(i,jp1)*(x2-x)*(y-y1) + sbc2(ip1,jp1)*(x-x1)*(y-y1) ) / denom
            else if (extrp == 1) then
                denom = (y2 - y1)
                data1 = ( sbc1(i,j)*(y2-y) + sbc1(ip1,jp1)*(y-y1) ) / denom
                data2 = ( sbc2(i,j)*(y2-y) + sbc2(ip1,jp1)*(y-y1) ) / denom
            else if (extrp == 2) then
                denom = (x2 - x1)
                data1 = ( sbc1(i,j)*(x2-x) + sbc1(ip1,jp1)*(x-x1) ) / denom
                data2 = ( sbc2(i,j)*(x2-x) + sbc2(ip1,jp1)*(x-x1) ) / denom
            else
                data1 = sbc1(i,j); data2 = sbc2(i,j)
            end if
            frc%coef_a(fld, ii) = (data2 - data1) / delta_t
            frc%coef_b(fld, ii) = data1 - frc%coef_a(fld, ii) * frc%f(fld)%nc_time(t_indx)
        end do
        call timer_stop(TMR_FRC_INTRP2)
        sbc1 => null(); sbc2 => null()
    end subroutine forcing_getcoeffld

    ! ---- g2r rotation of the wind interpolation coefficients (FESOM2 703-707) -----
    subroutine forcing_rotate_wind(frc, mesh, partit)
        type(t_atm_forcing), intent(inout) :: frc
        type(t_mesh),   intent(in), target :: mesh
        type(t_partit), intent(in), target :: partit
        integer :: i
        if (frc%i_xwind == 0 .or. frc%i_ywind == 0 .or. .not. frc%rotated_grid) return
        do i = 1, frc%nnod
            call vector_g2r(frc%coef_a(frc%i_xwind,i), frc%coef_a(frc%i_ywind,i), &
                            mesh%coord_nod2D(1,i), mesh%coord_nod2D(2,i), 0)
            call vector_g2r(frc%coef_b(frc%i_xwind,i), frc%coef_b(frc%i_ywind,i), &
                            mesh%coord_nod2D(1,i), mesh%coord_nod2D(2,i), 0)
        end do
    end subroutine forcing_rotate_wind

    ! ---- atmdata = rdate*coef_a + coef_b (FESOM2 data_timeinterp, 1041) ----------
    subroutine forcing_timeinterp(frc, rdate, partit)
        type(t_atm_forcing), intent(inout) :: frc
        real(WP), intent(in) :: rdate
        type(t_partit), intent(in), target :: partit
        integer :: fld, i
        do fld = 1, frc%nfld
            do i = 1, frc%nnod
                frc%atmdata(fld, i) = rdate * frc%coef_a(fld, i) + frc%coef_b(fld, i)
            end do
        end do
    end subroutine forcing_timeinterp

    ! ---- per-step record/day forcing rollover (FESOM2 sbc_do, body 1524-1576) ------
    ! Called every step from the driver (mirror FESOM2 update_atm_forcing -> sbc_do). Recompute
    ! the running model rdate (with the -dt/2 half-step shift) from the clock and, where it has
    ! crossed PAST the current bracket end (t_indx_p1) AND we are not already at the last record,
    ! re-read the two bracketing slices via forcing_getcoeffld (which re-stores the bracket). On a
    ! wind-coefficient refresh, rotate the wind interpolation coefficients g2r (as nc_sbc_ini's
    ! cold start did). M8c adds the year-file branch (sbc_do:1510-1519); the noleap/leap special-case
    ! (:1533-1554) stays OMITTED — it is gated on include_fleapyear==.false. and is dead for the
    ! production JRA55 config (include_fleapyear=.true., model + forcing share the gregorian calendar).
    ! data_timeinterp (atmdata = rdate*coef_a + coef_b) stays the caller's separate per-step step.
    subroutine forcing_sbc_do(frc, mesh, partit)
        use mod_clock,  only: yearnew, yearold, daynew, timenew
        use mod_config, only: dt
        type(t_atm_forcing), intent(inout)    :: frc
        type(t_mesh),   intent(in),    target :: mesh
        type(t_partit), intent(in),    target :: partit
        integer  :: fld
        real(WP) :: rdate
        logical  :: do_rotation_wind, force_newcoeff, needs_refresh
        ! M8c year-file rollover (sbc_do:1510-1519): on a year change re-read EVERY field's grid +
        ! time axis from the next year's file (forcing_read_grid re-anchors nc_time to yearnew, see its
        ! re-entrancy guards) and force an immediate coeff refresh. The per-node bilinear source
        ! indices (forcing_build_bilin) are NOT rebuilt — the grid is identical year-to-year, exactly
        ! as FESOM2 (nc_readTimeGrid only). yearnew/yearold come straight from the model clock.
        force_newcoeff = .false.
        if (yearnew /= yearold) then
            do fld = 1, frc%nfld
                call forcing_read_grid(frc, fld, yearnew)
                ! invalidate the double-buffer (new year => read both brackets fresh; mirror FESOM2's
                ! yearold==yearnew cache guard). getcoeffld reopens the persistent handle on ncid_year/=year.
                frc%f(fld)%ia_indx = -1; frc%f(fld)%ib_indx = -1
                ! ... and reset the time bracket. forcing_read_grid REALLOCATES nc_time to the new
                ! year's length, so a stale index from the previous year can point PAST THE END:
                ! leaving a leap year (2928 three-hourly records) for a normal one (2920) leaves
                ! t_indx_p1 up to 2928 against a 2920-long array. The crossing test below then reads
                ! out of bounds, and the ORDERED compare it compiles to (comisd) raises `floating
                ! invalid` the moment that garbage happens to be a NaN. getcoeffld does re-search and
                ! fix the index, but only AFTER the test that crashes. Symptom: the run dies with
                ! forrtl (65) at 1968->1969, 1972->1973, ... and never at any other rollover.
                frc%f(fld)%t_indx = 1; frc%f(fld)%t_indx_p1 = 1
            end do
            force_newcoeff = .true.
            if (partit%mype == 0) write(*,'(a,i0,a,i0)') &
                ' forcing_sbc_do: YEAR ROLLOVER ', yearold, ' -> ', yearnew
        end if
        do_rotation_wind = .false.
        do fld = 1, frc%nfld
            ! running model rdate on the field's calendar (sbc_do:1527-1528)
            rdate = real(forcing_julday(yearnew, 1, 1, frc%f(fld)%calendar), WP) &
                  + real(daynew - 1, WP) + timenew/86400._WP - dt/86400._WP/2._WP
            ! crossing test (sbc_do:1561): rdate past the bracket end AND not at the last record, OR a
            ! year rollover just forced a refresh (force_newcoeff). getcoeffld re-binarysearches rdate
            ! so the stale (prev-year) t_indx is reset to the new year's early-January bracket.
            ! NB: Fortran does NOT guarantee short-circuit evaluation, so `.or. force_newcoeff`
            ! is no protection for the array reads in the first operand -- they are evaluated
            ! regardless. Test the rollover flag FIRST, in its own branch, so the nc_time reads
            ! only happen when the bracket is known to belong to the current year's array.
            needs_refresh = force_newcoeff
            if (.not. needs_refresh) then
                needs_refresh = (rdate > frc%f(fld)%nc_time(frc%f(fld)%t_indx_p1)) .and. &
                                (frc%f(fld)%nc_time(frc%f(fld)%t_indx) <             &
                                 frc%f(fld)%nc_time(frc%f(fld)%ntime))
            end if
            if (needs_refresh) then
                call forcing_getcoeffld(frc, fld, yearnew, rdate, mesh, partit)
                ! M8b diagnostic (stdout only — never touches the dump): confirm the rollover fired.
                if (partit%mype == 0) write(*,'(a,i0,a,i0,a,i0,a,es18.10)') &
                    ' forcing_sbc_do: REFRESH fld=', fld, ' new bracket t_indx=', &
                    frc%f(fld)%t_indx, '/', frc%f(fld)%t_indx_p1, ' rdate=', rdate
                if (fld == frc%i_xwind .and. frc%rotated_grid) do_rotation_wind = .true.
            end if
        end do
        if (do_rotation_wind) call forcing_rotate_wind(frc, mesh, partit)
    end subroutine forcing_sbc_do

    ! lowercase a calendar string (FESOM2 lowercase, 2798-2832)
    function lc(s) result(o)
        character(len=*), intent(in) :: s
        character(len=len(s)) :: o
        integer :: i, k
        o = s
        do i = 1, len_trim(s)
            k = iachar(s(i:i))
            if (k >= iachar('A') .and. k <= iachar('Z')) o(i:i) = achar(k + 32)
        end do
    end function lc

end module mod_forcing_read
