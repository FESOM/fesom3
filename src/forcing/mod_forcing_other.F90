module mod_forcing_other
    ! M3f-3b: runoff + sea-surface-salinity (Ssurf) monthly-climatology READ. A faithful
    ! transcription of FESOM2 v2.7.3:
    !   read_other_NetCDF  (gen_modules_read_NetCDF.F90:6)
    !   interp_2d_field    (gen_interpolation.F90:145)
    !
    ! This is a DISTINCT reader from mod_forcing_read (the NCAR bilinear-coef path): it reads
    ! ONE 2D slice of a (lon,lat,time) field, fills missing/land values ON THE RAW REGULAR
    ! GRID (a 30-neighbour expanding-box average when check_dummy=.true.; 0.0 otherwise), then
    ! does a direct per-node bilinear interpolation to the model vertices. There is NO time
    ! interpolation — runoff is time-constant in CORE, and the SSS climatology is read once per
    ! month (a single record for a short January run, update_monthly_flag = mstep==1).
    !
    ! FESOM2 call sites (gen_surface_forcing.F90):
    !   runoff  :1302  read_other_NetCDF(nm_runoff_file,  'Foxx_o_roff', 1, runoff, .false., .true.)
    !                  then runoff = runoff/1000  (kg/s/m^2 -> m/s)
    !   Ssurf   :1601  read_other_NetCDF(nm_sss_data_file,'SALT', month, Ssurf,  .true.,  .true.)
    !
    ! BIT-IDENTITY NOTES:
    ! * The raw-grid dummy fill AND the per-node interpolation are PARTITION-INDEPENDENT — each
    !   owned node interpolates from the full global raw grid, with no cross-rank reduction — so
    !   the owned values are byte-identical at any partition. Like mod_forcing_read, every rank
    !   reads the file directly (the bytes are deterministic; the oracle's read-on-rank-0 + BCast
    !   moves the same bytes, so no BCast is needed here). M3f-4 reuses this verbatim.
    ! * The on-disk fields are float; netCDF converts float -> real64 on read EXACTLY as the
    !   oracle (real64 ncdata). The 'missing_value' attribute (1.e30f for runoff, -99.f for SALT)
    !   likewise converts float -> real64, so the ==miss equality test matches the oracle's.
    ! * All arithmetic is real64 (= WP in the dp anchor). geo_coord_nod2D/rad is the same
    !   byte-exact deg conversion already proven in M3f-3a (forcing_build_bilin).
    use, intrinsic :: iso_fortran_env, only: real64
    use mod_precision, only: WP
    use mod_constants,  only: rad
    use mod_io_netcdf
    use mod_mesh,   only: t_mesh
    use mod_partit, only: t_partit
    implicit none
    private
    public :: read_other_NetCDF

contains

    ! ---- read a 2D field, fill dummies on the raw grid, interpolate to vertices ----------
    subroutine read_other_NetCDF(file, vari, itime, model_2Darray, check_dummy, do_onvert, &
                                 mesh, partit)
        character(len=*), intent(in)  :: file, vari
        integer,          intent(in)  :: itime
        real(kind=WP),    intent(out) :: model_2Darray(:)
        logical,          intent(in)  :: check_dummy, do_onvert
        type(t_mesh),   intent(in)           :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: i, j, ii, jj, k, n, num, flag, cnt
        integer :: latlen, lonlen, ncid, st
        real(real64) :: miss, aux
        real(real64), allocatable :: lon(:), lat(:)
        real(real64), allocatable :: ncdata(:,:), ncdata_temp(:,:)
        real(real64), allocatable :: temp_x(:), temp_y(:)

        ncid   = nc_open_read(file)
        latlen = nc_dimlen(ncid, ['lat'])
        lonlen = nc_dimlen(ncid, ['lon'])

        ! lat / lon axes (float on disk -> real64)
        allocate(lat(latlen), lon(lonlen))
        call nc_get_axis_dp(ncid, ['lat'], lat)
        call nc_get_axis_dp(ncid, ['lon'], lon)

        ! make sure range 0. - 360.
        do n = 1, lonlen
            if (lon(n) < 0.0_WP) lon(n) = lon(n) + 360._WP
        end do

        ! data slice + missing value
        allocate(ncdata(lonlen,latlen), ncdata_temp(lonlen,latlen))
        ncdata = 0.0_WP
        call nc_get_slice_dp(ncid, [vari], itime, ncdata)
        ! missing_value: PHC2 SSS carries missing_value=-99 (st==0 => miss=-99). The Sweeney chl file
        ! has NO missing_value attribute — FESOM2 then leaves `miss` undefined, but the chl field is a
        ! fully-extrapolated [.001,.6] climatology with no -99/sentinels so the fill loop never fires;
        ! miss=-99 here reproduces that no-fill (the hardcoded -99 check below stays the only trigger).
        miss = nc_get_att_dp(ncid, [vari], 'missing_value', stat=st)
        if (st /= 0) miss = -99.0_real64
        call nc_close(ncid)

        ! fill missing values on the raw regular grid
        ncdata_temp = ncdata
        do i = 1, lonlen
            do j = 1, latlen
                if (ncdata(i,j) == miss .or. ncdata(i,j) == -99.0_WP) then  !!
                    if (check_dummy) then
                        aux = 0.0_WP
                        cnt = 0
                        do k = 1, 30
                            do ii = max(1,i-k), min(lonlen,i+k)
                                do jj = max(1,j-k), min(latlen,j+k)
                                    if (ncdata_temp(ii,jj) /= miss .and. ncdata_temp(ii,jj) /= -99.0_WP) then  !!
                                        aux = aux + ncdata_temp(ii,jj)
                                        cnt = cnt + 1
                                    end if
                                end do  !jj
                            end do  !ii
                            if (cnt > 0) then
                                ncdata(i,j) = aux/cnt
                                exit
                            end if
                        end do  !k
                    else
                        ncdata(i,j) = 0.0_WP
                    end if
                end if
            end do
        end do

        ! interpolation coordinates (vertices). do_onvert=.false. (element centroids) is
        ! unused by runoff/Ssurf and not ported (would need cyclic_length; gate when needed).
        if (do_onvert) then
            if (present(partit)) then
                num = partit%myDim_nod2D + partit%eDim_nod2D
            else
                num = mesh%nod2D
            end if
            allocate(temp_x(num), temp_y(num))
            do n = 1, num
                temp_x(n) = mesh%geo_coord_nod2D(1,n)/rad
                temp_y(n) = mesh%geo_coord_nod2D(2,n)/rad
                ! change lon range to [0 360]
                if (temp_x(n) < 0._WP) temp_x(n) = temp_x(n) + 360.0_WP
            end do
        else
            error stop 'read_other_NetCDF: do_onvert=.false. (centroid) path not ported'
        end if

        ! do interpolation
        flag = 0
#if defined(USE_SINGLE_PRECISION)
        ! the interpolation runs in real64; the model field is WP (narrowed once)
        block
            real(real64), allocatable :: dm(:)
            allocate(dm(num))
            call interp_2d_field(lonlen, latlen, lon, lat, ncdata, num, temp_x, temp_y, &
                                 dm, flag)
            model_2Darray(1:num) = real(dm, WP)
            deallocate(dm)
        end block
#else
        call interp_2d_field(lonlen, latlen, lon, lat, ncdata, num, temp_x, temp_y, &
                             model_2Darray, flag)
#endif
        deallocate(temp_y, temp_x, ncdata_temp, ncdata, lon, lat)
    end subroutine read_other_NetCDF

    ! ---- 2D bilinear from a regular global grid to specified nodes (FESOM2 interp_2d_field) ---
    ! Order of lon_reg: monotonically increasing in [0 360]; lat_reg: increasing in [-90 90];
    ! lon_mod is in [0 360]. phase_flag=1 interpolates a phase angle (dead here, flag=0).
    subroutine interp_2d_field(num_lon_reg, num_lat_reg, lon_reg, lat_reg, data_reg, &
                               num_mod, lon_mod, lat_mod, data_mod, phase_flag)
        integer,      intent(in)  :: num_lon_reg, num_lat_reg, num_mod, phase_flag
        real(real64), intent(in)  :: lon_reg(num_lon_reg), lat_reg(num_lat_reg)
        real(real64), intent(in)  :: data_reg(num_lon_reg, num_lat_reg)
        real(real64), intent(in)  :: lon_mod(num_mod), lat_mod(num_mod)
        real(real64), intent(out) :: data_mod(num_mod)
        integer  :: n, i
        integer  :: ind_lat_h, ind_lat_l, ind_lon_h, ind_lon_l
        real(real64) :: x, y, diff
        real(real64) :: rt_lat1, rt_lat2, rt_lon1, rt_lon2
        real(real64) :: data_ll, data_lh, data_hl, data_hh
        real(real64) :: data_lo, data_up

        if (lon_reg(1) < 0.0_WP .or. lon_reg(num_lon_reg) > 360._WP) then
            write(*,*) 'Error in 2D interpolation!'
            write(*,*) 'The regular grid is not in the proper longitude range.'
            error stop 1
        end if

        do n = 1, num_mod
            x = lon_mod(n)
            y = lat_mod(n)
            ! find the surrounding rectangular box and get interpolation ratios
            ! 1) north-south direction
            if (y < lat_reg(1)) then
                ind_lat_h = 2
                ind_lat_l = 1
                y = lat_reg(1)
            elseif (y > lat_reg(num_lat_reg)) then
                ind_lat_h = num_lat_reg
                ind_lat_l = num_lat_reg-1
                y = lat_reg(num_lat_reg)
            else
                do i = 2, num_lat_reg
                    if (lat_reg(i) >= y) then
                        ind_lat_h = i
                        ind_lat_l = i-1
                        exit
                    end if
                end do
            end if
            diff = lat_reg(ind_lat_h)-lat_reg(ind_lat_l)
            rt_lat1 = (lat_reg(ind_lat_h)-y)/diff
            rt_lat2 = 1.0_WP-rt_lat1
            ! 2) east_west direction
            if (x < lon_reg(1)) then
                ind_lon_h = 1
                ind_lon_l = num_lon_reg
                diff = lon_reg(ind_lon_h)+(360._WP-lon_reg(ind_lon_l))
                rt_lon1 = (lon_reg(ind_lon_h)-x)/diff
                rt_lon2 = 1.0_WP-rt_lon1
            elseif (x > lon_reg(num_lon_reg)) then
                ind_lon_h = 1
                ind_lon_l = num_lon_reg
                diff = lon_reg(ind_lon_h)+(360._WP-lon_reg(ind_lon_l))
                rt_lon2 = (x-lon_reg(ind_lon_l))/diff
                rt_lon1 = 1.0_WP-rt_lon2
            else
                do i = 2, num_lon_reg
                    if (lon_reg(i) >= x) then
                        ind_lon_h = i
                        ind_lon_l = i-1
                        exit
                    end if
                end do
                diff = lon_reg(ind_lon_h)-lon_reg(ind_lon_l)
                rt_lon1 = (lon_reg(ind_lon_h)-x)/diff
                rt_lon2 = 1.0_WP-rt_lon1
            end if
            !
            data_ll = data_reg(ind_lon_l,ind_lat_l)
            data_lh = data_reg(ind_lon_l,ind_lat_h)
            data_hl = data_reg(ind_lon_h,ind_lat_l)
            data_hh = data_reg(ind_lon_h,ind_lat_h)
            !
            ! interpolate data
            if (phase_flag == 1) then   ! interpolate phase value (0,360)
                if (abs(data_ll-data_hl) > 180._WP) then
                    if (data_ll < data_hl) then
                        data_ll = data_ll+360._WP
                    else
                        data_hl = data_hl+360._WP
                    end if
                end if
                if (abs(data_lh-data_hh) > 180._WP) then
                    if (data_lh < data_hh) then
                        data_lh = data_lh+360._WP
                    else
                        data_hh = data_hh+360._WP
                    end if
                end if
                data_lo = data_ll*rt_lon1 + data_hl*rt_lon2
                data_up = data_lh*rt_lon1 + data_hh*rt_lon2
                if (abs(data_lo-data_up) > 180._WP) then
                    if (data_lo < data_up) then
                        data_lo = data_lo+360._WP
                    else
                        data_up = data_up+360._WP
                    end if
                end if
                data_mod(n) = data_lo*rt_lat1 + data_up*rt_lat2
                if (data_mod(n) >= 360._WP) then
                    data_mod(n) = mod(data_mod(n), 360._WP)
                end if
            else   ! other case
                data_mod(n) = (data_ll*rt_lon1 + data_hl*rt_lon2)*rt_lat1 + &
                              (data_lh*rt_lon1 + data_hh*rt_lon2)*rt_lat2
            end if
        end do
    end subroutine interp_2d_field

end module mod_forcing_other
