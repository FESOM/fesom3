program fesom_icgather
    ! Cold-start initial T/S on dist_<NP>, gathered by global node id: the check of the IC
    ! hole filling's partition (in)dependence (FESOM3_IC_EXTRAP_DET, FESOM2 PR #979).
    !
    ! Builds the PHC3 IC exactly as fesom_lifecycle_native_mr does (flush-to-zero, the same
    ! mesh read and rotation, linfs Z_3d_n at the initial state, the same t_ic3d_config,
    ! do_ic3d through partit), without forcing or time stepping. Rank 0 then gathers the
    ! owned (level, node) values by GLOBAL node id, bit-exactly (an integer sum of the IEEE
    ! bit patterns; every node has one owner), and
    !   FESOM3_IC_OUT=<file>  writes them (Fortran stream: int32 nod2D, int32 nl-1, then
    !                         temp(nl-1, nod2D) and salt(nl-1, nod2D) real64 in global-id order)
    !   FESOM3_IC_REF=<file>  compares with such a file: per tracer the number of (level, node)
    !                         values whose bits differ and max|delta|, then "IC IDENTICAL" or
    !                         "IC DIFFER"; FESOM3_IC_EXPECT=identical|differ turns the other
    !                         verdict into an error stop (the ctests' teeth).
    !
    !   FESOM3_MESH_DIR       mesh dir (dist_<NP>/ for NP > 1)    default: CORE2
    !   FESOM3_IC_FILE        IC netcdf                            default: pool phc3.0_winter.nc
    !   FESOM3_IC_EXTRAP_DET, FESOM3_IC_EXTRAP_TOL  as in the lifecycle driver (oce_initial_state)
    use mpi
    use, intrinsic :: ieee_arithmetic
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use mod_precision,      only: WP, MP
    use mod_mesh,           only: t_mesh
    use mod_partit,         only: t_partit
    use mod_partitioning,   only: par_init, par_ex, set_partition
    use mod_mesh_read,      only: read_mesh
    use mod_mesh_areas,     only: compute_geometry
    use mod_tracer,         only: t_tracer
    use oce_initial_state,  only: t_ic3d_config, do_ic3d, ic3d_extrap_from_env
    implicit none

    character(len=512)  :: mesh_dir, ic_file, out_path, ref_path, expect
    type(t_partit)      :: partit
    type(t_mesh)        :: mesh
    type(t_tracer)      :: tracers
    type(t_ic3d_config) :: ic
    integer :: nsw, n, nl, nzmin, nzmax, nNodO, nNodL, tr, ierr, u, g, ios
    integer(int32) :: hdr(2)
    integer(int64) :: ndiff(2)
    real(kind=MP)  :: zbar_srf, zbar_bot
    real(real64)   :: maxabs(2)
    integer(int64), allocatable :: bits(:,:)
    real(real64),   allocatable :: glob(:,:,:), ref(:,:,:)
    logical :: identical

    call get_environment_variable('FESOM3_MESH_DIR', mesh_dir)
    if (len_trim(mesh_dir) == 0) mesh_dir = '/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2'
    call get_environment_variable('FESOM3_IC_FILE', ic_file)
    if (len_trim(ic_file) == 0) ic_file = '/pool/data/AWICM/FESOM2/INITIAL/phc3.0/phc3.0_winter.nc'
    call get_environment_variable('FESOM3_IC_OUT', out_path)
    call get_environment_variable('FESOM3_IC_REF', ref_path)
    call get_environment_variable('FESOM3_IC_EXPECT', expect)

    call par_init(partit)
    if (ieee_support_underflow_control(1.0_WP)) call ieee_set_underflow_mode(gradual=.false.)
    call set_partition(partit, trim(mesh_dir))
    call read_mesh(mesh, partit, trim(mesh_dir), 50.0_WP, 15.0_WP, -90.0_WP, 360.0_WP, &
                   force_rotation=.true., n_cw_swaps=nsw)
    call compute_geometry(mesh, partit, cartesian=.false.)
    nl = mesh%nl
    if (partit%npes == 1) then
        nNodO = mesh%nod2D;  nNodL = mesh%nod2D
    else
        nNodO = partit%myDim_nod2D
        nNodL = partit%myDim_nod2D + partit%eDim_nod2D
    end if

    ! linfs Z_3d_n at the initial state, as fesom_lifecycle_native_mr
    allocate(mesh%zbar_3d_n(nl, nNodL), mesh%Z_3d_n(nl-1, nNodL))
    mesh%zbar_3d_n = 0.0_MP; mesh%Z_3d_n = 0.0_MP
    do n = 1, nNodL
        if (mesh%nlevels_nod2D(n) <= 0) cycle
        nzmin = mesh%ulevels_nod2D(n); nzmax = mesh%nlevels_nod2D(n)
        zbar_srf = mesh%zbar(nzmin)
        zbar_bot = mesh%zbar(nzmax)
        mesh%zbar_3d_n(1:nzmin-1, n)       = mesh%zbar(1:nzmin-1)
        mesh%zbar_3d_n(nzmin, n)           = zbar_srf
        mesh%zbar_3d_n(nzmin+1:nzmax-1, n) = mesh%zbar(nzmin+1:nzmax-1)
        mesh%zbar_3d_n(nzmax, n)           = zbar_bot
        mesh%Z_3d_n(1:nzmin-1, n)          = mesh%Z(1:nzmin-1)
        mesh%Z_3d_n(nzmin, n)              = mesh%zbar_3d_n(nzmin,n)   + (mesh%zbar_3d_n(nzmin+1,n)-zbar_srf)/2
        mesh%Z_3d_n(nzmin+1:nzmax-2, n)    = mesh%Z(nzmin+1:nzmax-2)
        mesh%Z_3d_n(nzmax-1, n)            = mesh%zbar_3d_n(nzmax-1,n) + (zbar_bot-mesh%zbar_3d_n(nzmax-1,n))/2
    end do

    tracers%num_tracers = 2
    allocate(tracers%data(2))
    allocate(tracers%data(1)%values(nl-1, nNodL), tracers%data(2)%values(nl-1, nNodL))
    tracers%data(1)%values = 0.0_WP;  tracers%data(1)%ID = 1
    tracers%data(2)%values = 0.0_WP;  tracers%data(2)%ID = 2
    ic%n_ic3d      = 2
    ic%idlist(1:2) = [2, 1]
    ic%filelist(1) = trim(ic_file)
    ic%filelist(2) = trim(ic_file)
    ic%varlist(1)  = 'salt'
    ic%varlist(2)  = 'temp'
    ic%t_insitu    = .true.
    ic%ic_cyclic   = .true.
    ic%dummy       = 1.e10_WP
    call ic3d_extrap_from_env(ic)
    if (partit%mype == 0) write(*,'(a,i0,a,i0,a,i0,a,l1,a,es10.3)') 'fesom_icgather: npes=', partit%npes, &
        ' nod2D=', mesh%nod2D, ' nl=', nl, ' extrap_det=', ic%extrap_det, ' extrap_tol=', ic%extrap_tol
    call do_ic3d(tracers, ic, mesh, partit)

    ! gather by global id: integer sum of the bit patterns (owned nodes only, one owner each)
    allocate(bits(nl-1, mesh%nod2D))
    if (partit%mype == 0) allocate(glob(nl-1, mesh%nod2D, 2))
    do tr = 1, 2
        bits = 0_int64
        do n = 1, nNodO
            g = n
            if (partit%npes > 1) g = partit%myList_nod2D(n)
            bits(:, g) = transfer(real(tracers%data(tr)%values(:, n), real64), 0_int64, nl-1)
        end do
        if (partit%mype == 0) then
            call MPI_Reduce(MPI_IN_PLACE, bits, size(bits), MPI_INTEGER8, MPI_SUM, 0, partit%MPI_COMM_FESOM, ierr)
            glob(:, :, tr) = reshape(transfer(bits, 0.0_real64, size(bits)), [nl-1, mesh%nod2D])
        else
            call MPI_Reduce(bits, bits, size(bits), MPI_INTEGER8, MPI_SUM, 0, partit%MPI_COMM_FESOM, ierr)
        end if
    end do
    deallocate(bits)

    if (partit%mype == 0) then
        if (len_trim(out_path) > 0) then
            open(newunit=u, file=trim(out_path), access='stream', form='unformatted', status='replace')
            hdr = [int(mesh%nod2D, int32), int(nl-1, int32)]
            write(u) hdr
            write(u) glob(:, :, 1)
            write(u) glob(:, :, 2)
            close(u)
            write(*,'(a)') 'fesom_icgather: wrote '//trim(out_path)
        end if
        if (len_trim(ref_path) > 0) then
            allocate(ref(nl-1, mesh%nod2D, 2))
            open(newunit=u, file=trim(ref_path), access='stream', form='unformatted', status='old', &
                 action='read', iostat=ios)
            if (ios /= 0) then
                write(*,'(a)') 'fesom_icgather: cannot open '//trim(ref_path)
                error stop 1
            end if
            read(u) hdr
            if (hdr(1) /= mesh%nod2D .or. hdr(2) /= nl-1) then
                write(*,'(a,2i10)') 'fesom_icgather: reference has another shape: ', hdr
                error stop 1
            end if
            read(u) ref(:, :, 1)
            read(u) ref(:, :, 2)
            close(u)
            do tr = 1, 2
                ndiff(tr)  = count(transfer(glob(:, :, tr), 0_int64, size(ref(:,:,tr))) /= &
                                   transfer(ref(:, :, tr),  0_int64, size(ref(:,:,tr))))
                maxabs(tr) = maxval(abs(glob(:, :, tr) - ref(:, :, tr)))
            end do
            write(*,'(a,i0,a,es12.5)') 'fesom_icgather: temp differing values ', ndiff(1), '  max|delta| ', maxabs(1)
            write(*,'(a,i0,a,es12.5)') 'fesom_icgather: salt differing values ', ndiff(2), '  max|delta| ', maxabs(2)
            identical = all(ndiff == 0)
            if (identical) then
                write(*,'(a)') 'fesom_icgather: IC IDENTICAL to '//trim(ref_path)
            else
                write(*,'(a)') 'fesom_icgather: IC DIFFER from '//trim(ref_path)
            end if
            if (trim(expect) == 'identical' .and. .not. identical) error stop 'fesom_icgather: expected identical'
            if (trim(expect) == 'differ' .and. identical) error stop 'fesom_icgather: expected a difference'
        end if
    end if

    call par_ex(partit%MPI_COMM_FESOM, partit%mype)
end program fesom_icgather
