program fesom_restartroundtrip
    ! Restart Stage 4 (Task 4.1) GATE driver: the IN-PROCESS write -> corrupt -> read round-trip that
    ! proves the READ path reproduces the WRITE bit-for-bit over the FULL local extent (owned + halo +
    ! eXDim). The state analog of fesom_restartstate (same real pi mesh, NO ocean, same full oce+ice
    ! field set incl. EVP sigma and the real(MP) mesh%hbar/hnode), extended with the round-trip:
    !
    !   1. fill synthetic OWNED values for every field (partition-independent: value = canonical gid).
    !   2. restart_write a checkpoint (serializes only OWNED canonical values).
    !   3. restart_halo_exchange_all on the ORIGINALS so their halos are owner-derived — apples-to-apples
    !      with the read path (which reconstructs halos by the same exchange).
    !   4. deep-copy the FULL local arrays (owned + halo + eXDim) as the REFERENCE.
    !   5. CORRUPT every live array: the region restart RECONSTRUCTS (owned + the halo its variant
    !      refreshes) -> a wild sentinel; the tail restart does NOT own (eXDim of eDim fields; the halo of
    !      owned-only fields) -> 0.0, modelling a fresh process's zero-allocate (which a real split-restart
    !      relies on and Task 6.1 validates against straight-through).
    !   6. restart_read the checkpoint (gather owned + write back + halo-exchange).
    !   7. compare each live array to its reference over the FULL local extent -> max|Δ| must be 0 for
    !      EVERY field (incl. the MP hbar/hnode). A wrong exchange variant or a gather bug leaves a
    !      nonzero Δ in the reconstructed (owned/halo) region — that is the point.
    !
    ! Run at np=1 (gather is a self-copy; arrays are wholly owned) AND np=2 (real cross-rank gather +
    ! halo exchange). The whole comparison is in-process Fortran (no zarr_diff.py needed): it prints
    ! per-field global max|Δ| and 'ROUNDTRIP OK' / error-stops on any nonzero Δ.
    !
    !   FESOM3_MESH_DIR        mesh dir (default: pi)
    !   FESOM3_RESTARTRT_DIR   checkpoint dir (default: ./restartroundtrip)
    !   FESOM3_AB_ORDER        1|2|3 Adams-Bashforth order (default 2; 3 adds *_AB3 + <tr>_M2 stores)
    !   FESOM3_CHUNK_HORIZ     horizontal chunk (default 1000 => multiple chunks + writer subset)
    !   FESOM3_RESTARTRT_SE    0 (default) | 1 = also the split-explicit fields (dyn%use_ssh_se, the shapes
    !                          of se_init: +10 stores) | 2 = the SE read pre-check: write WITHOUT the SE
    !                          fields, then read with them registered -> must abort with "checkpoint has
    !                          no split-explicit state" (the caller checks the nonzero exit + message)
    use mpi
    use, intrinsic :: iso_fortran_env, only: real64
    use mod_precision,    only: WP, MP, MPI_WP
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init, par_ex, set_partition
    use mod_mesh_read,    only: read_mesh
    use mod_mesh_areas,   only: compute_geometry
    use mod_part_bounds,  only: is_multirank, local_dims
    use mod_dyn,          only: t_dyn
    use mod_tracer,       only: t_tracer
    use mod_ice,          only: t_ice
    use mod_io_restart
    implicit none

    ! per-field reference buffer (the full local array snapshot)
    type :: refbuf
        real(WP), allocatable :: a2(:)
        real(WP), allocatable :: a3(:,:)
    end type refbuf

    character(len=512) :: mesh_dir, out_dir, abbuf, sebuf
    type(t_partit)         :: partit
    type(t_mesh),   target :: mesh
    type(t_dyn),    target :: dyn
    type(t_tracer), target :: tracers
    type(t_ice),    target :: ice
    type(t_restart)        :: R, R2
    type(refbuf)           :: ref(64)
    logical :: mr
    integer :: nsw, i, j, L, gid, nl, ab, ios, k, ierr, nfail
    integer :: nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF
    integer :: full, rext, se_mode
    real(WP) :: vn, ve, v3, d, maxd
    real(WP), parameter :: SENT = -9.99e30_WP

    call get_environment_variable('FESOM3_MESH_DIR', mesh_dir)
    if (len_trim(mesh_dir) == 0) &
        mesh_dir = '/home/a/a270088/port2/fesom2/tests/data/MESHES/pi'
    call get_environment_variable('FESOM3_RESTARTRT_DIR', out_dir)
    if (len_trim(out_dir) == 0) out_dir = 'restartroundtrip'
    ab = 2
    call get_environment_variable('FESOM3_AB_ORDER', abbuf)
    if (len_trim(abbuf) > 0) then
        read(abbuf, *, iostat=ios) i
        if (ios == 0 .and. (i == 1 .or. i == 2 .or. i == 3)) ab = i
    end if
    se_mode = 0
    call get_environment_variable('FESOM3_RESTARTRT_SE', sebuf)
    if (len_trim(sebuf) > 0) then
        read(sebuf, *, iostat=ios) i
        if (ios == 0 .and. (i == 1 .or. i == 2)) se_mode = i
    end if

    call par_init(partit)
    call set_partition(partit, trim(mesh_dir))
    call read_mesh(mesh, partit, trim(mesh_dir), 50.0_WP, 15.0_WP, -90.0_WP, 360.0_WP, &
                   force_rotation=.true., n_cw_swaps=nsw)
    call compute_geometry(mesh, partit, cartesian=.false.)

    mr = is_multirank(partit)
    call local_dims(mesh, partit, nNodO, nNodL, nEdgeO, nEdgeL, nElemO, nElemL, nElemF)
    nl = mesh%nl

    call restart_init(R, mesh, partit, chunk_horiz=1000)

    ! ---- allocate correctly-shaped REAL model state (node arrays nNodL, element arrays nElemF) -------
    dyn%AB_order = ab
    allocate(dyn%eta_n(nNodL), dyn%d_eta(nNodL), dyn%ssh_rhs_old(nNodL))
    allocate(dyn%w(nl,nNodL), dyn%w_e(nl,nNodL), dyn%w_i(nl,nNodL))
    allocate(dyn%uv(2, nl-1, nElemF))
    allocate(dyn%uv_rhsAB(ab-1, 2, nl-1, nElemF))
    allocate(dyn%work%tke(nl, nNodL))                      ! exercise the optional TKE path (mix_scheme=5)
    if (.not. allocated(mesh%hbar))  allocate(mesh%hbar(nNodL))        ! real(MP) — precision gotcha
    if (.not. allocated(mesh%hnode)) allocate(mesh%hnode(nl-1, nNodL)) ! real(MP)

    tracers%num_tracers = 2
    allocate(tracers%data(2))
    do j = 1, 2
        tracers%data(j)%ID       = j                       ! 1=temp, 2=salt
        tracers%data(j)%AB_order = ab
        allocate(tracers%data(j)%values  (nl-1, nNodL))
        allocate(tracers%data(j)%valuesAB(nl-1, nNodL))
        allocate(tracers%data(j)%valuesold(max(1,ab-1), nl-1, nNodL))
    end do

    ice%num_itracers = 3
    allocate(ice%data(3))
    do j = 1, 3
        allocate(ice%data(j)%values(nNodL))
    end do
    allocate(ice%uice(nNodL), ice%vice(nNodL))
    allocate(ice%thermo%t_skin(nNodL))                     ! carried thermo skin temp (NODE 2-D)
    allocate(ice%work%sigma11(nElemF), ice%work%sigma12(nElemF), ice%work%sigma22(nElemF))
    ! split-explicit carried state, with se_init's shapes (node rings nNodL; element rings nElemL)
    allocate(dyn%se_eta(nNodL, 2), dyn%se_ubt(2, nElemL, 0:2))

    ! ---- zero-init the WHOLE arrays (halo + eXDim start finite at 0 = the fresh-allocate baseline) -----
    dyn%eta_n = 0; dyn%d_eta = 0; dyn%ssh_rhs_old = 0
    dyn%w = 0; dyn%w_e = 0; dyn%w_i = 0; dyn%uv = 0; dyn%uv_rhsAB = 0; dyn%work%tke = 0
    mesh%hbar = 0; mesh%hnode = 0
    do j = 1, 2
        tracers%data(j)%values = 0; tracers%data(j)%valuesAB = 0; tracers%data(j)%valuesold = 0
    end do
    do j = 1, 3
        ice%data(j)%values = 0
    end do
    ice%uice = 0; ice%vice = 0; ice%thermo%t_skin = 0
    ice%work%sigma11 = 0; ice%work%sigma12 = 0; ice%work%sigma22 = 0
    dyn%se_eta = 0; dyn%se_ubt = 0

    ! ---- fill OWNED nodes (value = g for 2-D ; g + 0.5*L for 3-D) -------------------------------------
    do i = 1, nNodO
        gid = i; if (mr) gid = partit%myList_nod2D(i)
        vn  = real(gid, WP)
        dyn%eta_n(i)          = vn
        dyn%d_eta(i)          = vn
        dyn%ssh_rhs_old(i)    = vn
        mesh%hbar(i)          = real(gid, MP)
        ice%data(1)%values(i) = vn; ice%data(2)%values(i) = vn; ice%data(3)%values(i) = vn
        ice%uice(i)           = vn; ice%vice(i)           = vn
        ice%thermo%t_skin(i)  = vn
        dyn%se_eta(i,1)       = vn + 0.25_WP; dyn%se_eta(i,2) = vn + 0.75_WP   ! distinct ring levels
        do L = 1, nl-1
            v3 = vn + 0.5_WP*real(L, WP)
            mesh%hnode(L,i)               = real(gid, MP) + 0.5_MP*real(L, MP)
            tracers%data(1)%values(L,i)   = v3; tracers%data(2)%values(L,i)   = v3
            tracers%data(1)%valuesAB(L,i) = v3; tracers%data(2)%valuesAB(L,i) = v3
            tracers%data(1)%valuesold(:,L,i) = v3; tracers%data(2)%valuesold(:,L,i) = v3
        end do
        do L = 1, nl
            v3 = vn + 0.5_WP*real(L, WP)
            dyn%w(L,i) = v3; dyn%w_e(L,i) = v3; dyn%w_i(L,i) = v3; dyn%work%tke(L,i) = v3
        end do
    end do

    ! ---- fill OWNED elements (value = g for 2-D ; g + 0.5*L for 3-D) ----------------------------------
    do i = 1, nElemO
        gid = i; if (mr) gid = partit%myList_elem2D(i)
        ve  = real(gid, WP)
        ice%work%sigma11(i) = ve; ice%work%sigma12(i) = ve; ice%work%sigma22(i) = ve
        do L = 0, 2                                                 ! distinct component / ring level
            dyn%se_ubt(1,i,L) = ve + 0.125_WP + 0.25_WP*real(L, WP)
            dyn%se_ubt(2,i,L) = ve + 0.25_WP  + 0.25_WP*real(L, WP)
        end do
        do L = 1, nl-1
            v3 = ve + 0.5_WP*real(L, WP)
            dyn%uv(:,L,i)        = v3
            dyn%uv_rhsAB(:,:,L,i) = v3
        end do
    end do

    dyn%use_ssh_se = (se_mode == 1)
    call restart_register_state(R, dyn, tracers, ice, mesh, mix_scheme=5)

    ! ===== (2) write a checkpoint from the pristine OWNED values =======================================
    call restart_write(R, trim(out_dir), 2000, 1, 3600.0_real64, globalstep=1)
    call restart_finalize(R)

    ! ===== SE read pre-check (FESOM3_RESTARTRT_SE=2): the checkpoint above has no SE fields; reading it
    ! with them registered must abort inside restart_read before any field is read.
    if (se_mode == 2) then
        dyn%use_ssh_se = .true.
        call restart_init(R2, mesh, partit, chunk_horiz=1000)
        call restart_register_state(R2, dyn, tracers, ice, mesh, mix_scheme=5)
        if (partit%mype == 0) write(*,'(a,i0,a,i0,a)') 'fesom_restartroundtrip: SE pre-check: wrote ', R%nf, &
            ' fields without SE, reading with ', R2%nf, ' registered (expect an abort)'
        call restart_read(R2, trim(out_dir), mesh, partit, &
                          clock_year=2000, clock_day=1, clock_time_sec=3600.0_real64)
        if (partit%mype == 0) write(*,'(a)') 'SE PRECHECK FAIL: restart_read did not abort'
        error stop 2
    end if

    ! ===== (3) halo-exchange the ORIGINALS so their halos are owner-derived (reference baseline) =======
    call restart_halo_exchange_all(R, partit)

    ! ===== (4) capture the REFERENCE: the full local array of every registered field ==================
    do k = 1, R%nf
        if (R%f(k)%ndim == 2) then
            if (R%f(k)%mp_src) then; ref(k)%a2 = real(R%f(k)%pmp2d, WP)
            else;                    ref(k)%a2 = R%f(k)%p2d; end if
        else
            if (R%f(k)%mp_src) then; ref(k)%a3 = real(R%f(k)%pmp3d, WP)
            else;                    ref(k)%a3 = R%f(k)%p3d; end if
        end if
    end do

    ! ===== (5) CORRUPT: reconstructed region -> sentinel ; un-owned tail -> 0 (fresh-allocate) =========
    do k = 1, R%nf
        associate(f => R%f(k))
        ! the array's own extent: nNodL / nElemF for the model fields, nElemL for the SE element rings
        if (f%ndim == 2) then
            if (f%mp_src) then; full = size(f%pmp2d);   else; full = size(f%p2d);   end if
        else
            if (f%mp_src) then; full = size(f%pmp3d,2); else; full = size(f%p3d,2); end if
        end if
        select case (f%halo)
        case (RESTART_HALO_NODE, RESTART_HALO_ELEM_FULL); rext = full      ! whole array reconstructed
        case (RESTART_HALO_ELEM);                         rext = nElemL    ! owned + eDim
        case default                                                       ! RESTART_HALO_NONE
            rext = merge(nNodO, nElemO, f%entity == DECOMP_NODE)           ! owned only
        end select
        if (f%ndim == 2) then
            if (f%mp_src) then
                f%pmp2d(1:rext) = real(SENT, MP); if (rext < full) f%pmp2d(rext+1:full) = 0.0_MP
            else
                f%p2d(1:rext)   = SENT;           if (rext < full) f%p2d(rext+1:full)   = 0.0_WP
            end if
        else
            if (f%mp_src) then
                f%pmp3d(:,1:rext) = real(SENT, MP); if (rext < full) f%pmp3d(:,rext+1:full) = 0.0_MP
            else
                f%p3d(:,1:rext)   = SENT;           if (rext < full) f%p3d(:,rext+1:full)   = 0.0_WP
            end if
        end if
        end associate
    end do

    ! ===== (6) read the checkpoint back (gather owned + write back + halo exchange) ====================
    call restart_read(R, trim(out_dir), mesh, partit, &
                      clock_year=2000, clock_day=1, clock_time_sec=3600.0_real64)

    ! ===== (7) compare each live array to its reference over the FULL local extent ====================
    if (partit%mype == 0) write(*,'(a)') '--- per-field max|Δ| (live vs reference, full local extent) ---'
    nfail = 0; maxd = 0.0_WP
    do k = 1, R%nf
        if (R%f(k)%ndim == 2) then
            if (R%f(k)%mp_src) then; d = maxval(abs(real(R%f(k)%pmp2d, WP) - ref(k)%a2))
            else;                    d = maxval(abs(R%f(k)%p2d        - ref(k)%a2)); end if
        else
            if (R%f(k)%mp_src) then; d = maxval(abs(real(R%f(k)%pmp3d, WP) - ref(k)%a3))
            else;                    d = maxval(abs(R%f(k)%p3d        - ref(k)%a3)); end if
        end if
        if (mr) call MPI_Allreduce(MPI_IN_PLACE, d, 1, MPI_WP, MPI_MAX, partit%MPI_COMM_FESOM, ierr)
        if (partit%mype == 0) &
            write(*,'(a,a16,a,i1,a,i1,a,es12.5)') '  ', adjustr(R%f(k)%name(1:16)), &
                '  ndim=', R%f(k)%ndim, ' halo=', R%f(k)%halo, '  max|Δ|=', d
        if (d > 0.0_WP) nfail = nfail + 1
        maxd = max(maxd, d)
    end do

    if (partit%mype == 0) then
        write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0)') 'fesom_restartroundtrip: nod2D=', mesh%nod2D, &
            ' elem2D=', mesh%elem2D, ' nl=', nl, ' AB_order=', ab, ' fields=', R%nf
        write(*,'(a,es12.5,a,i0,a)') 'global max|Δ| over all fields = ', maxd, &
            '   (', nfail, ' field(s) nonzero)'
        if (nfail == 0) then
            write(*,'(a)') 'ROUNDTRIP OK (write -> corrupt -> read reproduced every field max|Δ|=0)'
        else
            write(*,'(a)') 'ROUNDTRIP FAIL'
        end if
    end if
    call MPI_Barrier(partit%MPI_COMM_FESOM, ierr)
    if (nfail /= 0) error stop 1
    call par_ex(partit%MPI_COMM_FESOM, partit%mype)
end program fesom_restartroundtrip
