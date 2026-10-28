program fesom_tkereplay
    ! M7a-1 CONTROLLED REPLAY gate driver — the make-or-break test that locks the TKE
    ! (single precision: the replay arrays are WP_full, the kind integrate_tke runs in.)
    ! column algebra (integrate_tke + solve_tridiag) in ISOLATION, exactly as the validated
    ! C port's controlled replay did (it caught the 6.6-literal bug, then hit 1.1e-16).
    !
    ! It sets up the CORE2 linfs mesh (so dzw=hnode(nun:nln) is reconstructed byte-exact —
    ! the oracle dumps dztrr but NOT hnode), then INGESTS the oracle tke_dump per-step INPUT
    ! files (normstress/vshear2/bvfreq2/dztrr/tkeold — es24.16 round-trips doubles exactly),
    ! feeds them straight into integrate_tke per gid, and emits the OUTPUT tags (tke/tkeav/
    ! tkekv + lmix/pr/tbpr/tspr/tdif/tdis/twin/tiwf/tbck/ttot) in the oracle's EXACT format
    ! (tke_dump_s<step>_<tag>_rank0.txt). tools/tke_dump_diff.py then gates max|Δ|=0 vs the
    ! oracle outputs, all 3 steps. Steps 2-3 (tke_old≠0) exercise the prognostic solver;
    ! step 1 is the degenerate floor (tke=0 at init).
    !
    ! Pure column algebra on bit-identical inputs: any drift localizes to a single Part via
    ! its intermediate tag (lmix=mxl P1, pr=prandtl P2, tbpr/tspr P3, tdif/tdis P4).
    !
    !   FESOM3_MESH_DIR     CORE2 mesh dir (default core2)
    !   FESOM3_TKE_IN_DIR   oracle reference dump dir (read INPUT tags)
    !   FESOM3_TKE_OUT_DIR  replay output dir          (write OUTPUT tags; default '.')
    !   FESOM_TKE_DUMP_STEPS number of steps to replay (default 3)
    use mpi
    use mod_precision,    only: WP, MP, WP_full
    use mod_constants,    only: density_0, g
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init, par_ex
    use mod_mesh_read,    only: read_mesh
    use mod_mesh_areas,   only: compute_geometry
    use mod_param_phys,   only: tke_c_k, tke_c_eps, tke_cd, tke_alpha, tke_mxl_min, &
                                tke_kappaM_min, tke_kappaM_max, tke_min, tke_surf_min, &
                                tke_mxl_choice, tke_only, tke_use_ubound_dirichlet, &
                                tke_use_lbound_dirichlet, tke_dolangmuir
    use oce_mixing_tke,   only: tke_init, integrate_tke
    implicit none

    ! work_*_tke column constants (dt=1800; density_0=1030/g=9.81 from mod_constants).
    real(kind=WP_full), parameter :: dt_tke = 1800.0_WP_full

    character(len=512) :: mesh_dir, in_dir, out_dir, env_steps
    type(t_partit) :: partit
    type(t_mesh)   :: mesh
    integer :: nl, nsteps, ios, nsw, s, node, nz, nzmin, nzmax, nnod
    ! per-node oracle INPUT buffers (full nl columns; normstress is scalar per node)
    real(kind=WP_full), allocatable :: d_norm(:,:), d_vsh(:,:), d_bvf(:,:), d_dzt(:,:), d_tko(:,:)
    ! per-node OUTPUT buffers (full nl columns)
    real(kind=WP_full), allocatable :: o_tke(:,:), o_km(:,:), o_kh(:,:), o_lmix(:,:), o_pr(:,:), &
                                  o_tbpr(:,:), o_tspr(:,:), o_tdif(:,:), o_tdis(:,:), &
                                  o_twin(:,:), o_tiwf(:,:), o_tbck(:,:), o_ttot(:,:)
    ! column scratch (sized nl; the integrate_tke call uses 1:nlev / 1:nlev+1 slices)
    integer :: nun, nln, nlev
    real(kind=WP_full), allocatable :: zero_c(:)
    real(kind=WP_full), allocatable :: c_dzw(:)
    logical :: ok

    call get_environment_variable('FESOM3_MESH_DIR', mesh_dir)
    if (len_trim(mesh_dir) == 0) &
        mesh_dir = '/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2'
    call get_environment_variable('FESOM3_TKE_IN_DIR', in_dir)
    if (len_trim(in_dir) == 0) in_dir = '.'
    call get_environment_variable('FESOM3_TKE_OUT_DIR', out_dir)
    if (len_trim(out_dir) == 0) out_dir = '.'
    nsteps = 3
    call get_environment_variable('FESOM_TKE_DUMP_STEPS', env_steps, status=ios)
    if (ios == 0 .and. len_trim(env_steps) > 0) read(env_steps, *, iostat=ios) nsteps

    call par_init(partit)
    if (partit%npes /= 1) then
        if (partit%mype == 0) write(*,'(a)') 'fesom_tkereplay: requires 1 rank'
        call par_ex(partit%MPI_COMM_FESOM, partit%mype); error stop 1
    end if

    ! mesh + geometry + linfs hnode (dzw). Same CORE2 setup as fesom_lifecycle.
    call read_mesh(mesh, partit, trim(mesh_dir), 50.0_WP, 15.0_WP, -90.0_WP, 360.0_WP, &
                   force_rotation=.true., n_cw_swaps=nsw)
    call compute_geometry(mesh, partit, cartesian=.false.)
    nl   = mesh%nl
    nnod = mesh%nod2D
    allocate(mesh%hnode(nl-1, nnod)); mesh%hnode = 0.0_MP
    do node = 1, nnod
        do nz = mesh%ulevels_nod2D(node), mesh%nlevels_nod2D(node)-1
            mesh%hnode(nz, node) = mesh%zbar(nz) - mesh%zbar(nz+1)
        end do
    end do
    write(*,'(a,i0,a,i0,a,i0)') 'fesom_tkereplay: nod2D=', nnod, ' nl=', nl, ' steps=', nsteps

    ! store the &param_tke DOUBLES into the column constants (tke_init).
    call tke_init(tke_c_k, tke_c_eps, tke_cd, tke_alpha, tke_mxl_min, tke_kappaM_min, &
                  tke_kappaM_max, tke_min, tke_surf_min, tke_mxl_choice, tke_only, &
                  tke_use_ubound_dirichlet, tke_use_lbound_dirichlet, tke_dolangmuir)

    allocate(d_norm(1,nnod), d_vsh(nl,nnod), d_bvf(nl,nnod), d_dzt(nl,nnod), d_tko(nl,nnod))
    allocate(o_tke(nl,nnod), o_km(nl,nnod), o_kh(nl,nnod), o_lmix(nl,nnod), o_pr(nl,nnod), &
             o_tbpr(nl,nnod), o_tspr(nl,nnod), o_tdif(nl,nnod), o_tdis(nl,nnod), &
             o_twin(nl,nnod), o_tiwf(nl,nnod), o_tbck(nl,nnod), o_ttot(nl,nnod))
    allocate(zero_c(nl), c_dzw(nl))
    zero_c = 0.0_WP_full

    do s = 1, nsteps
        ! --- ingest the oracle INPUT dumps for this step ---
        call read_dump(in_dir, s, 'normstress', 1,  nnod, d_norm, ok); if (.not. ok) error stop 2
        call read_dump(in_dir, s, 'vshear2',    nl, nnod, d_vsh,  ok); if (.not. ok) error stop 2
        call read_dump(in_dir, s, 'bvfreq2',    nl, nnod, d_bvf,  ok); if (.not. ok) error stop 2
        call read_dump(in_dir, s, 'dztrr',      nl, nnod, d_dzt,  ok); if (.not. ok) error stop 2
        call read_dump(in_dir, s, 'tkeold',     nl, nnod, d_tko,  ok); if (.not. ok) error stop 2

        ! --- zero the full-column outputs (mirror the oracle module arrays: nun..nln+1
        !     written by the column call, the rest stays 0 from init) ---
        o_tke = 0.0_WP_full; o_km = 0.0_WP_full; o_kh = 0.0_WP_full; o_lmix = 0.0_WP_full; o_pr = 0.0_WP_full
        o_tbpr = 0.0_WP_full; o_tspr = 0.0_WP_full; o_tdif = 0.0_WP_full; o_tdis = 0.0_WP_full
        o_twin = 0.0_WP_full; o_tiwf = 0.0_WP_full; o_tbck = 0.0_WP_full; o_ttot = 0.0_WP_full

        ! --- per-node column solve on the oracle's bit-identical inputs ---
        do node = 1, nnod
            nln  = mesh%nlevels_nod2D(node)-1
            nun  = mesh%ulevels_nod2D(node)
            nlev = nln-nun+1
            c_dzw(1:nlev) = mesh%hnode(nun:nln, node)
            call integrate_tke( &
                 tke_old      = d_tko(nun:nln+1,node),  &
                 tke_new      = o_tke(nun:nln+1,node),  &
                 KappaM_out   = o_km (nun:nln+1,node),  &
                 KappaH_out   = o_kh (nun:nln+1,node),  &
                 dzw          = c_dzw(1:nlev),          &
                 dzt          = d_dzt(nun:nln+1,node),  &
                 nlev         = nlev,                   &
                 Ssqr         = d_vsh(nun:nln+1,node),  &
                 Nsqr         = d_bvf(nun:nln+1,node),  &
                 tke_Tbpr     = o_tbpr(nun:nln+1,node), &
                 tke_Tspr     = o_tspr(nun:nln+1,node), &
                 tke_Tdif     = o_tdif(nun:nln+1,node), &
                 tke_Tdis     = o_tdis(nun:nln+1,node), &
                 tke_Twin     = o_twin(nun:nln+1,node), &
                 tke_Tiwf     = o_tiwf(nun:nln+1,node), &
                 tke_Tbck     = o_tbck(nun:nln+1,node), &
                 tke_Ttot     = o_ttot(nun:nln+1,node), &
                 tke_Lmix     = o_lmix(nun:nln+1,node), &
                 tke_Pr       = o_pr  (nun:nln+1,node), &
                 tke_plc      = zero_c(1:nlev+1),       &
                 forc_tke_surf= d_norm(1,node),         &
                 E_iw         = zero_c(1:nlev+1),       &
                 dtime        = dt_tke,                 &
                 iw_diss      = zero_c(1:nlev+1),       &
                 forc_rho_surf= 0.0_WP_full,                 &
                 rho_ref      = real(density_0, WP_full), &
                 grav         = real(g, WP_full),       &
                 alpha_c      = zero_c(1:nlev+1))
            ! endpoint zeroing of tke_Av/tke_Kv (the driver does this; tke is NOT zeroed)
            o_km(nln+1,node) = 0.0_WP_full; o_kh(nln+1,node) = 0.0_WP_full
            o_km(nun  ,node) = 0.0_WP_full; o_kh(nun  ,node) = 0.0_WP_full
        end do

        ! --- emit the OUTPUT tags in the oracle's exact format ---
        call write_dump(out_dir, s, 'tke',   nl, nnod, o_tke,  partit)
        call write_dump(out_dir, s, 'tkeav', nl, nnod, o_km,   partit)
        call write_dump(out_dir, s, 'tkekv', nl, nnod, o_kh,   partit)
        call write_dump(out_dir, s, 'lmix',  nl, nnod, o_lmix, partit)
        call write_dump(out_dir, s, 'pr',    nl, nnod, o_pr,   partit)
        call write_dump(out_dir, s, 'tbpr',  nl, nnod, o_tbpr, partit)
        call write_dump(out_dir, s, 'tspr',  nl, nnod, o_tspr, partit)
        call write_dump(out_dir, s, 'tdif',  nl, nnod, o_tdif, partit)
        call write_dump(out_dir, s, 'tdis',  nl, nnod, o_tdis, partit)
        call write_dump(out_dir, s, 'twin',  nl, nnod, o_twin, partit)
        call write_dump(out_dir, s, 'tiwf',  nl, nnod, o_tiwf, partit)
        call write_dump(out_dir, s, 'tbck',  nl, nnod, o_tbck, partit)
        call write_dump(out_dir, s, 'ttot',  nl, nnod, o_ttot, partit)
        write(*,'(a,i0)') 'fesom_tkereplay: replayed step ', s
    end do

    call par_ex(partit%MPI_COMM_FESOM, partit%mype)

contains

    ! Read one tke_dump_s<step>_<tag>_rank0.txt into vals(1:ncomp, 1:nnod), keyed by gid.
    subroutine read_dump(dir, step, tag, ncomp, nnod, vals, ok)
        character(len=*), intent(in)  :: dir, tag
        integer,          intent(in)  :: step, ncomp, nnod
        real(kind=WP_full),    intent(out) :: vals(:,:)
        logical,          intent(out) :: ok
        character(len=640) :: path, hdr
        real(kind=WP_full)      :: row(ncomp)   ! read into a temp, THEN index by gid (don't use a
        integer :: u, iostat, r, gid, c    ! just-read var as a subscript in the same READ)
        write(path,'(a,"/tke_dump_s",i0,"_",a,"_rank0.txt")') trim(dir), step, trim(tag)
        ok = .false.
        open(newunit=u, file=trim(path), status='old', action='read', iostat=iostat)
        if (iostat /= 0) then
            write(*,'(a)') 'fesom_tkereplay: cannot open '//trim(path); return
        end if
        read(u,'(a)', iostat=iostat) hdr           ! skip the "# step=.." header
        if (iostat /= 0) then
            write(*,'(a)') 'fesom_tkereplay: empty '//trim(path); close(u); return
        end if
        vals = 0.0_WP_full
        do r = 1, nnod
            read(u, *, iostat=iostat) gid, (row(c), c=1, ncomp)
            if (iostat /= 0) then
                write(*,'(a,i0,a)') 'fesom_tkereplay: short read at row ', r, ' of '//trim(path)
                close(u); return
            end if
            if (gid < 1 .or. gid > nnod) then
                write(*,'(a,i0)') 'fesom_tkereplay: gid out of range ', gid; close(u); return
            end if
            vals(1:ncomp, gid) = row(1:ncomp)
        end do
        close(u)
        ok = .true.
    end subroutine read_dump

    ! Write vals(1:ncomp, 1:nnod) as gid-keyed es24.16 rows — EXACT mirror of
    ! tke_dump_mod::tke_dump_nod so the files diff byte-for-byte against the oracle.
    subroutine write_dump(dir, step, tag, ncomp, nnod, vals, partit)
        character(len=*), intent(in) :: dir, tag
        integer,          intent(in) :: step, ncomp, nnod
        real(kind=WP_full),    intent(in) :: vals(:,:)
        type(t_partit),   intent(in) :: partit
        character(len=640) :: path
        integer :: u, nn, c, iostat
        write(path,'(a,"/tke_dump_s",i0,"_",a,"_rank",i0,".txt")') &
             trim(dir), step, trim(tag), partit%mype
        open(newunit=u, file=trim(path), status='replace', action='write', iostat=iostat)
        if (iostat /= 0) then
            write(*,'(a)') 'fesom_tkereplay: cannot write '//trim(path); return
        end if
        write(u,'("# step=",i0," tag=",a," rank=",i0," N=",i0," ncomp=",i0)') &
             step, trim(tag), partit%mype, nnod, ncomp
        ! 1-rank: gid == local node index (synthesize_1rank identity) == the oracle's
        ! myList_nod2D(nn). Emit nn directly (myList_nod2D is not populated on this driver's
        ! par_init/read_mesh path, and the replay asserts npes==1).
        do nn = 1, nnod
            write(u,'(i0)', advance='no') nn
            do c = 1, ncomp
                write(u,'(" ",es24.16)', advance='no') vals(c, nn)
            end do
            write(u,'(a)') ''
        end do
        close(u)
    end subroutine write_dump

end program fesom_tkereplay
