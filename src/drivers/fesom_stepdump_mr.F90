program fesom_stepdump_mr
    ! M2.12c-1/c-2 MULTI-RANK dynamics byte-gate driver. Runs on dist_<NP> (NP>=1) and
    ! drives the lifted dynamics chain THROUGH the SSH solve (compute_ssh_rhs_ale ->
    ! init_stiff_mat_ale -> solve_ssh_ale) on the SAME analytic state as the 1-rank step
    ! gate (and the FESOM2 oracle src/fesom_step_dump.F90 multi-rank branch), then dumps
    ! the per-rank probe fields density / pressure / bvfreq / Kv / ssh_rhs / d_eta. The
    ! gate tools/run_stepdyn_gate_multirank.sh compares FESOM3 dist_NP vs FESOM2 dist_NP by
    ! GLOBAL probe id (mod_dump is gid-keyed, per-rank <prefix>.<mype5>) for max|delta|=0 —
    ! the L8 same-partition rule.
    !
    ! This exercises EVERY pre-SSH halo exchange (compute_vel_nodes exchange_nod(UVnode),
    ! momentum_adv_scalar exchange_nod(UVnode_rhs), visc_filt_bidiff exchange_elem(U_c/V_c),
    ! compute_ssh_rhs_ale exchange_nod(ssh_rhs), smooth_nod per-sweep exchange) AND the SSH
    ! solve (c-2): the preconditioner exchange_nod(diag_values), the CG per-iter
    ! exchange_nod(pp/rr) + allreduce_sum dot-products. It calls the kernels DIRECTLY (not
    ! via step_oce) and stops at d_eta (the post-solve ALE update/tracers are M2.12c-3).
    !
    ! Inputs are prescribed at owned+halo from the (geometry-gated) rotated coordinates:
    ! node fields directly; the element velocity at owned elements then exchange_elem_full
    ! (halo elements have no local elem2D_nodes), exactly as the M2.12b advhordump_mr driver.
    !
    !   FESOM3_MESH_DIR   mesh dir (must contain dist_<NP>/)
    !   FESOM_DUMP_FILE   node dump prefix (mod_dump; per rank <prefix>.<mype5>)
    use mpi
    use mod_precision,    only: WP, MP
    use mod_constants,    only: density_0
    use mod_param_phys,   only: N2smth_h, alpha, theta
    use mod_param_phys,   only: mix_coeff_PP, A_ver, K_ver, Kv0_const
    use mod_param_phys,   only: use_instabmix, instabmix_kv, use_momix, use_windmix
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init, par_ex, set_partition
    use mod_mesh_read,    only: read_mesh
    use mod_mesh_areas,   only: compute_geometry
    use mod_halo,         only: exchange_elem_full
    use mod_dyn,          only: t_dyn
    use mod_tracer,       only: t_tracer
    use oce_ale,          only: compute_vel_nodes
    use oce_pressure_bv,  only: pressure_bv
    use oce_pgf,          only: pressure_force_4_linfs_fullcell
    use oce_ale_mixing_pp,only: oce_mixing_pp
    use oce_mo_conv,      only: mo_convect
    use oce_vert_spline,  only: compute_shear2
    use oce_dyn_velrhs,   only: compute_vel_rhs
    use oce_dyn_visc,     only: viscosity_filter
    use oce_dyn_ivertvisc,only: impl_vert_visc_ale
    use oce_ssh_rhs,      only: compute_ssh_rhs_ale, init_stiff_mat_ale
    use oce_ssh_solve,    only: solve_ssh_ale
    use mod_dump,         only: dump_init, dump_finalize, dump_node, dump_node_2d, &
                                DUMP_SUBSTEP_PRESSURE_BV, DUMP_SUBSTEP_MIXING, &
                                DUMP_SUBSTEP_SSH_RHS, DUMP_SUBSTEP_SSH_SOLVE
    implicit none

    ! pi namelist timestep: dt = 86400/36 = 2400 s (the value the 1-rank step gate uses).
    real(kind=WP), parameter :: dt = 86400.0_WP / real(36, WP)

    character(len=512) :: mesh_dir
    type(t_partit)     :: partit
    type(t_mesh)       :: mesh
    type(t_dyn)        :: dyn
    type(t_tracer)     :: tracers
    integer :: nsw, n, nz, nl, nzmin, nzmax, e, tr_num, niter
    integer :: nNodO, nNodL, nEdgeO, nElemO, nElemF
    real(kind=WP) :: lon, lat
    real(kind=MP) :: zbar_srf, zbar_bot
    real(kind=WP), allocatable :: Ki(:,:), stress_surf(:,:)

    call get_environment_variable('FESOM3_MESH_DIR', mesh_dir)
    if (len_trim(mesh_dir) == 0) &
        mesh_dir = '/home/a/a270088/port2/fesom2/tests/data/MESHES/pi'

    call par_init(partit)
    call set_partition(partit, trim(mesh_dir))
    call read_mesh(mesh, partit, trim(mesh_dir), 50.0_WP, 15.0_WP, -90.0_WP, 360.0_WP, &
                   force_rotation=.true., n_cw_swaps=nsw)
    call compute_geometry(mesh, partit, cartesian=.false.)
    nl = mesh%nl

    if (partit%npes == 1) then
        nNodO = mesh%nod2D;  nNodL = mesh%nod2D
        nEdgeO = mesh%edge2D; nElemO = mesh%elem2D; nElemF = mesh%elem2D
    else
        nNodO  = partit%myDim_nod2D
        nNodL  = partit%myDim_nod2D + partit%eDim_nod2D
        nEdgeO = partit%myDim_edge2D
        nElemO = partit%myDim_elem2D
        nElemF = partit%myDim_elem2D + partit%eDim_elem2D + partit%eXDim_elem2D
    end if

    !===========================================================================
    ! ALE depth/thickness state (linfs full cells; pure functions of the halo-filled
    ! nlevels, built over the full local range — as in fesom_advhordump_mr).
    allocate(mesh%hnode(nl-1, nNodL)); mesh%hnode = 0.0_MP
    do n = 1, nNodL
        if (mesh%nlevels_nod2D(n) <= 0) cycle
        do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n)-1
            mesh%hnode(nz, n) = mesh%zbar(nz) - mesh%zbar(nz+1)
        end do
    end do
    allocate(mesh%helem(nl-1, nElemF)); mesh%helem = 0.0_MP
    allocate(mesh%zbar_e_bot(nElemF)); mesh%zbar_e_bot = 0.0_MP
    do e = 1, nElemF
        if (mesh%nlevels(e) <= 0) cycle
        do nz = mesh%ulevels(e), mesh%nlevels(e)-1
            mesh%helem(nz, e) = mesh%zbar(nz) - mesh%zbar(nz+1)
        end do
        mesh%zbar_e_bot(e) = mesh%zbar(mesh%nlevels(e))
    end do
    allocate(mesh%zbar_3d_n(nl, nNodL), mesh%Z_3d_n(nl-1, nNodL))
    mesh%zbar_3d_n = 0.0_MP; mesh%Z_3d_n = 0.0_MP
    do n = 1, nNodL
        if (mesh%nlevels_nod2D(n) <= 0) cycle
        nzmin = mesh%ulevels_nod2D(n); nzmax = mesh%nlevels_nod2D(n)
        zbar_srf = mesh%zbar(nzmin); zbar_bot = mesh%zbar(nzmax)
        mesh%zbar_3d_n(1:nzmin-1, n)       = mesh%zbar(1:nzmin-1)
        mesh%zbar_3d_n(nzmin, n)           = zbar_srf
        mesh%zbar_3d_n(nzmin+1:nzmax-1, n) = mesh%zbar(nzmin+1:nzmax-1)
        mesh%zbar_3d_n(nzmax, n)           = zbar_bot
        mesh%Z_3d_n(1:nzmin-1, n)          = mesh%Z(1:nzmin-1)
        mesh%Z_3d_n(nzmin, n)              = mesh%zbar_3d_n(nzmin,n)   + (mesh%zbar_3d_n(nzmin+1,n)-zbar_srf)/2
        mesh%Z_3d_n(nzmin+1:nzmax-2, n)    = mesh%Z(nzmin+1:nzmax-2)
        mesh%Z_3d_n(nzmax-1, n)            = mesh%zbar_3d_n(nzmax-1,n) + (zbar_bot-mesh%zbar_3d_n(nzmax-1,n))/2
    end do
    allocate(mesh%hbar(nNodL), mesh%hbar_old(nNodL), mesh%dhe(nElemF))
    allocate(mesh%hnode_new(nl-1, nNodL))
    mesh%hbar_old = 0.0_MP; mesh%dhe = 0.0_MP
    mesh%hnode_new = mesh%hnode

    !===========================================================================
    ! dynamics state (local sizes; elements full halo, nodes owned+halo).
    allocate(dyn%uv(2, nl-1, nElemF), dyn%uv_rhs(2, nl-1, nElemF))
    allocate(dyn%uv_rhsAB(1, 2, nl-1, nElemF))
    allocate(dyn%uvnode(2, nl-1, nNodL))
    allocate(dyn%eta_n(nNodL), dyn%d_eta(nNodL))
    allocate(dyn%ssh_rhs(nNodL), dyn%ssh_rhs_old(nNodL))
    allocate(dyn%w(nl, nNodL), dyn%w_e(nl, nNodL), dyn%w_i(nl, nNodL))
    allocate(dyn%cfl_z(nl, nNodL))
    allocate(dyn%work%density_ref(nl-1, nNodL), dyn%work%density_m_rho0(nl-1, nNodL))
    allocate(dyn%work%hpressure(nl, nNodL), dyn%work%bvfreq(nl, nNodL), dyn%work%bvfreq_raw(nl, nNodL), dyn%work%shear2(nl, nNodL))
    allocate(dyn%work%pgf_x(nl-1, nElemF), dyn%work%pgf_y(nl-1, nElemF))
    allocate(dyn%work%u_c(nl-1, nElemF), dyn%work%v_c(nl-1, nElemF))
    allocate(dyn%work%uvnode_rhs(2, nl-1, nNodL))
    allocate(dyn%work%vorticity(nl-1, nNodL))   ! momadv_opt==1 (vector-invariant) scratch
    dyn%work%vorticity = 0.0_WP
    allocate(dyn%work%Kv(nl, nNodL), dyn%work%Av(nl, nElemF))
    dyn%uv = 0.0_WP; dyn%uv_rhs = 0.0_WP; dyn%uv_rhsAB = 0.0_WP; dyn%uvnode = 0.0_WP
    dyn%eta_n = 0.0_WP; dyn%d_eta = 0.0_WP; dyn%ssh_rhs = 0.0_WP; dyn%ssh_rhs_old = 0.0_WP
    dyn%w = 0.0_WP; dyn%w_e = 0.0_WP; dyn%w_i = 0.0_WP; dyn%cfl_z = 0.0_WP
    dyn%work%density_m_rho0 = 0.0_WP; dyn%work%hpressure = 0.0_WP; dyn%work%bvfreq = 0.0_WP; dyn%work%bvfreq_raw = 0.0_WP; dyn%work%shear2 = 0.0_WP
    dyn%work%pgf_x = 0.0_WP; dyn%work%pgf_y = 0.0_WP
    dyn%work%u_c = 0.0_WP; dyn%work%v_c = 0.0_WP; dyn%work%uvnode_rhs = 0.0_WP
    dyn%work%Kv = 0.0_WP; dyn%work%Av = 0.0_WP
    dyn%work%density_ref = density_0
    dyn%AB_order      = 2
    dyn%momadv_opt    = 2
    dyn%opt_visc      = 7
    dyn%visc_gamma0   = 0.003_WP
    dyn%visc_gamma1   = 0.1_WP
    dyn%visc_gamma2   = 0.285_WP
    dyn%visc_gamma0_h = 0.0_WP
    dyn%visc_gamma1_h = 0.0_WP
    dyn%use_wsplit    = .false.
    dyn%wsplit_maxcfl = 1.0_WP

    !===========================================================================
    ! tracers (T/S) — only data(1)/data(2)%values needed for pressure_bv.
    tracers%num_tracers = 2
    allocate(tracers%data(2))
    allocate(tracers%data(1)%values(nl-1, nNodL), tracers%data(2)%values(nl-1, nNodL))

    !===========================================================================
    ! prescribe the analytic state at owned+halo (= the FESOM2 oracle on the same
    ! partition, same coords) — node fields directly; element velocity owned + exchange.
    do n = 1, nNodL
        lon = mesh%coord_nod2D(1, n); lat = mesh%coord_nod2D(2, n)
        do nz = 1, nl-1
            tracers%data(1)%values(nz,n) = 12.0_WP + 8.0_WP*cos(lat)*cos(lon) - 0.20_WP*real(nz,WP) &
                        + 8.0_WP*max(0.0_WP, cos(lat)*cos(lon))*min(real(nz,WP),8.0_WP)/8.0_WP
            tracers%data(2)%values(nz,n) = 34.5_WP + 0.5_WP*sin(2.0_WP*lon)*cos(lat) + 0.03_WP*real(nz,WP)
        end do
        dyn%eta_n(n) = 0.5_WP*cos(lat)*sin(lon) + 0.3_WP*sin(2.0_WP*lat)
        do nz = 1, nl
            dyn%w_e(nz,n) = 1.0e-4_WP*sin(2.0_WP*lon)*cos(lat)*cos(0.3_WP*real(nz,WP))
            dyn%w_i(nz,n) = 2.0e-4_WP*sin(lon)*cos(2.0_WP*lat)*cos(0.25_WP*real(nz,WP))
        end do
        mesh%hbar(n) = 0.4_WP*sin(lon)*cos(lat) - 0.2_WP*cos(2.0_WP*lat)
    end do
    dyn%w = dyn%w_e

    allocate(stress_surf(2, nElemF)); stress_surf = 0.0_WP
    do e = 1, nElemO
        lon = mesh%coord_nod2D(1, mesh%elem2D_nodes(1,e))
        lat = mesh%coord_nod2D(2, mesh%elem2D_nodes(1,e))
        do nz = 1, nl-1
            dyn%uv(1,nz,e) =  0.50_WP*cos(lat)*sin(lon)        - 0.005_WP*real(nz,WP)
            dyn%uv(2,nz,e) = -0.40_WP*sin(lat)*cos(2.0_WP*lon) + 0.004_WP*real(nz,WP)
            dyn%uv_rhsAB(1,1,nz,e) =  1.0e6_WP*sin(lon)*cos(lat)        + 1.0e4_WP*real(nz,WP)
            dyn%uv_rhsAB(1,2,nz,e) = -1.0e6_WP*cos(lon)*sin(2.0_WP*lat) - 1.0e4_WP*real(nz,WP)
        end do
        stress_surf(1,e) =  0.10_WP*cos(lat)*sin(lon)
        stress_surf(2,e) = -0.08_WP*sin(lat)*cos(2.0_WP*lon)
    end do
    if (partit%npes > 1) call exchange_elem_full(dyn%uv, partit)

    !===========================================================================
    ! reduced-M2 module config (= the FESOM2 oracle / pi namelist).
    alpha = 1.0_WP; theta = 1.0_WP
    N2smth_h     = .true.
    mix_coeff_PP = 0.01_WP
    A_ver        = 1.0e-4_WP
    K_ver        = 1.0e-5_WP
    Kv0_const    = .true.
    use_instabmix = .true.
    instabmix_kv  = 0.1_WP
    use_momix     = .false.
    use_windmix   = .false.

    !===========================================================================
    ! drive the lifted dynamics chain UP TO compute_ssh_rhs_ale (pass partit).
    call compute_vel_nodes(dyn, mesh, partit)
    call pressure_bv(tracers%data(1)%values, tracers%data(2)%values, dyn%work%density_ref, &
                     mesh, dyn%work%density_m_rho0, dyn%work%hpressure, dyn%work%bvfreq, partit, &
                     bvfreq_raw=dyn%work%bvfreq_raw)
    call pressure_force_4_linfs_fullcell(dyn%work%hpressure, mesh, dyn%work%pgf_x, dyn%work%pgf_y, partit)
    call compute_shear2(dyn, mesh, partit)
    call oce_mixing_pp(dyn, mesh, partit)
    call mo_convect(dyn, mesh, partit)
    call compute_vel_rhs(dyn, mesh, dt, .true., partit)
    call viscosity_filter(dyn%opt_visc, dyn, mesh, dt, partit)
    call impl_vert_visc_ale(dyn, mesh, dt, dyn%work%Av, stress_surf, partit)
    call compute_ssh_rhs_ale(dyn, mesh, partit)

    ! M2.12c-2: SSH stiffness (built ONCE) + the preconditioned CG -> d_eta. x0=d_eta=0
    ! (matches the oracle's step-1 d_eta). The CG dot-products allreduce across ranks.
    call init_stiff_mat_ale(mesh, dt, partit)
    call solve_ssh_ale(dyn, mesh, n_iter=niter, partit=partit)
    write(*,'(a,i0,a,i0)') 'fesom_stepdump_mr: rank ', partit%mype, ' CG iters = ', niter

    !===========================================================================
    ! per-rank dump (mod_dump: gid-keyed probes, the rank owning a probe writes it).
    call dump_init(partit%mype, nNodO, partit%myList_nod2D, nElemO, partit%myList_elem2D)
    call dump_node(DUMP_SUBSTEP_PRESSURE_BV, 1, 'density',  dyn%work%density_m_rho0, mesh%nlevels_nod2D)
    call dump_node(DUMP_SUBSTEP_PRESSURE_BV, 1, 'pressure', dyn%work%hpressure,      mesh%nlevels_nod2D)
    call dump_node(DUMP_SUBSTEP_PRESSURE_BV, 1, 'bvfreq',   dyn%work%bvfreq,         mesh%nlevels_nod2D)
    call dump_node(DUMP_SUBSTEP_MIXING,      1, 'Kv',       dyn%work%Kv,             mesh%nlevels_nod2D)
    call dump_node_2d(DUMP_SUBSTEP_SSH_RHS,  1, 'ssh_rhs',  dyn%ssh_rhs)
    call dump_node_2d(DUMP_SUBSTEP_SSH_SOLVE,1, 'd_eta',    dyn%d_eta)
    call dump_finalize()
    write(*,'(a,i0,a)') 'fesom_stepdump_mr: rank ', partit%mype, ' done.'

    call par_ex(partit%MPI_COMM_FESOM, partit%mype)
end program fesom_stepdump_mr
