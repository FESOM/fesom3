program test_minlayers
    ! The setup-time >= 2-layer column invariant of the mesh walk (mod_mesh_read::
    ! assert_min_layers, called by derive_vertical_bounds on every mesh path):
    !     nlevels_nod2D(n) - ulevels_nod2D(n) >= 2   for every owned column.
    ! It is the precondition of the vertical kernels' row layouts (the 1-layer trap of
    ! adv_tra_ver_upw1; the surface/bottom rows of adv_tra_vert_impl, do_wimpl and the
    ! momentum TDMA), which validate nothing themselves: config-time checks, stateless
    ! kernels (docs/plans/completed/2026-10-02-wsplit-smooth.md, review round 2). Loads pi (minimum
    ! 4 layers: the walk passes inside read_mesh), then shrinks one owned column to ONE
    ! layer and re-walks: the walk must error stop with 'fewer than 2 layers' (ctest
    ! PASS_REGULAR_EXPRESSION); reaching the final write is the failure
    ! (FAIL_REGULAR_EXPRESSION 'did NOT stop'). 1 rank; the collective verdict's MPI sum
    ! runs in every multi-rank read_mesh of the suite.
    use mod_precision,    only: WP
    use mod_mesh,         only: t_mesh
    use mod_partit,       only: t_partit
    use mod_partitioning, only: par_init, par_ex
    use mod_mesh_read,    only: read_mesh, derive_vertical_bounds
    implicit none

    character(len=512) :: mesh_dir
    type(t_partit) :: partit
    type(t_mesh)   :: mesh
    integer :: nsw, n

    call get_environment_variable('FESOM3_MESH_DIR', mesh_dir)
    if (len_trim(mesh_dir) == 0) &
        mesh_dir = '/home/a/a270088/port2/fesom2/tests/data/MESHES/pi'
    call par_init(partit)
    if (partit%npes /= 1) then
        if (partit%mype == 0) write(*,'(a)') 'test_minlayers: requires 1 rank'
        call par_ex(partit%MPI_COMM_FESOM, partit%mype); error stop 1
    end if

    call read_mesh(mesh, partit, trim(mesh_dir), 50.0_WP, 15.0_WP, -90.0_WP, 360.0_WP, &
                   force_rotation=.true., n_cw_swaps=nsw)
    write(*,'(a,i0,a)') 'test_minlayers: pi read, minimum layers per column = ', &
        minval(mesh%nlevels_nod2D - mesh%ulevels_nod2D), ' (the setup walk passed)'

    ! one owned column shrunk to ONE layer: the re-walk must stop (assert_min_layers runs
    ! before assert_bottom_invariant, so this is the message that appears)
    n = 1
    mesh%nlevels_nod2D(n) = mesh%ulevels_nod2D(n) + 1
    call derive_vertical_bounds(mesh, mesh%elem2D, mesh%nod2D, 'test_minlayers', partit=partit)
    write(*,'(a)') 'test_minlayers: derive_vertical_bounds did NOT stop on a 1-layer column'
    call par_ex(partit%MPI_COMM_FESOM, partit%mype)
    error stop 1
end program test_minlayers
