program test_redi_kmin
    ! init_Redi_GM, the Redi_Ktaper blend in the tapered zones (oce_fer_gm.F90):
    !   Ki = Ki_interior*sqrt(tap) + kmin*(1 - sqrt(tap)),
    ! kmin = max(fer_scal*Redi_Kmin, K_GM_min) with scaling_resolution (a fixed Redi_Kmin, as
    ! FESOM2, without). The drivers' work_core GM + Redi parameters on the mesh read from
    ! FESOM3_MESH_DIR (default the NEMO-mixed cut-out, 0.25 deg, so fer_scal < 1 there) at 1 or N
    ! ranks; flat levels (zbar_3d_n = zbar), a uniform N^2. Checked on every owned wet level:
    !   K1 fer_scal = min(sqrt(2*area/(100 km)^2), 1), and fer_scal < 1 on some node
    !   K2 tap = 1: Ki = max(fer_scal*Redi_Kmax, K_GM_min)*0.5*(zs(nz)+zs(nz+1)), the interior
    !      value with the GMzexp depth factor zs -- kmin does not enter
    !   K3 tap = 0: Ki = max(fer_scal*Redi_Kmin, K_GM_min) at every level (the scaled kmin)
    !   K4 tap = 1/4: Ki = (Ki(tap=1) + Ki(tap=0))/2
    !   K5 scaling_resolution off: fer_scal = 1 and tap = 0 gives the fixed Redi_Kmin
    use mpi
    use test_util,       only: setup_test_mesh, check, finish, max_all, sum_all
    use mod_precision,   only: WP, MP, WP_full
    use mod_mesh,        only: t_mesh
    use mod_partit,      only: t_partit
    use mod_partitioning, only: par_init
    use mod_part_bounds, only: owned_bounds
    use mod_param_phys,  only: Fer_GM, Redi, Redi_Kmax, Redi_Kmin, Redi_Ktaper, K_hor, &
                               K_GM_max, K_GM_min, K_GM_cm, K_GM_cmin, K_GM_resscalorder, K_GM_bvref, &
                               K_GM_rampmax, K_GM_rampmin, K_GM_Ktaper, &
                               scaling_Ferreira, scaling_Rossby, scaling_resolution, &
                               scaling_FESOM14, scaling_GMzexp, scaling_GINsea, &
                               GMzexp_zref, GMzexp_smin
    use oce_fer_gm,      only: init_Redi_GM
    implicit none

    type(t_mesh), target :: mesh
    type(t_partit)       :: partit
    integer :: nNodO, nNodL, nEdgeO, nElemO, n, nz, nl, nfine
    real(kind=WP), allocatable :: bvfreq(:,:), fer_K(:,:), fer_c(:), fer_scal(:), tap(:,:)
    real(kind=WP), allocatable :: Ki1(:,:), Ki0(:,:), Kiq(:,:), Kif(:,:), zs(:)
    real(kind=WP) :: e1, e2, e3, e4, e5, ex, smax

    call par_init(partit)
    call setup_test_mesh(mesh, partit, '/work/ab0995/a270029/meshes/nemo025mix_cut', cartesian=.false., &
                         partition_np1=.false.)
    call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
    nl = mesh%nl

    ! flat levels: layer thickness from zbar, zbar_3d_n = zbar
    allocate(mesh%hnode_new(nl-1, mesh%nod2D), mesh%zbar_3d_n(nl, mesh%nod2D))
    do n = 1, mesh%nod2D
        do nz = 1, nl-1
            mesh%hnode_new(nz, n) = real(abs(mesh%zbar(nz) - mesh%zbar(nz+1)), WP_full)
        end do
        mesh%zbar_3d_n(:, n) = real(mesh%zbar(:), MP)
    end do

    allocate(bvfreq(nl, mesh%nod2D), fer_K(nl, mesh%nod2D), fer_c(mesh%nod2D), fer_scal(mesh%nod2D))
    allocate(tap(nl-1, mesh%nod2D), Ki1(nl-1, mesh%nod2D), Ki0(nl-1, mesh%nod2D))
    allocate(Kiq(nl-1, mesh%nod2D), Kif(nl-1, mesh%nod2D), zs(nl))
    bvfreq = 1.0e-5_WP

    ! the drivers' work_core GM + Redi block (fesom_lifecycle_native_mr)
    Fer_GM = .true.
    K_GM_max = 1000.0_WP; K_GM_min = 2.0_WP; K_GM_bvref = 1
    K_GM_rampmax = -1.0_WP; K_GM_rampmin = -1.0_WP; K_GM_resscalorder = 2.0_WP
    K_GM_cm = 3.0_WP; K_GM_cmin = 0.1_WP; K_GM_Ktaper = .false.
    scaling_Ferreira = .false.; scaling_Rossby = .false.; scaling_resolution = .true.
    scaling_FESOM14 = .false.; scaling_GMzexp = .true.; scaling_GINsea = .false.
    GMzexp_zref = 500.0_WP; GMzexp_smin = 0.6_WP
    Redi = .true.; Redi_Ktaper = .true.; Redi_Kmax = 0.0_WP; Redi_Kmin = 100.0_WP
    K_hor = 0.0_WP

    call run(1.0_WP, Ki1)
    call run(0.0_WP, Ki0)
    call run(0.25_WP, Kiq)

    e1 = 0.0_WP; e2 = 0.0_WP; e3 = 0.0_WP; e4 = 0.0_WP; nfine = 0; smax = 0.0_WP
    do n = 1, nNodO
        ex = min(sqrt(mesh%area(n)/(100000.0_WP**2)*2.0_WP), 1.0_WP)
        e1 = max(e1, abs(fer_scal(n) - ex))
        if (fer_scal(n) < 1.0_WP) nfine = nfine + 1
        do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n)
            zs(nz) = GMzexp_smin + (1.0_WP - GMzexp_smin)*exp(-abs(mesh%zbar_3d_n(nz, n)/GMzexp_zref))
            zs(nz) = max(min(zs(nz), 1.0_WP), GMzexp_smin)
        end do
        do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
            ex = max(fer_scal(n)*K_GM_max, K_GM_min)*0.5_WP*(zs(nz) + zs(nz+1))
            e2 = max(e2, abs(Ki1(nz, n) - ex)/ex)
            ex = max(fer_scal(n)*Redi_Kmin, K_GM_min)
            e3 = max(e3, abs(Ki0(nz, n) - ex)/ex)
            ex = 0.5_WP*(Ki1(nz, n) + Ki0(nz, n))
            e4 = max(e4, abs(Kiq(nz, n) - ex)/ex)
            smax = max(smax, Ki0(nz, n))
        end do
    end do
    call max_all(e1, partit); call max_all(e2, partit); call max_all(e3, partit); call max_all(e4, partit)
    call max_all(smax, partit)
    call sum_all(nfine, partit)
    if (partit%mype == 0) write(*,'(a,4es10.2,a,i0,a,f7.2)') 'test_redi_kmin: K1-K4 errors', e1, e2, e3, e4, &
        '  nodes with fer_scal < 1: ', nfine, '  max tapered Ki: ', smax
    call check(e1 <= 1.0e-12_WP, 'K1 fer_scal = min(sqrt(2*area)/100 km, 1)', partit%mype)
    call check(nfine > 0, 'K1 fer_scal < 1 on some node (the mesh is finer than 100 km)', partit%mype)
    call check(e2 <= 1.0e-12_WP, 'K2 tap = 1: Ki = interior value, kmin does not enter', partit%mype)
    call check(e3 <= 1.0e-12_WP, 'K3 tap = 0: Ki = max(fer_scal*Redi_Kmin, K_GM_min)', partit%mype)
    call check(smax < Redi_Kmin, 'K3 the scaled kmin is below the fixed Redi_Kmin on this mesh', partit%mype)
    call check(e4 <= 1.0e-12_WP, 'K4 tap = 1/4: Ki = mean of tap = 1 and tap = 0', partit%mype)

    ! K5: without scaling_resolution, fer_scal = 1 and the tapered Ki is the fixed Redi_Kmin
    scaling_resolution = .false.
    call run(0.0_WP, Kif)
    e5 = 0.0_WP
    do n = 1, nNodO
        e5 = max(e5, abs(fer_scal(n) - 1.0_WP))
        do nz = mesh%ulevels_nod2D(n), mesh%nlevels_nod2D(n) - 1
            e5 = max(e5, abs(Kif(nz, n) - Redi_Kmin))
        end do
    end do
    call max_all(e5, partit)
    call check(e5 == 0.0_WP, 'K5 scaling_resolution off: fer_scal = 1, tapered Ki = Redi_Kmin', partit%mype)

    call finish(partit, 'test_redi_kmin')

contains

    subroutine run(t, Ki)
        ! init_Redi_GM with a uniform taper factor t (fer_K enters at 500, as the drivers set it)
        real(kind=WP), intent(in)  :: t
        real(kind=WP), intent(out) :: Ki(:,:)
        fer_K = 500.0_WP; fer_c = 1.0_WP; fer_scal = 0.0_WP
        tap = t; Ki = 0.0_WP
        call init_Redi_GM(mesh, bvfreq, fer_K, fer_c, fer_scal, partit, Ki=Ki, fer_tapfac=tap)
    end subroutine run

end program test_redi_kmin
