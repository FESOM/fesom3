program test_ssh_ic0
    ! oce_ssh_ic0, the block-Jacobi incomplete Cholesky preconditioner of the SSH CG
    ! (FESOM3_SSH_PRECOND=ic0), on the SSH matrix (init_stiff_mat_ale at dt = 3600 s, the stiffest
    ! step in use) of the mesh read from FESOM3_MESH_DIR (default the NEMO quad cut-out) at 1 or N
    ! ranks. B is the block of the matrix spanned by a rank's owned nodes, L its IC(0) factor:
    !   C1 every pivot is positive without a diagonal shift
    !   C2 the IC(0) property: (L L^T)_ij = B_ij at every (i, j) of the lower pattern of B and on the
    !      diagonal, to roundoff (IC(0) drops the fill only)
    !   C3 the factor uses owned columns only (block Jacobi: no halo column, no communication)
    !   C4 ic0_apply solves L L^T z = r to roundoff (r hashed from the global id)
    !   C5 a preconditioned CG on B z = b (b = B x, x smooth) needs less than half the iterations
    !      with IC(0) than with the diagonal (Jacobi) preconditioner
    !   C6 solve_ssh_ale with FESOM3_SSH_PRECOND=ic0 converges to x for b = A x on the full matrix
    !      (all ranks), as test_ssh_stiff S7 with Jacobi; the ctest cases set the variable, without
    !      it C6 checks the Jacobi path
    use mpi
    use test_util,     only: setup_test_mesh, check, finish, max_all, hval, set_depths
    use mod_precision, only: WP, WP_full
    use mod_mesh,      only: t_mesh
    use mod_partit,    only: t_partit
    use mod_dyn,       only: t_dyn
    use mod_partitioning, only: par_init
    use mod_part_bounds,  only: owned_bounds
    use oce_ssh_rhs,   only: init_stiff_mat_ale
    use oce_ssh_solve, only: solve_ssh_ale
    use oce_ssh_ic0,   only: t_ic0, ic0_factor, ic0_apply
    implicit none
    real(kind=WP), parameter :: dt = 3600.0_WP
    type(t_partit) :: partit
    type(t_mesh), target :: mesh
    type(t_dyn)    :: dyn
    type(t_ic0)    :: f
    integer :: nNodO, nNodL, nEdgeO, nElemO, row, k, p, q, i, j, n_iter, it_ic, it_jac, nhalo
    logical :: ok
    real(kind=WP) :: e2, e4, err, xmax
    real(kind=WP), allocatable :: r(:), z(:), x(:)
    real(kind=WP_full), allocatable :: lrow(:)

    call par_init(partit)
    call setup_test_mesh(mesh, partit, '/work/ab0995/a270029/meshes/nemo025q_cut', cartesian=.false., &
                         partition_np1=.true.)
    call set_depths(mesh, partit)
    call init_stiff_mat_ale(mesh, dt, partit)
    call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
    associate(A => mesh%ssh_stiff)

    call ic0_factor(f, nNodO, A%rowptr_loc, A%colind_loc, A%values, ok)
    e2 = merge(0.0_WP, 1.0_WP, ok .and. f%shift == 0.0_WP)
    call max_all(e2, partit)
    call check(e2 == 0.0_WP, 'C1 IC(0) without a diagonal shift', partit%mype)

    ! C2: (L L^T)_ij for the stored (i, j), j <= i: sum over m of L_im L_jm, row i scattered into lrow
    allocate(lrow(nNodO)); lrow = 0.0_WP_full
    e2 = 0.0_WP; nhalo = 0
    do i = 1, nNodO
        do p = f%ptr(i), f%ptr(i+1) - 1
            lrow(f%col(p)) = f%val(p)
            if (f%col(p) > nNodO) nhalo = nhalo + 1
        end do
        lrow(i) = f%diag(i)
        do k = A%rowptr_loc(i), A%rowptr_loc(i+1) - 1
            j = A%colind_loc(k)
            if (j > i .or. j > nNodO) cycle
            ! sum_m L_im L_jm over the entries of row j (strict lower part and its diagonal)
            err = real(lrow(j)*f%diag(j), WP)
            do q = f%ptr(j), f%ptr(j+1) - 1
                err = err + real(lrow(f%col(q))*f%val(q), WP)
            end do
            e2 = max(e2, abs(err - real(A%values(k), WP))/abs(real(A%values(A%rowptr_loc(i)), WP)))
        end do
        do p = f%ptr(i), f%ptr(i+1) - 1
            lrow(f%col(p)) = 0.0_WP_full
        end do
        lrow(i) = 0.0_WP_full
    end do
    call max_all(e2, partit)
    if (partit%mype == 0) write(*,'(a,es10.2)') 'C2 max |(L L^T)_ij - B_ij| / B_ii on the pattern: ', e2
    call check(e2 < 1.0e-12_WP, 'C2 L L^T = B on the pattern of B', partit%mype)
    e4 = real(nhalo, WP)
    call max_all(e4, partit)
    call check(e4 == 0.0_WP, 'C3 the factor uses owned columns only', partit%mype)

    ! C4: L L^T z = r for r hashed from the global id
    allocate(r(nNodL), z(nNodL), x(nNodL))
    do row = 1, nNodO
        r(row) = hval(gid(row), 1)
    end do
    call ic0_apply(f, r, z)
    call llt_times(z, x)                                  ! x = L L^T z
    e4 = maxval(abs(x(1:nNodO) - r(1:nNodO)))/maxval(abs(r(1:nNodO)))
    call max_all(e4, partit)
    if (partit%mype == 0) write(*,'(a,es10.2)') 'C4 max |L L^T z - r| / max|r|: ', e4
    call check(e4 < 1.0e-12_WP, 'C4 ic0_apply solves L L^T z = r', partit%mype)

    ! C5: PCG on the owned block, IC(0) against Jacobi, b = B x for a smooth x
    do row = 1, nNodL
        x(row) = cos(3.0_WP*mesh%coord_nod2D(2, row))*sin(5.0_WP*mesh%coord_nod2D(1, row)) &
               + 0.3_WP*sin(11.0_WP*mesh%coord_nod2D(2, row))
    end do
    call block_times(x, r)                                ! r = b = B x
    it_ic  = block_pcg(r, .true.)
    it_jac = block_pcg(r, .false.)
    if (partit%mype == 0) write(*,'(a,i0,a,i0,a)') 'C5 block CG to 1e-10: ', it_ic, ' iterations with IC(0), ', &
        it_jac, ' with Jacobi (rank 0)'
    call check(2*it_ic < it_jac, 'C5 IC(0) needs less than half the Jacobi iterations', partit%mype)
    end associate

    ! C6: the production solve, preconditioner from FESOM3_SSH_PRECOND
    allocate(dyn%d_eta(nNodL), dyn%ssh_rhs(nNodL))
    dyn%ssh_rhs = 0.0_WP; dyn%d_eta = 0.0_WP
    do row = 1, nNodO
        do k = mesh%ssh_stiff%rowptr_loc(row), mesh%ssh_stiff%rowptr_loc(row+1) - 1
            dyn%ssh_rhs(row) = dyn%ssh_rhs(row) + mesh%ssh_stiff%values(k)*x(mesh%ssh_stiff%colind_loc(k))
        end do
    end do
    dyn%solverinfo%soltol = 1.0e-12_WP
    call solve_ssh_ale(dyn, mesh, n_iter, partit)
    err  = maxval(abs(dyn%d_eta(1:nNodO) - x(1:nNodO)))
    xmax = maxval(abs(x(1:nNodO)))
    call max_all(err, partit); call max_all(xmax, partit)
    err = err/xmax
    if (partit%mype == 0) write(*,'(a,i0,a,es10.2)') 'C6 solve_ssh_ale iterations=', n_iter, ' max rel error: ', err
    call check(n_iter < dyn%solverinfo%maxiter, 'C6 solve_ssh_ale converges', partit%mype)
    call check(err < 1.0e-8_WP, 'C6 solve_ssh_ale solution', partit%mype)

    call finish(partit, 'test_ssh_ic0')

contains

    integer function gid(n)
        integer, intent(in) :: n
        if (partit%npes == 1) then
            gid = n
        else
            gid = partit%myList_nod2D(n)
        end if
    end function

    subroutine llt_times(v, w)
        ! w = L (L^T v) on the owned rows
        real(kind=WP), intent(in)  :: v(:)
        real(kind=WP), intent(out) :: w(:)
        real(kind=WP_full), allocatable :: t(:)
        integer :: ii, pp
        allocate(t(nNodO)); t = 0.0_WP_full
        do ii = 1, nNodO                                  ! t = L^T v: column ii of L^T is row ii of L
            t(ii) = t(ii) + f%diag(ii)*v(ii)
            do pp = f%ptr(ii), f%ptr(ii+1) - 1
                t(f%col(pp)) = t(f%col(pp)) + f%val(pp)*v(ii)
            end do
        end do
        do ii = 1, nNodO                                  ! w = L t
            w(ii) = real(f%diag(ii)*t(ii), WP)
            do pp = f%ptr(ii), f%ptr(ii+1) - 1
                w(ii) = w(ii) + real(f%val(pp)*t(f%col(pp)), WP)
            end do
        end do
    end subroutine

    subroutine block_times(v, w)
        ! w = B v (owned rows and columns only)
        real(kind=WP), intent(in)  :: v(:)
        real(kind=WP), intent(out) :: w(:)
        integer :: ii, kk
        do ii = 1, nNodO
            w(ii) = 0.0_WP
            do kk = mesh%ssh_stiff%rowptr_loc(ii), mesh%ssh_stiff%rowptr_loc(ii+1) - 1
                if (mesh%ssh_stiff%colind_loc(kk) <= nNodO) &
                    w(ii) = w(ii) + real(mesh%ssh_stiff%values(kk), WP)*v(mesh%ssh_stiff%colind_loc(kk))
            end do
        end do
    end subroutine

    integer function block_pcg(b, use_ic)
        ! iterations of a PCG on B y = b from 0 to |r| < 1e-10 |b| (local, the rank's own block)
        real(kind=WP), intent(in) :: b(:)
        logical,       intent(in) :: use_ic
        real(kind=WP), allocatable :: y(:), rr(:), zz(:), pp(:), ap(:)
        real(kind=WP_full) :: rz, rz_new, al, bnorm
        integer :: it
        allocate(y(nNodO), rr(nNodO), zz(nNodO), pp(nNodO), ap(nNodO))
        y = 0.0_WP; rr = b(1:nNodO)
        bnorm = sqrt(sum(real(rr, WP_full)**2))
        call precond(rr, zz, use_ic)
        pp = zz; rz = sum(real(rr, WP_full)*zz)
        block_pcg = 5000
        do it = 1, 5000
            call block_times(pp, ap)
            al = rz/sum(real(pp, WP_full)*ap)
            y  = real(y  + al*pp, WP)
            rr = real(rr - al*ap, WP)
            if (sqrt(sum(real(rr, WP_full)**2)) < 1.0e-10_WP_full*bnorm) then
                block_pcg = it
                exit
            end if
            call precond(rr, zz, use_ic)
            rz_new = sum(real(rr, WP_full)*zz)
            pp = real(zz + (rz_new/rz)*pp, WP)
            rz = rz_new
        end do
    end function

    subroutine precond(rin, zout, use_ic)
        ! IC(0) or the diagonal of B
        real(kind=WP), intent(in)    :: rin(:)
        real(kind=WP), intent(inout) :: zout(:)
        logical,       intent(in)    :: use_ic
        integer :: ii
        if (use_ic) then
            call ic0_apply(f, rin, zout)
        else
            do ii = 1, nNodO
                zout(ii) = rin(ii)/real(mesh%ssh_stiff%values(mesh%ssh_stiff%rowptr_loc(ii)), WP)
            end do
        end if
    end subroutine

end program test_ssh_ic0
