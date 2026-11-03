module oce_ssh_ic0
    ! Block-Jacobi incomplete Cholesky preconditioner for the SSH CG (FESOM3_SSH_PRECOND=ic0,
    ! oce_ssh_solve). Each rank factors the block of ssh_stiff that its owned nodes span (owned rows,
    ! owned columns; the halo columns are dropped, so applying it needs no communication):
    !   B ~ L L^T,  L lower triangular on the pattern of B (no fill-in, IC(0)).
    ! Row-oriented factorisation: for row i and each lower column k (ascending)
    !   L_ik = (B_ik - sum_{m<k} L_im L_km) / L_kk,   L_ii = sqrt(B_ii - sum_{m<i} L_im^2),
    ! the sums over the pattern of B only. ssh_stiff is symmetric positive definite, but on quads it is
    ! not an M-matrix (positive off-diagonals at the quad diagonals), so IC(0) can meet a non-positive
    ! pivot; a pivot <= 1e-12*B_ii restarts the factorisation with the diagonal scaled by 1 + shift,
    ! shift = 1e-3, 1e-2, 0.1, 1 (Manteuffel 1980). NEMO025 on 1280 ranks and the test meshes need none.
    ! Applying M^-1 = (L L^T)^-1: a forward sweep L y = r and a backward sweep L^T z = y (column
    ! oriented over the rows of L). The factor depends on the partition, so the CG iterates (not the
    ! converged tolerance) differ between rank counts; at a given rank count it is deterministic.
    use mod_precision, only: WP, MP, WP_full
    implicit none
    private
    public :: t_ic0, ic0_factor, ic0_apply

    type t_ic0
        integer :: n = 0                                    ! owned rows
        integer,            allocatable :: ptr(:), col(:)   ! strict lower part of row i: ptr(i):ptr(i+1)-1,
        real(kind=WP_full), allocatable :: val(:)           ! columns ascending
        real(kind=WP_full), allocatable :: diag(:)          ! L_ii
        real(kind=WP_full), allocatable :: y(:)             ! work vector of ic0_apply
        real(kind=WP_full)              :: shift = 0.0_WP_full  ! diagonal shift that was needed
    end type t_ic0

contains

    !===========================================================================
    subroutine ic0_factor(f, n, rowptr, colind, values, ok)
        ! IC(0) of the block of the CSR matrix (rowptr, colind, values) spanned by rows and columns
        ! 1..n, indexed as the CG reads ssh_stiff (row i is values(rowptr(i):rowptr(i+1)-1) with the
        ! local columns colind(rowptr(i):rowptr(i+1)-1)); columns > n (the halo) are dropped.
        ! ok = .false. if even the largest shift leaves a non-positive pivot.
        type(t_ic0),   intent(inout) :: f
        integer,       intent(in)    :: n, rowptr(:), colind(:)
        real(kind=MP), intent(in)    :: values(:)
        logical,       intent(out)   :: ok
        real(kind=WP_full), parameter :: shifts(5) = (/ 0.0_WP_full, 1.0e-3_WP_full, 1.0e-2_WP_full, &
                                                        1.0e-1_WP_full, 1.0_WP_full /)
        integer :: i, j, p, q, k, m, ishift
        integer, allocatable :: w(:)
        real(kind=WP_full), allocatable :: bdiag(:)
        real(kind=WP_full) :: d, sfac

        if (allocated(f%ptr)) deallocate(f%ptr, f%col, f%val, f%diag, f%y)
        f%n = n
        allocate(f%ptr(n+1), bdiag(n), f%diag(n), f%y(n), w(n))
        f%ptr(1) = 1
        do i = 1, n
            k = 0
            do p = rowptr(i), rowptr(i+1) - 1
                if (colind(p) < i) k = k + 1
            end do
            f%ptr(i+1) = f%ptr(i) + k
        end do
        allocate(f%col(f%ptr(n+1) - 1), f%val(f%ptr(n+1) - 1))
        w = 0
        ok = .false.
        do ishift = 1, size(shifts)
            sfac = 1.0_WP_full + shifts(ishift)
            ! load B: the strict lower entries, insertion-sorted by column, and the diagonal
            do i = 1, n
                q = f%ptr(i) - 1
                do p = rowptr(i), rowptr(i+1) - 1
                    j = colind(p)
                    if (j == i) then
                        bdiag(i) = real(values(p), WP_full)
                    else if (j < i) then
                        q = q + 1
                        k = q
                        do while (k > f%ptr(i))
                            if (f%col(k-1) <= j) exit
                            f%col(k) = f%col(k-1); f%val(k) = f%val(k-1)
                            k = k - 1
                        end do
                        f%col(k) = j; f%val(k) = real(values(p), WP_full)
                    end if
                end do
            end do
            ok = .true.
            do i = 1, n
                do p = f%ptr(i), f%ptr(i+1) - 1
                    w(f%col(p)) = p                              ! where L_i,col sits in row i
                end do
                d = sfac*bdiag(i)
                do p = f%ptr(i), f%ptr(i+1) - 1
                    k = f%col(p)
                    do q = f%ptr(k), f%ptr(k+1) - 1              ! L_km, m < k
                        m = f%col(q)
                        if (w(m) > 0) f%val(p) = f%val(p) - f%val(w(m))*f%val(q)
                    end do
                    f%val(p) = f%val(p)/f%diag(k)
                    d = d - f%val(p)**2
                end do
                do p = f%ptr(i), f%ptr(i+1) - 1
                    w(f%col(p)) = 0
                end do
                if (d <= 1.0e-12_WP_full*sfac*bdiag(i)) then
                    ok = .false.
                    exit
                end if
                f%diag(i) = sqrt(d)
            end do
            if (ok) then
                f%shift = shifts(ishift)
                exit
            end if
        end do
        deallocate(w, bdiag)
    end subroutine ic0_factor

    !===========================================================================
    subroutine ic0_apply(f, r, z)
        ! z = (L L^T)^-1 r on the f%n owned rows
        type(t_ic0),   intent(inout) :: f
        real(kind=WP), intent(in)    :: r(:)
        real(kind=WP), intent(inout) :: z(:)
        integer :: i, p
        real(kind=WP_full) :: s
        do i = 1, f%n                                            ! L y = r
            s = real(r(i), WP_full)
            do p = f%ptr(i), f%ptr(i+1) - 1
                s = s - f%val(p)*f%y(f%col(p))
            end do
            f%y(i) = s/f%diag(i)
        end do
        do i = f%n, 1, -1                                        ! L^T z = y
            f%y(i) = f%y(i)/f%diag(i)
            do p = f%ptr(i), f%ptr(i+1) - 1
                f%y(f%col(p)) = f%y(f%col(p)) - f%val(p)*f%y(i)
            end do
        end do
        z(1:f%n) = real(f%y(1:f%n), WP)
    end subroutine ic0_apply

end module oce_ssh_ic0
