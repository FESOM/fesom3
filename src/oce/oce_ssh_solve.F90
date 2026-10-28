module oce_ssh_solve
    ! Preconditioned conjugate-gradient solve of the free-surface system
    !       ssh_stiff * d_eta = ssh_rhs
    ! built by oce_ssh_rhs. Run in the timestep after compute_ssh_rhs_ale (FESOM2
    ! oce_ale.F90:3930 -> solve_ssh_ale). Transcribed from FESOM2 v2.7.3:
    !   solve_ssh_ale            (oce_ale.F90:3272-3311) — driver (precond once, then CG)
    !   ssh_solve_preconditioner — diagonal (Jacobi) M^-1 (FESOM2 solver.F90:31-95 uses MITgcm's)
    !   ssh_solve_cg             (solver.F90:98-281)      — the CG iteration
    !
    ! PRECONDITIONER: diagonal (Jacobi), M^-1 = diag(1/a_rr), stored in pr_values(nNodO).
    ! FESOM2 uses the MITgcm one (JGR 102,5753-5766,1997: one symmetrised Jacobi sweep,
    ! off-diagonals K_ri = -0.5*(a_ri/a_rr)/(a_rr + a_ii) on the ssh_stiff pattern); the
    ! diagonal one was faster in FESOM3 runs: M^-1 r is local, so the per-iteration halo
    ! exchange of r is gone (one exchange_nod(pp) + two allreduces per iteration remain).
    ! Not bit-identical to FESOM2 any more (different iterates and iteration count).
    !
    ! CG (standard preconditioned CG; the FESOM2 comment notes it follows Wikipedia,
    ! not the MITgcm beta): solves A x = b for x = d_eta, x0 = the incoming d_eta.
    !   r = b - A x ; z = M^-1 r ; p = z ; rho = r.z
    !   loop: Ap = A p ; al = rho/(p.Ap) ; x += al p ; r -= al Ap ; z = M^-1 r
    !         rho_new = r.z ; if sqrt((r.r)/nod2D) < rtol exit
    !         be = rho_new/rho ; rho = rho_new ; p = z + be p
    !   rtol = soltol*sqrt((b.b)/nod2D), soltol=1e-5.
    !
    ! BIT-IDENTITY NOTES:
    !  - Given byte-identical operands (the M2.6-gated ssh_stiff + ssh_rhs) the CG is a
    !    deterministic recurrence: same matrix+rhs+x0 -> same per-iteration scalars ->
    !    same iteration count (the convergence test fires at the same iter) -> same x.
    !    So the gate reduces to "are A and b byte-identical?" — the L9 transitive pattern.
    !  - REDUCTION FORM matters for the bits. The FESOM2 oracle is ENABLE_OPENMP=OFF and
    !    __openmp_reproducible is NOT defined, so its dot-products compile to the explicit
    !    serial `DO row; s=s+...; END DO` (the !$OMP REDUCTION clause is an inert comment).
    !    This file transcribes that serial DO form for s_old/s_aux/sprod — NOT a sum()
    !    intrinsic (which the compiler may reduce in a different order). The matrix-vector
    !    products DO use the sum() intrinsic over a CSR slice, exactly as FESOM2 — both
    !    codes use sum() there, same flags, byte-identical operands -> byte-identical.
    !  - (historical, with FESOM2's preconditioner) the off-diagonal precond divide needed
    !    a NOVECTOR directive to stay scalar like the oracle: packed divpd != scalar divsd at
    !    ~1 ULP was THE seed of the CORE2 d_eta "CG floor" (L28 -> RESOLVED in L29).
    !
    ! 1-RANK SCOPE (multi-rank lifted at M2.12):
    !  - exchange_nod(pp/x/d_eta) are halo broadcasts -> no-ops at 1-rank,
    !    dropped. The MPI_Allreduce of s_old/s_aux/sprod over 1 rank is the identity (the
    !    local serial sum IS the global sum), dropped. So the CG here is purely local +
    !    deterministic, matching the FESOM2 1-rank reduction order. The new multi-rank
    !    bit-identity risk (the cross-rank dot-product reduction order) is an M2.12 concern.
    !  - rr/zz/pp/App sized nod2D (no eDim halo); CSR via colind_loc/rowptr_loc (local).
    !
    ! MULTI-RANK (M2.12c-2, OPTIONAL partit): arrays rr/zz/pp/App sized owned+halo
    ! (nNodL); owned loops 1..nNodO; exchange_nod(pp/x) in the CG fill the halo COLUMNS
    ! the local mat-vec reads;
    ! the dot-products sum OWNED entries (1..nNodO) then allreduce_sum across ranks
    ! (= the FESOM2 oracle solver.F90). rtol / the convergence test divide by the GLOBAL
    ! nod2D (mesh%nod2D, which is global at npes>1). partit absent OR npes==1 => the
    ! proven 1-rank path VERBATIM (nNodO=nNodL=mesh%nod2D, no exchanges, the local sum
    ! already IS the global sum). The new MR bit-identity risk is the cross-rank reduction
    ! order — allreduce_sum byte-matches the oracle because both use the same OpenMPI +
    ! comm size + op + 8-byte type (deterministic tree, LESSONS L6).
    use mod_precision, only: WP, MP, WP_full
    use mod_mesh,      only: t_mesh
    use mod_dyn,       only: t_dyn
    use mod_partit,    only: t_partit
    use mod_part_bounds, only: owned_bounds, is_multirank
    use mod_halo,      only: exchange_nod, allreduce_sum
    implicit none
    private
    public :: solve_ssh_ale
    ! The zstar exact resume (FESOM3_RESTART_ZSTAR_EXACT, fesom_lifecycle_native_mr) builds the
    ! preconditioner itself, from the cold matrix, before restart_read restores the stiffness values.
    public :: ssh_solve_preconditioner

contains

    !===========================================================================
    subroutine solve_ssh_ale(dynamics, mesh, n_iter, partit)
        ! Build the preconditioner on the first call (unless the caller already built it),
        ! then CG-solve for d_eta.
        ! n_iter (optional, NOT part of FESOM2's signature) returns the CG iteration
        ! count for the driver's non-vacuity diagnostic only.
        type(t_dyn),  intent(inout), target  :: dynamics
        type(t_mesh), intent(inout), target  :: mesh
        integer, intent(out), optional       :: n_iter
        type(t_partit), intent(in), optional :: partit
        logical, save :: lfirst = .true.

        if (.not. allocated(mesh%ssh_stiff%pr_values)) call ssh_solve_preconditioner(dynamics, mesh, partit)
        if (lfirst) call ssh_solve_env_overrides(dynamics, partit)
        call ssh_solve_cg(dynamics%d_eta, dynamics%ssh_rhs, dynamics, mesh, n_iter, partit)
        if (is_multirank(partit)) call exchange_nod(dynamics%d_eta, partit)  ! FESOM2 :3308
        lfirst = .false.
    end subroutine solve_ssh_ale

    !===========================================================================
    subroutine ssh_solve_env_overrides(dynamics, partit)
        ! Test/diagnostic knobs, read once at the first solve (unset => defaults unchanged:
        ! soltol 1e-5, maxiter 2000):
        !   FESOM3_SSH_SOLTOL   relative CG tolerance soltol (rtol = soltol*rms(rhs))
        !   FESOM3_SSH_MAXITER  CG iteration cap
        ! Used by tools/run_tri_regression.sh to solve to roundoff, so that two builds that
        ! differ only at roundoff stay at roundoff after the solve (with 1e-5 the iterate
        ! differences, ~1e-5 of eta, swamp the roundoff).
        type(t_dyn),    intent(inout)        :: dynamics
        type(t_partit), intent(in), optional :: partit
        character(len=64) :: env
        integer :: env_len, ios
        logical :: root

        root = .true.
        if (present(partit)) root = (partit%mype == 0)
        call get_environment_variable('FESOM3_SSH_SOLTOL', env, length=env_len, status=ios)
        if (ios == 0 .and. env_len > 0) then
            read(env, *, iostat=ios) dynamics%solverinfo%soltol
            if (ios /= 0) then
                write(*,'(a)') 'solve_ssh_ale: FESOM3_SSH_SOLTOL: not a real: "'//env(1:min(env_len,64))//'"'
                error stop 1
            end if
            if (root) write(*,'(a,es10.3)') ' solve_ssh_ale: FESOM3_SSH_SOLTOL soltol=', &
                dynamics%solverinfo%soltol
        end if
        call get_environment_variable('FESOM3_SSH_MAXITER', env, length=env_len, status=ios)
        if (ios == 0 .and. env_len > 0) then
            read(env, *, iostat=ios) dynamics%solverinfo%maxiter
            if (ios /= 0) then
                write(*,'(a)') 'solve_ssh_ale: FESOM3_SSH_MAXITER: not an integer: "'//env(1:min(env_len,64))//'"'
                error stop 1
            end if
            if (root) write(*,'(a,i0)') ' solve_ssh_ale: FESOM3_SSH_MAXITER maxiter=', &
                dynamics%solverinfo%maxiter
        end if
    end subroutine ssh_solve_env_overrides

    !===========================================================================
    subroutine ssh_solve_preconditioner(dynamics, mesh, partit)
        ! Diagonal (Jacobi) preconditioner: pr_values(row) = 1/a_rr over the OWNED rows.
        ! M^-1 r is then purely local (no halo columns), so neither the diagonal nor r
        ! needs a halo exchange. Built ONCE and then frozen: under zstar the matrix moves
        ! (update_stiff_mat_ale) but pr_values keeps the diagonal of the first solve, which
        ! is the cold matrix (step 1 adds -dhe = 0).
        type(t_dyn),  intent(inout), target :: dynamics
        type(t_mesh), intent(inout), target :: mesh
        type(t_partit), intent(in), optional :: partit
        integer :: row, n
        integer :: nNodO, nNodL, nEdgeO, nElemO

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)

        associate(ssh_stiff => mesh%ssh_stiff)
        allocate(ssh_stiff%pr_values(nNodO))
        ! the diagonal of A is the first CSR entry of each owned row
        do row = 1, nNodO
            ssh_stiff%pr_values(row) = 1.0_WP/ssh_stiff%values(ssh_stiff%rowptr_loc(row) - ssh_stiff%rowptr_loc(1) + 1)
        end do

        n = nNodL   ! owned+halo: halo columns hold the exchanged owner values
        allocate(dynamics%solverinfo%rr(n),  dynamics%solverinfo%zz(n),  &
                 dynamics%solverinfo%pp(n),  dynamics%solverinfo%App(n))
        dynamics%solverinfo%rr  = 0.0_WP
        dynamics%solverinfo%zz  = 0.0_WP
        dynamics%solverinfo%pp  = 0.0_WP
        dynamics%solverinfo%App = 0.0_WP
        end associate
    end subroutine ssh_solve_preconditioner

    !===========================================================================
    subroutine ssh_solve_cg(x, rhs, dynamics, mesh, n_iter, partit)
        real(kind=WP), intent(inout)         :: x(:)     ! d_eta (in: x0, out: solution)
        real(kind=WP), intent(in)            :: rhs(:)   ! ssh_rhs
        type(t_dyn),  intent(inout), target  :: dynamics
        type(t_mesh), intent(inout), target  :: mesh
        integer, intent(out), optional       :: n_iter
        type(t_partit), intent(in), optional :: partit
        integer                  :: row, iter
        integer                  :: nNodO, nNodL, nEdgeO, nElemO
        ! CG scalars in WP_full (upstream FESOM/fesom2#994). Each is a global reduction over
        ! nod2D or derived from one, and they set the search direction and the stopping test,
        ! so rounding here steers the iteration. The vectors rr/zz/pp/App and the matrix stay
        ! WP: the bandwidth is theirs. The sparse-row sums of A*v accumulate in WP_full and
        ! round once into the WP vector. In double precision every real(., WP_full) is the
        ! identity and this is the previous arithmetic.
        real(kind=WP_full)       :: sprod(2), s_old, s_aux, al, be, rtol
        real(kind=MP), pointer   :: values(:), pr_values(:)
        real(kind=WP), pointer   :: rr(:), zz(:), pp(:), App(:)
        integer,       pointer   :: rptr(:), cind(:)

        call owned_bounds(mesh, nNodO, nNodL, nEdgeO, nElemO, partit)
        values    => mesh%ssh_stiff%values
        pr_values => mesh%ssh_stiff%pr_values
        cind      => mesh%ssh_stiff%colind_loc
        rptr      => mesh%ssh_stiff%rowptr_loc
        rr  => dynamics%solverinfo%rr
        zz  => dynamics%solverinfo%zz
        pp  => dynamics%solverinfo%pp
        App => dynamics%solverinfo%App

        !__________________________________________________________________
        ! working tolerance rtol = soltol*sqrt((b.b)/nod2D)  [nod2D GLOBAL]
        s_old = 0.0_WP_full
        do row = 1, nNodO
            s_old = s_old + real(rhs(row), WP_full)*real(rhs(row), WP_full)
        end do
        if (is_multirank(partit)) call allreduce_sum(s_old, partit)
        rtol = dynamics%solverinfo%soltol*sqrt(s_old/real(mesh%nod2D,WP_full))

        !__________________________________________________________________
        ! r0 = b - A x0  (owned rows; x's halo columns are the prescribed x0 halo);
        ! z0 = M^-1 r0 (diagonal: local) ; search direction pp = z0 ; rho = r0.z0
        s_old = 0.0_WP_full
        do row = 1, nNodO
            rr(row) = real(real(rhs(row), WP_full) - sum(real(values(rptr(row):rptr(row+1)-1), WP_full) &
                           *real(x(cind(rptr(row):rptr(row+1)-1)), WP_full)), WP)
            zz(row) = pr_values(row)*rr(row)
            pp(row) = zz(row)
            s_old   = s_old + real(rr(row), WP_full)*real(zz(row), WP_full)
        end do
        if (is_multirank(partit)) call allreduce_sum(s_old, partit)

        !__________________________________________________________________
        ! iterations
        if (present(n_iter)) n_iter = dynamics%solverinfo%maxiter   ! fallback: no convergence
        do iter = 1, dynamics%solverinfo%maxiter
            if (is_multirank(partit)) call exchange_nod(pp, partit)   ! halo cols for A*pp
            s_aux = 0.0_WP_full
            do row = 1, nNodO
                App(row) = real(sum(real(values(rptr(row):rptr(row+1)-1), WP_full) &
                                    *real(pp(cind(rptr(row):rptr(row+1)-1)), WP_full)), WP)
                s_aux    = s_aux + real(pp(row), WP_full)*real(App(row), WP_full)
            end do
            if (is_multirank(partit)) call allreduce_sum(s_aux, partit)
            al = s_old/s_aux

            ! x += al p ; r -= al Ap ; z = M^-1 r (diagonal: local, no halo exchange)
            sprod(1:2) = 0.0_WP_full
            do row = 1, nNodO
                x(row)   = real(x(row)  + al*pp(row), WP)
                rr(row)  = real(rr(row) - al*App(row), WP)
                zz(row)  = pr_values(row)*rr(row)
                sprod(1) = sprod(1) + real(rr(row), WP_full)*real(zz(row), WP_full)
                sprod(2) = sprod(2) + real(rr(row), WP_full)*real(rr(row), WP_full)
            end do
            if (is_multirank(partit)) call allreduce_sum(sprod, partit)

            if (sqrt(sprod(2)/mesh%nod2D) < rtol) then   ! nod2D GLOBAL
                if (present(n_iter)) n_iter = iter
                exit
            end if

            be    = sprod(1)/s_old
            s_old = sprod(1)
            do row = 1, nNodO
                pp(row) = real(zz(row) + be*pp(row), WP)
            end do
        end do
        if (is_multirank(partit)) call exchange_nod(x, partit)   ! FESOM2 solver.F90:279
    end subroutine ssh_solve_cg

end module oce_ssh_solve
