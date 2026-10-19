module mod_precision
    ! Two-tier precision scaffolding (decision D3).
    !   WP = working precision (switchable; the precision under test)
    !   MP = mesh precision = max(WP, single) — geometry/coords/work arrays,
    !        kept at >= single so mixed-precision (FP16) builds don't overflow.
    !   MPI_WP = the MPI datatype matching WP, for halo/reductions.
    !
    ! v1 anchor: WP = MP = 8 (double). The FESOM2 v2.7.3 oracle hardcodes WP=8
    ! with no MP (src/oce_modules.F90:8); MP is the dwarf's scaffolding (D3) and
    ! collapses to WP at double precision, so the anchor build is bit-identical
    ! to FESOM2 while the mixed-precision foundation is in place but unexercised.
    use mpi, only: MPI_DOUBLE_PRECISION, MPI_REAL
    implicit none
    public

#if defined(USE_HALF_PRECISION)
    integer, parameter :: WP = 2            ! half  (FP16; NVHPC only)
#elif defined(USE_SINGLE_PRECISION)
    integer, parameter :: WP = 4            ! single
#else
    integer, parameter :: WP = 8            ! double (anchor; FESOM2 oce_modules.F90:8)
#endif

    integer, parameter :: MP = max(WP, 4)   ! mesh precision: at least single

    ! WP_full: always real64, independent of WP (upstream FESOM2 o_PARAM WP_full,
    ! FESOM/fesom2#992). The single-precision build keeps in WP_full what must not be
    ! narrowed: the forcing time axis, global integrals and their reductions, the CG
    ! scalars, the SSH stiffness accumulator, the tracer tendency accumulators, the
    ! layer thickness hnode/hnode_new and the TKE (CVMix) arithmetic. In a double-
    ! precision build WP_full == WP, so every real(x, WP_full) is the identity and the
    ! DP build compiles to the same arithmetic as before. MPI_WP_FULL is its datatype.
    integer, parameter :: WP_full = 8
    integer, parameter :: MPI_WP_FULL = MPI_DOUBLE_PRECISION

    ! Tracer carry (upstream use_tracer_carry, FESOM/fesom2#1206): in a single-precision
    ! build each write of the tracer state keeps its thickness-weighted rounding residual
    ! in tracers%work%carry and feeds it into the next write. Derived, not a switch.
    logical, parameter :: use_tracer_carry = (WP /= WP_full)
    integer, parameter :: MAX_PATH = 4096   ! max file-path length (oce_modules.F90:9)

    ! MPI datatype matching WP (compile-time parameter; dwarf
    ! gen_modules_partitioning.F90:177-181). FP16 halo is handled separately
    ! and compiled out, so MPI_REAL is a placeholder there.
#if defined(USE_SINGLE_PRECISION) || defined(USE_HALF_PRECISION)
    integer, parameter :: MPI_WP = MPI_REAL
#else
    integer, parameter :: MPI_WP = MPI_DOUBLE_PRECISION
#endif

end module mod_precision
