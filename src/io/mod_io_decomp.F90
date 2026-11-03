module mod_io_decomp
    ! Canonical-order + distributed-chunk-writer redistribution for FESOM3 Zarr output (M9 Task 1.1).
    !
    ! Goal: take a field laid out in the COMPUTE partition (each rank owns myDim entities, local index
    ! i -> global/canonical id myList(i)) and redistribute it to a CANONICAL chunked layout on a subset
    ! of `n_writers` writer ranks, so the on-disk store is partition-INDEPENDENT (identical at any rank
    ! count) with no single-rank gather. Mirrors the FESOM2 io_gather canonical map (myList_*), but
    ! scatters chunks across writers instead of funnelling to rank 0.
    !
    ! Scheme (per entity dim, uniform chunk size C, nchunks = ceil(N/C)):
    !   - chunks are block-assigned to writer ranks 0..n_writers-1 (contiguous blocks).
    !   - each compute rank maps each owned entity i -> canonical id g=myList(i) -> chunk (g-1)/C ->
    !     destination writer rank; buckets its owned values by destination; ONE MPI_Alltoallv ships
    !     them. Non-writers receive nothing (zero recvcounts).
    !   - the send/recv PLAN (counts, displs, the per-recv-slot target index in the writer's canonical
    !     buffer) is computed ONCE in decomp_init via a gid Alltoallv, then reused for every field.
    !   - 3D loops levels, reusing the 2D plan per level => in-flight memory O(C*nlev) per writer.
    !   - at npes==1 the Alltoallv is a self-copy => the 1-rank identity (one writer holds all in
    !     canonical order) falls out of the same code path.
    !
    ! The writer's canonical buffer spans its OWN chunks fully (size = nchunks_owned*C); positions for
    ! canonical id g land at g - w_base_gid. Positions beyond N (the partial last chunk) stay `fill`
    ! (the only padding; matches mod_io_zarr's chunk padding, so the reader truncates them away).
    use mpi
    use mod_precision, only: WP, MPI_WP, WP_full, MPI_WP_FULL
    use mod_mesh,      only: t_mesh
    use mod_partit,    only: t_partit
    use mod_part_bounds, only: is_multirank
    implicit none
    private

    ! entity selector for the convenience wrapper decomp_init_entity
    integer, parameter, public :: DECOMP_NODE = 1, DECOMP_ELEM = 2, DECOMP_EDGE = 3

    type, public :: t_io_decomp
        integer :: N        = 0          ! global (canonical) entity count
        integer :: C        = 1          ! chunk size along the entity dim
        integer :: nchunks  = 0          ! ceil(N/C)
        integer :: n_writers= 1          ! actual number of writer ranks (<= min(npes, nchunks))
        integer :: npes     = 1, mype = 0, comm = MPI_COMM_NULL
        integer :: myDim    = 0          ! owned entities on this rank
        ! writer role (this rank)
        logical :: is_writer    = .false.
        integer :: w_first_chunk= 0, w_last_chunk = -1   ! 0-based inclusive
        integer :: w_nbuf       = 0      ! canonical buffer length on this writer = nchunks_owned*C
        integer :: w_base_gid   = 0      ! canonical id of buffer slot 1 minus 1 (slot = g - w_base_gid)
        ! send plan (this rank, owned entities) — type-independent, built once
        integer :: send_total = 0, recv_total = 0
        integer, allocatable :: sendcounts(:), sdispls(:)   ! size npes
        integer, allocatable :: recvcounts(:), rdispls(:)   ! size npes
        integer, allocatable :: send_pos(:)                 ! send_pos(i) 0-based slot in sendbuf, i=1..myDim
        integer, allocatable :: recv_target(:)              ! recv_target(k) 1-based slot in writer buffer
    end type t_io_decomp

    public :: decomp_init, decomp_init_entity
    public :: decomp_redistribute, decomp_gather, decomp_is_writer, decomp_writer_chunk_range

    interface decomp_redistribute
        module procedure decomp_redistribute_2d_r, decomp_redistribute_3d_r, &
                         decomp_redistribute_2d_i, decomp_redistribute_3d_i
#if defined(USE_SINGLE_PRECISION)
        ! single precision: the WP_full (real64) restart fields (hnode, ssh_stiff) move without a
        ! round trip through WP; in double precision WP_full == WP and the real variants serve
        module procedure decomp_redistribute_2d_r8, decomp_redistribute_3d_r8
#endif
    end interface decomp_redistribute

    ! Inverse of decomp_redistribute (the read/restore direction): pull canonical writer-buffer
    ! values back to the compute partition's owned entities, reusing the SAME t_io_decomp plan.
    ! Real WP only (restart serializes WP fields); no int variant needed.
    interface decomp_gather
        module procedure decomp_gather_2d_r, decomp_gather_3d_r
#if defined(USE_SINGLE_PRECISION)
        module procedure decomp_gather_2d_r8, decomp_gather_3d_r8
#endif
    end interface decomp_gather

contains

    ! Block-assign chunk c (0-based) to a writer index (0-based). base>=1 (guaranteed by clamp).
    pure integer function chunk_to_writer(c, nchunks, n_writers) result(w)
        integer, intent(in) :: c, nchunks, n_writers
        integer :: base, rem, split
        base  = nchunks / n_writers
        rem   = mod(nchunks, n_writers)
        split = rem * (base + 1)          ! first `rem` writers get base+1 chunks
        if (c < split) then
            w = c / (base + 1)
        else
            w = rem + (c - split) / base
        end if
    end function chunk_to_writer

    ! Core init from RAW arrays (unit-testable without a mesh). myList(1:myDim) = owned canonical ids.
    subroutine decomp_init(D, C, n_writers, N, myList, myDim, comm, mype, npes)
        type(t_io_decomp), intent(out) :: D
        integer, intent(in) :: C, n_writers, N, myDim, comm, mype, npes
        integer, intent(in) :: myList(:)
        integer :: base, rem, cnt, i, g, c0, w, p, ierr
        integer, allocatable :: cursor(:), send_gid(:), recv_gid(:)

        ! clamp chunk size to N so (a) the last chunk isn't absurdly over-padded and (b) callers can
        ! read back D%C as the authoritative chunk size for the matching Zarr array.
        D%N = N; D%C = max(1, min(C, max(1, N))); D%comm = comm; D%mype = mype; D%npes = npes; D%myDim = myDim
        D%nchunks = (N + D%C - 1) / D%C
        D%n_writers = n_writers
        if (D%n_writers <= 0) D%n_writers = min(npes, D%nchunks)
        D%n_writers = max(1, min(D%n_writers, min(npes, D%nchunks)))

        ! this rank's writer role
        base = D%nchunks / D%n_writers
        rem  = mod(D%nchunks, D%n_writers)
        D%is_writer = (mype < D%n_writers)
        if (D%is_writer) then
            if (mype < rem) then
                D%w_first_chunk = mype*(base+1);                 cnt = base + 1
            else
                D%w_first_chunk = rem*(base+1) + (mype-rem)*base; cnt = base
            end if
            D%w_last_chunk = D%w_first_chunk + cnt - 1
            D%w_nbuf       = cnt * D%C
            D%w_base_gid   = D%w_first_chunk * D%C
        else
            D%w_first_chunk = 0; D%w_last_chunk = -1; D%w_nbuf = 0; D%w_base_gid = 0
        end if

        ! ---- send plan: bucket owned entities by destination writer rank ----
        allocate(D%sendcounts(npes), D%sdispls(npes), D%recvcounts(npes), D%rdispls(npes))
        allocate(cursor(npes), D%send_pos(max(1,myDim)))
        D%sendcounts = 0
        do i = 1, myDim
            g  = myList(i)
            c0 = (g - 1) / D%C
            w  = chunk_to_writer(c0, D%nchunks, D%n_writers)    ! writer index == physical rank
            D%sendcounts(w+1) = D%sendcounts(w+1) + 1
        end do
        D%sdispls(1) = 0
        do p = 2, npes
            D%sdispls(p) = D%sdispls(p-1) + D%sendcounts(p-1)
        end do
        D%send_total = sum(D%sendcounts)
        ! per-entity send slot + the parallel gid stream (for the plan exchange)
        allocate(send_gid(max(1,D%send_total)))
        cursor = 0
        do i = 1, myDim
            g  = myList(i)
            c0 = (g - 1) / D%C
            w  = chunk_to_writer(c0, D%nchunks, D%n_writers)
            p  = D%sdispls(w+1) + cursor(w+1)        ! 0-based slot
            D%send_pos(i) = p
            send_gid(p+1) = g
            cursor(w+1) = cursor(w+1) + 1
        end do

        ! ---- recv plan: counts via Alltoall, then exchange gids to learn placement ----
        call MPI_Alltoall(D%sendcounts, 1, MPI_INTEGER, D%recvcounts, 1, MPI_INTEGER, comm, ierr)
        D%rdispls(1) = 0
        do p = 2, npes
            D%rdispls(p) = D%rdispls(p-1) + D%recvcounts(p-1)
        end do
        D%recv_total = sum(D%recvcounts)
        allocate(recv_gid(max(1,D%recv_total)))
        call MPI_Alltoallv(send_gid, D%sendcounts, D%sdispls, MPI_INTEGER, &
                           recv_gid, D%recvcounts, D%rdispls, MPI_INTEGER, comm, ierr)
        ! each received gid -> its slot in this writer's canonical buffer
        allocate(D%recv_target(max(1,D%recv_total)))
        do i = 1, D%recv_total
            D%recv_target(i) = recv_gid(i) - D%w_base_gid     ! 1-based (gid >= w_base_gid+1)
        end do
    end subroutine decomp_init

    ! Convenience wrapper: pull the entity's canonical map from an OPTIONAL partit.
    !   present + npes>1 -> the partition (myList/myDim/comm).
    !   absent  or npes==1 -> the 1-rank identity (myDim=N, myList=1..N, self comm).
    subroutine decomp_init_entity(D, C, n_writers, entity, mesh, partit)
        type(t_io_decomp), intent(out) :: D
        integer,           intent(in)  :: C, n_writers, entity
        type(t_mesh),      intent(in)  :: mesh
        type(t_partit),    intent(in), optional :: partit
        integer :: N, myDim, comm, mype, npes, i
        integer, allocatable :: myList(:)

        select case (entity)
        case (DECOMP_NODE); N = mesh%nod2D
        case (DECOMP_ELEM); N = mesh%elem2D
        case (DECOMP_EDGE); N = mesh%edge2D
        case default;       N = mesh%nod2D
        end select

        if (is_multirank(partit)) then
            comm = partit%MPI_COMM_FESOM; mype = partit%mype; npes = partit%npes
            select case (entity)
            case (DECOMP_NODE); myDim = partit%myDim_nod2D;  myList = partit%myList_nod2D(1:myDim)
            case (DECOMP_ELEM); myDim = partit%myDim_elem2D; myList = partit%myList_elem2D(1:myDim)
            case (DECOMP_EDGE); myDim = partit%myDim_edge2D; myList = partit%myList_edge2D(1:myDim)
            end select
        else
            comm = MPI_COMM_SELF; mype = 0; npes = 1
            if (present(partit)) comm = partit%MPI_COMM_FESOM
            myDim = N
            allocate(myList(N))
            do i = 1, N
                myList(i) = i
            end do
        end if
        call decomp_init(D, C, n_writers, N, myList, myDim, comm, mype, npes)
    end subroutine decomp_init_entity

    logical function decomp_is_writer(D)
        type(t_io_decomp), intent(in) :: D
        decomp_is_writer = D%is_writer
    end function decomp_is_writer

    subroutine decomp_writer_chunk_range(D, first_chunk, last_chunk)
        type(t_io_decomp), intent(in)  :: D
        integer,           intent(out) :: first_chunk, last_chunk
        first_chunk = D%w_first_chunk
        last_chunk  = D%w_last_chunk
    end subroutine decomp_writer_chunk_range

    ! ---- redistribution (real) ----

    subroutine decomp_redistribute_2d_r(D, field_owned, buf_writer, fill)
        type(t_io_decomp), intent(in)  :: D
        real(WP),          intent(in)  :: field_owned(:)   ! 1..myDim (halo beyond ignored)
        real(WP),          intent(out) :: buf_writer(:)    ! 1..w_nbuf
        real(WP),          intent(in)  :: fill
        real(WP), allocatable :: sbuf(:), rbuf(:)
        integer :: i, k, ierr
        allocate(sbuf(max(1,D%send_total)), rbuf(max(1,D%recv_total)))
        do i = 1, D%myDim
            sbuf(D%send_pos(i)+1) = field_owned(i)
        end do
        call MPI_Alltoallv(sbuf, D%sendcounts, D%sdispls, MPI_WP, &
                           rbuf, D%recvcounts, D%rdispls, MPI_WP, D%comm, ierr)
        if (D%w_nbuf > 0) then
            buf_writer(1:D%w_nbuf) = fill
            do k = 1, D%recv_total
                buf_writer(D%recv_target(k)) = rbuf(k)
            end do
        end if
    end subroutine decomp_redistribute_2d_r

    subroutine decomp_redistribute_3d_r(D, field_owned, buf_writer, fill)
        type(t_io_decomp), intent(in)  :: D
        real(WP),          intent(in)  :: field_owned(:,:) ! (nlev, 1..myDim)
        real(WP),          intent(out) :: buf_writer(:,:)  ! (nlev, 1..w_nbuf)
        real(WP),          intent(in)  :: fill
        real(WP), allocatable :: col(:), outrow(:)
        integer :: nlev, L
        nlev = size(field_owned, 1)
        allocate(col(max(1,D%myDim)), outrow(max(1,D%w_nbuf)))
        do L = 1, nlev
            col(1:D%myDim) = field_owned(L, 1:D%myDim)
            call decomp_redistribute_2d_r(D, col, outrow, fill)
            if (D%w_nbuf > 0) buf_writer(L, 1:D%w_nbuf) = outrow(1:D%w_nbuf)
        end do
    end subroutine decomp_redistribute_3d_r

    ! ---- gather (real) — exact transpose of decomp_redistribute, run backwards on the SAME plan ----
    ! redistribute (write):  field(i)->sbuf(send_pos(i)+1) ; Alltoallv(send->recv) ; rbuf(k)->buf(recv_target(k))
    ! gather      (read):    buf(recv_target(k))->rbuf(k) ; Alltoallv(recv->send, counts/displs swapped) ;
    !                        sbuf(send_pos(i)+1)->field(i)
    ! The store holds only canonical OWNED values, so the partial last chunk's pad slots are never read
    ! (recv_target only indexes real entities) — padding is ignored. npes==1 => the Alltoallv is a self-copy.

    subroutine decomp_gather_2d_r(D, buf_writer, field_owned)
        type(t_io_decomp), intent(in)  :: D
        real(WP),          intent(in)  :: buf_writer(:)    ! 1..w_nbuf (canonical writer buffer)
        real(WP),          intent(out) :: field_owned(:)   ! 1..myDim  (compute-partition owned)
        real(WP), allocatable :: sbuf(:), rbuf(:)
        integer :: i, k, ierr
        allocate(sbuf(max(1,D%send_total)), rbuf(max(1,D%recv_total)))
        if (D%w_nbuf > 0) then
            do k = 1, D%recv_total
                rbuf(k) = buf_writer(D%recv_target(k))
            end do
        end if
        ! reversed Alltoallv: recv-side counts become SEND args, send-side counts become RECV args
        call MPI_Alltoallv(rbuf, D%recvcounts, D%rdispls, MPI_WP, &
                           sbuf, D%sendcounts, D%sdispls, MPI_WP, D%comm, ierr)
        do i = 1, D%myDim
            field_owned(i) = sbuf(D%send_pos(i)+1)
        end do
    end subroutine decomp_gather_2d_r

    subroutine decomp_gather_3d_r(D, buf_writer, field_owned)
        type(t_io_decomp), intent(in)  :: D
        real(WP),          intent(in)  :: buf_writer(:,:)  ! (nlev, 1..w_nbuf)
        real(WP),          intent(out) :: field_owned(:,:) ! (nlev, 1..myDim)
        real(WP), allocatable :: col(:), inrow(:)
        integer :: nlev, L
        nlev = size(field_owned, 1)
        allocate(col(max(1,D%myDim)), inrow(max(1,D%w_nbuf)))
        do L = 1, nlev
            if (D%w_nbuf > 0) inrow(1:D%w_nbuf) = buf_writer(L, 1:D%w_nbuf)
            call decomp_gather_2d_r(D, inrow, col)
            field_owned(L, 1:D%myDim) = col(1:D%myDim)
        end do
    end subroutine decomp_gather_3d_r

#if defined(USE_SINGLE_PRECISION)
    ! ---- redistribution and gather (WP_full; single precision only) ----
    ! The real variants above with WP -> WP_full: the same plan, the same slots, real64 values.

    subroutine decomp_redistribute_2d_r8(D, field_owned, buf_writer, fill)
        type(t_io_decomp),  intent(in)  :: D
        real(WP_full),      intent(in)  :: field_owned(:)
        real(WP_full),      intent(out) :: buf_writer(:)
        real(WP_full),      intent(in)  :: fill
        real(WP_full), allocatable :: sbuf(:), rbuf(:)
        integer :: i, k, ierr
        allocate(sbuf(max(1,D%send_total)), rbuf(max(1,D%recv_total)))
        do i = 1, D%myDim
            sbuf(D%send_pos(i)+1) = field_owned(i)
        end do
        call MPI_Alltoallv(sbuf, D%sendcounts, D%sdispls, MPI_WP_FULL, &
                           rbuf, D%recvcounts, D%rdispls, MPI_WP_FULL, D%comm, ierr)
        if (D%w_nbuf > 0) then
            buf_writer(1:D%w_nbuf) = fill
            do k = 1, D%recv_total
                buf_writer(D%recv_target(k)) = rbuf(k)
            end do
        end if
    end subroutine decomp_redistribute_2d_r8

    subroutine decomp_redistribute_3d_r8(D, field_owned, buf_writer, fill)
        type(t_io_decomp),  intent(in)  :: D
        real(WP_full),      intent(in)  :: field_owned(:,:)
        real(WP_full),      intent(out) :: buf_writer(:,:)
        real(WP_full),      intent(in)  :: fill
        real(WP_full), allocatable :: col(:), outrow(:)
        integer :: nlev, L
        nlev = size(field_owned, 1)
        allocate(col(max(1,D%myDim)), outrow(max(1,D%w_nbuf)))
        do L = 1, nlev
            col(1:D%myDim) = field_owned(L, 1:D%myDim)
            call decomp_redistribute_2d_r8(D, col, outrow, fill)
            if (D%w_nbuf > 0) buf_writer(L, 1:D%w_nbuf) = outrow(1:D%w_nbuf)
        end do
    end subroutine decomp_redistribute_3d_r8

    subroutine decomp_gather_2d_r8(D, buf_writer, field_owned)
        type(t_io_decomp),  intent(in)  :: D
        real(WP_full),      intent(in)  :: buf_writer(:)
        real(WP_full),      intent(out) :: field_owned(:)
        real(WP_full), allocatable :: sbuf(:), rbuf(:)
        integer :: i, k, ierr
        allocate(sbuf(max(1,D%send_total)), rbuf(max(1,D%recv_total)))
        if (D%w_nbuf > 0) then
            do k = 1, D%recv_total
                rbuf(k) = buf_writer(D%recv_target(k))
            end do
        end if
        call MPI_Alltoallv(rbuf, D%recvcounts, D%rdispls, MPI_WP_FULL, &
                           sbuf, D%sendcounts, D%sdispls, MPI_WP_FULL, D%comm, ierr)
        do i = 1, D%myDim
            field_owned(i) = sbuf(D%send_pos(i)+1)
        end do
    end subroutine decomp_gather_2d_r8

    subroutine decomp_gather_3d_r8(D, buf_writer, field_owned)
        type(t_io_decomp),  intent(in)  :: D
        real(WP_full),      intent(in)  :: buf_writer(:,:)
        real(WP_full),      intent(out) :: field_owned(:,:)
        real(WP_full), allocatable :: col(:), inrow(:)
        integer :: nlev, L
        nlev = size(field_owned, 1)
        allocate(col(max(1,D%myDim)), inrow(max(1,D%w_nbuf)))
        do L = 1, nlev
            if (D%w_nbuf > 0) inrow(1:D%w_nbuf) = buf_writer(L, 1:D%w_nbuf)
            call decomp_gather_2d_r8(D, inrow, col)
            field_owned(L, 1:D%myDim) = col(1:D%myDim)
        end do
    end subroutine decomp_gather_3d_r8
#endif

    ! ---- redistribution (integer) ----

    subroutine decomp_redistribute_2d_i(D, field_owned, buf_writer, fill)
        type(t_io_decomp), intent(in)  :: D
        integer,           intent(in)  :: field_owned(:)
        integer,           intent(out) :: buf_writer(:)
        integer,           intent(in)  :: fill
        integer, allocatable :: sbuf(:), rbuf(:)
        integer :: i, k, ierr
        allocate(sbuf(max(1,D%send_total)), rbuf(max(1,D%recv_total)))
        do i = 1, D%myDim
            sbuf(D%send_pos(i)+1) = field_owned(i)
        end do
        call MPI_Alltoallv(sbuf, D%sendcounts, D%sdispls, MPI_INTEGER, &
                           rbuf, D%recvcounts, D%rdispls, MPI_INTEGER, D%comm, ierr)
        if (D%w_nbuf > 0) then
            buf_writer(1:D%w_nbuf) = fill
            do k = 1, D%recv_total
                buf_writer(D%recv_target(k)) = rbuf(k)
            end do
        end if
    end subroutine decomp_redistribute_2d_i

    subroutine decomp_redistribute_3d_i(D, field_owned, buf_writer, fill)
        type(t_io_decomp), intent(in)  :: D
        integer,           intent(in)  :: field_owned(:,:)
        integer,           intent(out) :: buf_writer(:,:)
        integer,           intent(in)  :: fill
        integer, allocatable :: col(:), outrow(:)
        integer :: nlev, L
        nlev = size(field_owned, 1)
        allocate(col(max(1,D%myDim)), outrow(max(1,D%w_nbuf)))
        do L = 1, nlev
            col(1:D%myDim) = field_owned(L, 1:D%myDim)
            call decomp_redistribute_2d_i(D, col, outrow, fill)
            if (D%w_nbuf > 0) buf_writer(L, 1:D%w_nbuf) = outrow(1:D%w_nbuf)
        end do
    end subroutine decomp_redistribute_3d_i

end module mod_io_decomp
