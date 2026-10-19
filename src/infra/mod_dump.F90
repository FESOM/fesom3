module mod_dump
    ! Gid-keyed per-substep validation dump (decision D8). Byte-format-IDENTICAL to
    ! the FESOM2 v2.7.3 oracle shim (port2/fesom2/src/fesom_dump_shim.F90) so the
    ! same tools/dump_diff.py compares FESOM3 vs FESOM2 dumps. Keyed by 1-based
    ! GLOBAL id => rank-order independent. Env-gated, compiled-in but a no-op when
    ! off (zero overhead via the active-flag early return).
    !
    ! Record (little-endian, stream, no header):
    !   int32 step | int32 substep_id | int32 probe_gid | int32 nlevels |
    !   char[24] field_name | real64 values[nlevels]      (40 + 8*nlevels bytes)
    !
    ! Two parallel streams (FESOM2's shim is node-only; elements are the NEW shim
    ! the plan calls for): FESOM_DUMP_FILE -> node fields (gids on myList_nod2D),
    ! FESOM_DUMP_FILE_ELEM -> element fields (gids on myList_elem2D). Each per-rank:
    ! <prefix>.<mype:05d>.
    use, intrinsic :: iso_fortran_env, only: int32, real64
    use mod_precision, only: WP, WP_full
    implicit none
    private

    ! dump_node is generic: in a single-precision build the WP_full layer thickness
    ! (hnode/hnode_new, #1207) is dumped through its own specific (the dump is real64)
    interface dump_node
        module procedure dump_node_wp
#if defined(USE_SINGLE_PRECISION)
        module procedure dump_node_full
#endif
    end interface

    ! ---- substep IDs (mirror FESOM2 fesom_dump_shim.F90:37-53) ----
    integer, parameter, public :: DUMP_SUBSTEP_INIT        = 0
    integer, parameter, public :: DUMP_SUBSTEP_PRESSURE_BV = 1
    integer, parameter, public :: DUMP_SUBSTEP_SW_AB       = 2
    integer, parameter, public :: DUMP_SUBSTEP_PGF         = 3
    integer, parameter, public :: DUMP_SUBSTEP_MIXING      = 4
    integer, parameter, public :: DUMP_SUBSTEP_VEL_RHS     = 5
    integer, parameter, public :: DUMP_SUBSTEP_VISC_FILTER = 6
    integer, parameter, public :: DUMP_SUBSTEP_IMPL_VISC   = 7
    integer, parameter, public :: DUMP_SUBSTEP_SSH_RHS     = 8
    integer, parameter, public :: DUMP_SUBSTEP_SSH_SOLVE   = 9
    integer, parameter, public :: DUMP_SUBSTEP_UPDATE_VEL  = 10
    integer, parameter, public :: DUMP_SUBSTEP_HBAR        = 11
    integer, parameter, public :: DUMP_SUBSTEP_ETA_N       = 12
    integer, parameter, public :: DUMP_SUBSTEP_ALE         = 13
    integer, parameter, public :: DUMP_SUBSTEP_GM_BOLUS    = 14
    integer, parameter, public :: DUMP_SUBSTEP_TRACERS     = 15
    integer, parameter, public :: DUMP_SUBSTEP_THICKNESS   = 16

    ! ---- probe global ids (1-based). Node set matches FESOM2 shim exactly. ----
    integer, parameter, public :: DUMP_NPROBES_NOD = 5
    integer, parameter, public :: DUMP_PROBE_GIDS_NOD(DUMP_NPROBES_NOD) = &
        [1001, 1500, 2000, 2500, 3000]
    integer, parameter, public :: DUMP_NPROBES_ELEM = 5
    integer, parameter, public :: DUMP_PROBE_GIDS_ELEM(DUMP_NPROBES_ELEM) = &
        [1000, 2000, 3000, 4000, 5000]

    ! ---- internal state ----
    logical, save :: inited       = .false.
    logical, save :: nod_active   = .false.
    logical, save :: elem_active  = .false.
    integer, save :: nod_unit     = -1
    integer, save :: elem_unit    = -1
    integer, save :: max_steps    = 10
    integer, save :: probe_loc_nod(DUMP_NPROBES_NOD)   = -1
    integer, save :: probe_loc_elem(DUMP_NPROBES_ELEM) = -1
    ! M8c Step-3 bisection: FESOM_DUMP_ALL dumps EVERY owned node/elem (not the 5 probes); paired
    ! with FESOM_DUMP_MINSTEP it dumps a narrow step WINDOW [min_step, max_steps] so the all-node
    ! volume stays bounded. Byte-format-identical to the FESOM2 shim's dump_all (gid-keyed).
    logical, save :: dump_all  = .false.
    integer, save :: min_step  = 0
    integer, save :: my_nod    = 0, my_elem = 0
    integer, allocatable, save :: list_nod(:), list_elem(:)

    public :: dump_init, dump_finalize
    public :: dump_node, dump_node_2d, dump_elem, dump_elem_2d
    public :: dump_is_active

contains

    subroutine dump_init(mype, myDim_nod2D, myList_nod2D, myDim_elem2D, myList_elem2D, &
                         node_prefix, elem_prefix)
        integer, intent(in) :: mype, myDim_nod2D, myDim_elem2D
        integer, intent(in) :: myList_nod2D(:), myList_elem2D(:)
        character(len=*), intent(in), optional :: node_prefix, elem_prefix
        character(len=512) :: npfx, epfx, env_max, fname
        integer :: ios, env_len

        if (inited) return
        inited = .true.

        ! max steps (env, optional)
        call get_environment_variable('FESOM_DUMP_MAXSTEPS', env_max, status=ios)
        if (ios == 0 .and. len_trim(env_max) > 0) then
            read(env_max, *, iostat=ios) max_steps
            if (ios /= 0) max_steps = 10
        end if
        ! M8c Step-3 bisection: FESOM_DUMP_ALL (every owned node) + FESOM_DUMP_MINSTEP (window start)
        call get_environment_variable('FESOM_DUMP_ALL', env_max, status=ios)
        dump_all = (ios == 0 .and. len_trim(env_max) > 0)
        call get_environment_variable('FESOM_DUMP_MINSTEP', env_max, status=ios)
        if (ios == 0 .and. len_trim(env_max) > 0) then
            read(env_max, *, iostat=ios) min_step
            if (ios /= 0) min_step = 0
        end if
        if (dump_all) then
            my_nod  = myDim_nod2D;  allocate(list_nod(my_nod));   list_nod  = myList_nod2D(1:my_nod)
            my_elem = myDim_elem2D; allocate(list_elem(my_elem)); list_elem = myList_elem2D(1:my_elem)
        end if

        ! ---- node stream ----
        npfx = ''
        if (present(node_prefix)) then
            npfx = node_prefix
        else
            call get_environment_variable('FESOM_DUMP_FILE', npfx, length=env_len, status=ios)
            if (ios /= 0) npfx = ''
        end if
        if (len_trim(npfx) > 0) then
            call resolve_probes(DUMP_PROBE_GIDS_NOD, myDim_nod2D, myList_nod2D, probe_loc_nod)
            write(fname, '(A,".",I5.5)') trim(npfx), mype
            open(newunit=nod_unit, file=trim(fname), status='replace', &
                 form='unformatted', access='stream', action='write', iostat=ios)
            nod_active = (ios == 0)
        end if

        ! ---- element stream ----
        epfx = ''
        if (present(elem_prefix)) then
            epfx = elem_prefix
        else
            call get_environment_variable('FESOM_DUMP_FILE_ELEM', epfx, length=env_len, status=ios)
            if (ios /= 0) epfx = ''
        end if
        if (len_trim(epfx) > 0) then
            call resolve_probes(DUMP_PROBE_GIDS_ELEM, myDim_elem2D, myList_elem2D, probe_loc_elem)
            write(fname, '(A,".",I5.5)') trim(epfx), mype
            open(newunit=elem_unit, file=trim(fname), status='replace', &
                 form='unformatted', access='stream', action='write', iostat=ios)
            elem_active = (ios == 0)
        end if
    end subroutine dump_init

    subroutine resolve_probes(gids, myDim, myList, locals)
        integer, intent(in)  :: gids(:), myDim, myList(:)
        integer, intent(out) :: locals(:)
        integer :: i, k
        do i = 1, size(gids)
            locals(i) = -1
            do k = 1, myDim
                if (myList(k) == gids(i)) then
                    locals(i) = k; exit
                end if
            end do
        end do
    end subroutine resolve_probes

    logical function dump_is_active()
        dump_is_active = nod_active .or. elem_active
    end function

#if defined(USE_SINGLE_PRECISION)
    subroutine dump_node_full(substep_id, step, field_name, field_3d, nlevels_nod2D)
        integer,            intent(in) :: substep_id, step
        character(len=*),   intent(in) :: field_name
        real(kind=WP_full), intent(in) :: field_3d(:,:)
        integer,            intent(in) :: nlevels_nod2D(:)
        integer :: i, lid, nlev
        character(len=24) :: name24
        if (.not. nod_active .or. step > max_steps .or. step < min_step) return
        name24 = field_name
        if (dump_all) then
            do i = 1, my_nod
                nlev = nlevels_nod2D(i)
                write(nod_unit) int(step,int32), int(substep_id,int32), &
                    int(list_nod(i),int32), int(nlev,int32), &
                    name24, real(field_3d(1:nlev, i), real64)
            end do
            return
        end if
        do i = 1, DUMP_NPROBES_NOD
            lid = probe_loc_nod(i)
            if (lid <= 0) cycle
            nlev = nlevels_nod2D(lid)
            write(nod_unit) int(step,int32), int(substep_id,int32), &
                int(DUMP_PROBE_GIDS_NOD(i),int32), int(nlev,int32), &
                name24, real(field_3d(1:nlev, lid), real64)
        end do
    end subroutine dump_node_full
#endif

    subroutine dump_node_wp(substep_id, step, field_name, field_3d, nlevels_nod2D)
        ! Column node field (nl, nod_total), truncated to nlevels_nod2D(local).
        integer,          intent(in) :: substep_id, step
        character(len=*), intent(in) :: field_name
        real(kind=WP),    intent(in) :: field_3d(:,:)
        integer,          intent(in) :: nlevels_nod2D(:)
        integer :: i, lid, nlev
        character(len=24) :: name24
        if (.not. nod_active .or. step > max_steps .or. step < min_step) return
        name24 = field_name
        if (dump_all) then
            do i = 1, my_nod
                nlev = nlevels_nod2D(i)
                write(nod_unit) int(step,int32), int(substep_id,int32), &
                    int(list_nod(i),int32), int(nlev,int32), &
                    name24, real(field_3d(1:nlev, i), real64)
            end do
            return
        end if
        do i = 1, DUMP_NPROBES_NOD
            lid = probe_loc_nod(i)
            if (lid <= 0) cycle
            nlev = nlevels_nod2D(lid)
            write(nod_unit) int(step,int32), int(substep_id,int32), &
                int(DUMP_PROBE_GIDS_NOD(i),int32), int(nlev,int32), &
                name24, real(field_3d(1:nlev, lid), real64)
        end do
    end subroutine dump_node_wp

    subroutine dump_node_2d(substep_id, step, field_name, field_2d)
        integer,          intent(in) :: substep_id, step
        character(len=*), intent(in) :: field_name
        real(kind=WP),    intent(in) :: field_2d(:)
        integer :: i, lid
        character(len=24) :: name24
        if (.not. nod_active .or. step > max_steps .or. step < min_step) return
        name24 = field_name
        if (dump_all) then
            do i = 1, my_nod
                write(nod_unit) int(step,int32), int(substep_id,int32), &
                    int(list_nod(i),int32), int(1,int32), &
                    name24, [real(field_2d(i), real64)]
            end do
            return
        end if
        do i = 1, DUMP_NPROBES_NOD
            lid = probe_loc_nod(i)
            if (lid <= 0) cycle
            write(nod_unit) int(step,int32), int(substep_id,int32), &
                int(DUMP_PROBE_GIDS_NOD(i),int32), int(1,int32), &
                name24, [real(field_2d(lid), real64)]
        end do
    end subroutine dump_node_2d

    subroutine dump_elem(substep_id, step, field_name, field_3d, nlevels_elem)
        ! Column element field (nl, elem_total), truncated to nlevels_elem(local).
        integer,          intent(in) :: substep_id, step
        character(len=*), intent(in) :: field_name
        real(kind=WP),    intent(in) :: field_3d(:,:)
        integer,          intent(in) :: nlevels_elem(:)
        integer :: i, lid, nlev
        character(len=24) :: name24
        if (.not. elem_active .or. step > max_steps .or. step < min_step) return
        name24 = field_name
        if (dump_all) then
            do i = 1, my_elem
                nlev = nlevels_elem(i)
                write(elem_unit) int(step,int32), int(substep_id,int32), &
                    int(list_elem(i),int32), int(nlev,int32), &
                    name24, real(field_3d(1:nlev, i), real64)
            end do
            return
        end if
        do i = 1, DUMP_NPROBES_ELEM
            lid = probe_loc_elem(i)
            if (lid <= 0) cycle
            nlev = nlevels_elem(lid)
            write(elem_unit) int(step,int32), int(substep_id,int32), &
                int(DUMP_PROBE_GIDS_ELEM(i),int32), int(nlev,int32), &
                name24, real(field_3d(1:nlev, lid), real64)
        end do
    end subroutine dump_elem

    subroutine dump_elem_2d(substep_id, step, field_name, field_2d)
        integer,          intent(in) :: substep_id, step
        character(len=*), intent(in) :: field_name
        real(kind=WP),    intent(in) :: field_2d(:)
        integer :: i, lid
        character(len=24) :: name24
        if (.not. elem_active .or. step > max_steps .or. step < min_step) return
        name24 = field_name
        if (dump_all) then
            do i = 1, my_elem
                write(elem_unit) int(step,int32), int(substep_id,int32), &
                    int(list_elem(i),int32), int(1,int32), &
                    name24, [real(field_2d(i), real64)]
            end do
            return
        end if
        do i = 1, DUMP_NPROBES_ELEM
            lid = probe_loc_elem(i)
            if (lid <= 0) cycle
            write(elem_unit) int(step,int32), int(substep_id,int32), &
                int(DUMP_PROBE_GIDS_ELEM(i),int32), int(1,int32), &
                name24, [real(field_2d(lid), real64)]
        end do
    end subroutine dump_elem_2d

    subroutine dump_finalize()
        if (nod_active)  then; close(nod_unit);  nod_active  = .false.; end if
        if (elem_active) then; close(elem_unit); elem_active = .false.; end if
    end subroutine dump_finalize

end module mod_dump
