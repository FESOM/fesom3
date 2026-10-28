module mod_binary_arrays
    ! Size-prefixed unformatted (de)serialization of allocatable arrays, used by
    ! the derived-type read(unformatted)/write(unformatted) bindings. Ported from
    ! the tracer_dwarf (MOD_WRITE/READ_BINARY_ARRAYS), combined into one infra
    ! module, with `use o_PARAM` -> `use mod_precision`.
    !
    ! Only WP-real variants exist: at double/single precision MP == WP (MP=max(WP,4)),
    ! so MP-typed mesh arrays serialize through these too. Half precision has no IO
    ! (MP/=WP there) and the callers compile it out (#ifndef USE_HALF_PRECISION).
    use mod_precision, only: WP, WP_full
    implicit none
    private
    public :: write_bin_array, read_bin_array, write1d_int_static, read1d_int_static

    interface write_bin_array
        module procedure write1d_real, write1d_int, write1d_char, &
                         write2d_real, write2d_int, write3d_real, write3d_int, &
                         write4d_real, write4d_int
#if defined(USE_SINGLE_PRECISION)
        module procedure write2d_full
#endif
    end interface

    interface read_bin_array
        module procedure read1d_real, read1d_int, read1d_char, &
                         read2d_real, read2d_int, read3d_real, read3d_int, &
                         read4d_real, read4d_int
#if defined(USE_SINGLE_PRECISION)
        module procedure read2d_full
#endif
    end interface

contains

    ! ---------------- writers ----------------
    subroutine write1d_real(arr, unit, iostat, iomsg)
        real(kind=WP), allocatable, intent(in) :: arr(:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1
        if (allocated(arr)) then
            s1 = size(arr,1)
            write(unit, iostat=iostat, iomsg=iomsg) s1
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0
        end if
    end subroutine

    subroutine write1d_int(arr, unit, iostat, iomsg)
        integer, allocatable, intent(in) :: arr(:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1
        if (allocated(arr)) then
            s1 = size(arr,1)
            write(unit, iostat=iostat, iomsg=iomsg) s1
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0
        end if
    end subroutine

    subroutine write1d_char(arr, unit, iostat, iomsg)
        character, allocatable, intent(in) :: arr(:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1
        if (allocated(arr)) then
            s1 = size(arr,1)
            write(unit, iostat=iostat, iomsg=iomsg) s1
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0
        end if
    end subroutine

    subroutine write1d_int_static(arr, unit, iostat, iomsg)
        ! Fixed-size (non-allocatable) integer array, e.g. com_struct%rPE.
        integer, intent(in) :: arr(:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1
        s1 = size(arr,1)
        write(unit, iostat=iostat, iomsg=iomsg) s1
        write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1)
    end subroutine

    subroutine write2d_real(arr, unit, iostat, iomsg)
        real(kind=WP), allocatable, intent(in) :: arr(:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2
        if (allocated(arr)) then
            s1 = size(arr,1); s2 = size(arr,2)
            write(unit, iostat=iostat, iomsg=iomsg) s1, s2
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0, 0
        end if
    end subroutine

    subroutine write2d_int(arr, unit, iostat, iomsg)
        integer, allocatable, intent(in) :: arr(:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2
        if (allocated(arr)) then
            s1 = size(arr,1); s2 = size(arr,2)
            write(unit, iostat=iostat, iomsg=iomsg) s1, s2
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0, 0
        end if
    end subroutine

    subroutine write3d_real(arr, unit, iostat, iomsg)
        real(kind=WP), allocatable, intent(in) :: arr(:,:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2, s3
        if (allocated(arr)) then
            s1 = size(arr,1); s2 = size(arr,2); s3 = size(arr,3)
            write(unit, iostat=iostat, iomsg=iomsg) s1, s2, s3
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2,1:s3)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0, 0, 0
        end if
    end subroutine

    subroutine write3d_int(arr, unit, iostat, iomsg)
        integer, allocatable, intent(in) :: arr(:,:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2, s3
        if (allocated(arr)) then
            s1 = size(arr,1); s2 = size(arr,2); s3 = size(arr,3)
            write(unit, iostat=iostat, iomsg=iomsg) s1, s2, s3
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2,1:s3)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0, 0, 0
        end if
    end subroutine

    subroutine write4d_real(arr, unit, iostat, iomsg)
        real(kind=WP), allocatable, intent(in) :: arr(:,:,:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2, s3, s4
        if (allocated(arr)) then
            s1 = size(arr,1); s2 = size(arr,2); s3 = size(arr,3); s4 = size(arr,4)
            write(unit, iostat=iostat, iomsg=iomsg) s1, s2, s3, s4
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2,1:s3,1:s4)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0, 0, 0, 0
        end if
    end subroutine

    subroutine write4d_int(arr, unit, iostat, iomsg)
        integer, allocatable, intent(in) :: arr(:,:,:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2, s3, s4
        if (allocated(arr)) then
            s1 = size(arr,1); s2 = size(arr,2); s3 = size(arr,3); s4 = size(arr,4)
            write(unit, iostat=iostat, iomsg=iomsg) s1, s2, s3, s4
            write(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2,1:s3,1:s4)
        else
            write(unit, iostat=iostat, iomsg=iomsg) 0, 0, 0, 0
        end if
    end subroutine

    ! ---------------- readers ----------------
    subroutine read1d_real(arr, unit, iostat, iomsg)
        real(kind=WP), allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1
        read(unit, iostat=iostat, iomsg=iomsg) s1
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0) then
            allocate(arr(s1)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1)
        end if
    end subroutine

    subroutine read1d_int(arr, unit, iostat, iomsg)
        integer, allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1
        read(unit, iostat=iostat, iomsg=iomsg) s1
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0) then
            allocate(arr(s1)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1)
        end if
    end subroutine

    subroutine read1d_char(arr, unit, iostat, iomsg)
        character, allocatable, intent(inout) :: arr(:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1
        read(unit, iostat=iostat, iomsg=iomsg) s1
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0) then
            allocate(arr(s1)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1)
        end if
    end subroutine

    subroutine read1d_int_static(arr, unit, iostat, iomsg)
        integer, intent(inout) :: arr(:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1
        read(unit, iostat=iostat, iomsg=iomsg) s1
        read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1)
    end subroutine

    subroutine read2d_real(arr, unit, iostat, iomsg)
        real(kind=WP), allocatable, intent(inout) :: arr(:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2
        read(unit, iostat=iostat, iomsg=iomsg) s1, s2
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0 .and. s2 > 0) then
            allocate(arr(s1,s2)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2)
        end if
    end subroutine

#if defined(USE_SINGLE_PRECISION)
    ! Single precision: WP_full arrays (layer thickness, tracer tendency accumulators) go
    ! through the WP binary format, as upstream FESOM2 (#1206 write_acc_array, #1207
    ! write_h_array). Double precision serves them with the WP specifics.
    subroutine write2d_full(arr, unit, iostat, iomsg)
        real(kind=WP_full), allocatable, intent(in) :: arr(:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        real(kind=WP), allocatable :: tmp(:,:)
        if (allocated(arr)) tmp = real(arr, WP)
        call write2d_real(tmp, unit, iostat, iomsg)
    end subroutine

    subroutine read2d_full(arr, unit, iostat, iomsg)
        real(kind=WP_full), allocatable, intent(inout) :: arr(:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        real(kind=WP), allocatable :: tmp(:,:)
        call read2d_real(tmp, unit, iostat, iomsg)
        if (allocated(arr)) deallocate(arr)
        if (allocated(tmp)) arr = real(tmp, WP_full)
    end subroutine
#endif

    subroutine read2d_int(arr, unit, iostat, iomsg)
        integer, allocatable, intent(inout) :: arr(:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2
        read(unit, iostat=iostat, iomsg=iomsg) s1, s2
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0 .and. s2 > 0) then
            allocate(arr(s1,s2)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2)
        end if
    end subroutine

    subroutine read3d_real(arr, unit, iostat, iomsg)
        real(kind=WP), allocatable, intent(inout) :: arr(:,:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2, s3
        read(unit, iostat=iostat, iomsg=iomsg) s1, s2, s3
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0 .and. s2 > 0 .and. s3 > 0) then
            allocate(arr(s1,s2,s3)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2,1:s3)
        end if
    end subroutine

    subroutine read3d_int(arr, unit, iostat, iomsg)
        integer, allocatable, intent(inout) :: arr(:,:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2, s3
        read(unit, iostat=iostat, iomsg=iomsg) s1, s2, s3
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0 .and. s2 > 0 .and. s3 > 0) then
            allocate(arr(s1,s2,s3)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2,1:s3)
        end if
    end subroutine

    subroutine read4d_real(arr, unit, iostat, iomsg)
        real(kind=WP), allocatable, intent(inout) :: arr(:,:,:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2, s3, s4
        read(unit, iostat=iostat, iomsg=iomsg) s1, s2, s3, s4
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0 .and. s2 > 0 .and. s3 > 0 .and. s4 > 0) then
            allocate(arr(s1,s2,s3,s4)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2,1:s3,1:s4)
        end if
    end subroutine

    subroutine read4d_int(arr, unit, iostat, iomsg)
        integer, allocatable, intent(inout) :: arr(:,:,:,:)
        integer, intent(in) :: unit
        integer, intent(out) :: iostat
        character(*), intent(inout) :: iomsg
        integer :: s1, s2, s3, s4
        read(unit, iostat=iostat, iomsg=iomsg) s1, s2, s3, s4
        if (allocated(arr)) deallocate(arr)
        if (s1 > 0 .and. s2 > 0 .and. s3 > 0 .and. s4 > 0) then
            allocate(arr(s1,s2,s3,s4)); read(unit, iostat=iostat, iomsg=iomsg) arr(1:s1,1:s2,1:s3,1:s4)
        end if
    end subroutine

end module mod_binary_arrays
