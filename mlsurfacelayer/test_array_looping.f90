subroutine test_loop(arr, ims, ime, its, ite)
    implicit none
    integer, intent(in) :: ims, ime, its, ite
    real, dimension(ims:ime), intent(in) :: arr
    integer :: i
    do i=its, ite
        print *, arr(i)
    end do
end subroutine test_loop

program test_array_looping
    implicit none
    integer, parameter ::ims = 1, ime = 10, its= 2, ite = 15
    real, dimension(ims:ime) :: arr
    arr(ims:ime) = (/ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 /)
    call test_loop(arr(ims:ime), ims, ime, its, ite)
end program test_array_looping