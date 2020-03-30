module scale_value_reader
implicit none
contains
subroutine load_scale_values(filename, num_inputs, scale_values)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: num_inputs
        real(8), intent(out) :: scale_values(num_inputs, 2)
        character(len=40) :: row_name
        integer :: isu, i
        isu = 2
        open(isu, file=filename, access="sequential", form="formatted")
        read(isu, "(A)")
        do i=1, num_inputs
            read(isu, *) row_name, scale_values(i, 1), scale_values(i, 2)
        end do
        close(isu)
    end subroutine load_scale_values
end module scale_value_reader
