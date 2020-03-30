program run_random_forest
    use module_random_forest
    implicit none
    character(len=200) :: file_path
    type(decision_tree), allocatable :: rf(:)
    real(kind=8) :: input_data(1, 15)
    real(kind=8) :: prediction(1)
    integer :: n, num_trees
    file_path = "/Users/dgagne/mlsurfacelayer/ustar_rf/"
    print*, file_path
    call load_random_forest(file_path, rf)
    open(345, file="test_input.csv")
    read(345, "(A)")
    read(345, *) input_data
    close(345)
    print *, input_data
    print *, "Input Size", size(input_data, 1), size(input_data, 2)
    print *, "RF Size ", size(rf)
    num_trees = size(rf)
    do n=1,num_trees
        print*, rf(n)%feature(1), rf(n)%impurity(1)
    end do
    call random_forest_predict(input_data, rf, prediction)
    print *, prediction
end program run_random_forest
