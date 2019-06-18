module module_random_forest
    implicit none

    type decision_tree
        integer :: nodes
        integer, allocatable :: node(:)
        integer, allocatable :: feature(:)
        real(kind=8), allocatable :: threshold(:)
        real(kind=8), allocatable :: tvalue(:)
        integer, allocatable :: children_left(:)
        integer, allocatable :: children_right(:)
        real(kind=8), allocatable :: impurity(:)
    end type decision_tree

    contains

    subroutine load_decision_tree(filename, tree)
        character(len=200), intent(in) :: filename
        type(decision_tree), intent(out) :: tree
        integer :: n, stat
        tree%nodes = 0
        stat = 0
        open(100, file=trim(filename), access="sequential", form="formatted")
        read(100, '(A)')
        do
            read(100, *, iostat=stat)
            if (stat < 0) exit
            tree%nodes = tree%nodes + 1
        end do
        rewind(100)
        allocate(tree%node(tree%nodes))
        allocate(tree%feature(tree%nodes))
        allocate(tree%threshold(tree%nodes))
        allocate(tree%tvalue(tree%nodes))
        allocate(tree%children_left(tree%nodes))
        allocate(tree%children_right(tree%nodes))
        allocate(tree%impurity(tree%nodes))
        read(100, '(A)')
        do n=1,tree%nodes
           read(100, *) tree%node(n), tree%feature(n), tree%threshold(n), tree%tvalue(n), &
                   tree%children_left(n), tree%children_right(n), tree%impurity(n)
        end do
        close(100)
    end subroutine load_decision_tree

    subroutine load_random_forest(file_path, random_forest_array)
        character(len=*), intent(in) :: file_path
        character(len=300), allocatable :: filenames(:)
        type(decision_tree), allocatable, intent(out) :: random_forest_array(:)
        integer :: num_trees, n, stat
        character(len=6 + len_trim(file_path)) :: command_start
        character(len=len_trim(file_path) + 35) :: command
        command_start = "ls -1 " // trim(file_path) 
        command = trim(command_start) // "*_tree_*.csv > tree_files.txt"
        print *, command
        call system(command)
        num_trees = 0
        stat = 0
        open(55, file="tree_files.txt", access="sequential", form="formatted")
        do
            read(55, '(A)', iostat=stat)
            if (stat /= 0) exit
            num_trees = num_trees + 1
        end do
        rewind(55)
        allocate(filenames(num_trees))
        allocate(random_forest_array(num_trees))
        do n=1,num_trees
            read(55, '(A)') filenames(n)
            call load_decision_tree(filenames(n), random_forest_array(n))
        end do
        close(55)
        call system("rm tree_files.txt")
        deallocate(filenames)
    end subroutine load_random_forest

    function random_forest_predict(input_data, random_forest_array) result(prediction)
        real(kind=8), intent(in) :: input_data(:)
        type(decision_tree), intent(in) :: random_forest_array(:)
        real(kind=8) :: prediction
        integer :: n
        integer :: num_trees
        real(kind=8), allocatable :: tree_predictions(:)
        num_trees = size(random_forest_array)
        allocate(tree_predictions(num_trees))
        tree_predictions = 0
        do n=1, num_trees
            tree_predictions(n) = decision_tree_predict(input_data, random_forest_array(n))
        end do
        prediction = sum(tree_predictions) / real(num_trees)
        deallocate(tree_predictions)
    end function random_forest_predict

    function decision_tree_predict(input_data_tree, tree) result(tree_prediction)
        real(kind=8), intent(in) :: input_data_tree(:)
        type(decision_tree), intent(in) :: tree
        integer :: node
        real(kind=8) :: tree_prediction
        logical :: not_leaf
        logical :: exceeds
        node = 1
        tree_prediction = -999
        not_leaf = .TRUE.
        do while (not_leaf)
            if (tree%feature(node) == -2) then
                tree_prediction = tree%tvalue(node)
                not_leaf = .FALSE.
            else
                if (tree%feature(node) + 1 > size(input_data_tree)) then
                    print*, tree%feature(node) + 1, size(input_data_tree)
                end if
                exceeds = input_data_tree(tree%feature(node) + 1) > tree%threshold(node)
                if (exceeds) then
                    node = tree%children_right(node) + 1
                else
                    node = tree%children_left(node) + 1
                end if
            end if
        end do
    end function decision_tree_predict
end module module_random_forest
