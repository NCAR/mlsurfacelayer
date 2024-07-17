script=my_script.py && chmod +x $script && ./$script

#Or you can define a function for it:


run_script() {
    chmod +x $1 && ./$1
}
run_script my_script.py