#!/bin/bash
DEBUG_FLAGS="-g -fPIC -fimplicit-none  -Wall  -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fwhole-file  -fcheck=all  -std=f2008  -pedantic  -fbacktrace -fbounds-check -ffpe-trap=zero,invalid,overflow,underflow"
NC_FLAGS="-I/opt/local/include/ -L/opt/local/lib -lnetcdff"
rm *.mod *.o
gfortran $DEBUG_FLAGS -c module_random_forest.f90 $NC_FLAGS
gfortran $DEBUG_FLAGS -c module_sf_sfclay_random_forest.f90 *.o $NC_FLAGS
gfortran $DEBUG_FLAGS run_random_forest.f90 *.o -o run_random_forest $NC_FLAGS 
