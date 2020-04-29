#!/bin/bash
DEBUG_FLAGS="-fPIC -g -fimplicit-none  -Wall  -O3 -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fwhole-file  -fcheck=all  -std=f2008  -pedantic  -fbacktrace -fbounds-check -ffpe-trap=zero,invalid,overflow,underflow"
NC_FLAGS="-I/opt/local/include/ -L/opt/local/lib -lnetcdff -llapack -lblas"
rm *.mod *.o
gfortran $DEBUG_FLAGS -c module_neural_net.f90 $NC_FLAGS
gfortran $DEBUG_FLAGS -c module_sf_sfclay_neural_net.f90 *.o $NC_FLAGS
gfortran $DEBUG_FLAGS test_neural_net.f90 *.o -o test_neural_net $NC_FLAGS 
