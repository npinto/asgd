# GCC
CC = gcc
#DEBUG = -g
CFLAGS = -Wall -std=c99 -fPIC -O3 -march=nocona

# ICC
#CC = icc
#CFLAGS = -Wall -std=c99 -fPIC -O2 -xhost -ipo -fno-fnalias

# stock ATLAS
INCDIR = -I/usr/include/
LIBDIR = -L/usr/lib64/atlas

# custom ATLAS
#INCDIR = -I/home/cloud/projects/asgd/ATLASobj/include
#LIBDIR = -L/home/cloud/projects/asgd/ATLASobj/lib

# MKL
#INCDIR = -I/opt/intel/mkl/include
#LIBDIR = -L/opt/intel/mkl/lib/intel64

MATHLIB = -lm
#MATHLIB =

ifeq ($(TYPE),blas)
LIBS = $(MATHLIB) -lcblas
DEFS = -DASGD_BLAS
OBJS = 
else
LIBS = $(MATHLIB)
DEFS = 
OBJS = simple_blas.o
endif

asgd_unit: asgd.o asgd_core.o simple_blas.o tests/asgd_unit.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIBS) $(DEFS) -o bin/asgd_unit $(OBJS) asgd.o asgd_core.o tests/asgd_unit.c

asgd.o: simple_blas.o asgd_core.o asgd.c asgd.h
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIBS) $(DEFS) -c -o asgd.o asgd.c

asgd_core.o: simple_blas.o asgd_core.c asgd_core.h
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIBS) $(DEFS) -c -o asgd_core.o asgd_core.c
	$(CC) -shared $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIBS) $(DEFS) -o asgd_core.so $(OBJS) asgd_core.c

simple_blas_unit: simple_blas.c simple_blas.h tests/simple_blas_unit.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIBS) $(DEFS) -o bin/simple_blas_unit tests/simple_blas_unit.c

simple_blas.o: simple_blas.c simple_blas.h
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIBS) $(DEFS) -c -o simple_blas.o simple_blas.c

# clean up compilation byproducts
.PHONY: clean
clean:
	rm -fR *.o
	rm -fR *.so
	rm -f bin/asgd_unit
	rm -f bin/simple_blas_unit
