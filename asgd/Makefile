# ******* INSTRUCTIONS ******* #
# ASGD can be compiled either against an external cBLAS library (provided by the installer),
# or against a naive BLAS implementation (part of the ASGD distribution).
#
# make will default to the naive implementation.
# To use an external BLAS library, please set:
# TYPE = blas		(for instance, call `make TYPE=blas`)
# BLAS_INCDIRS =	(dirs with cBLAS headers)
# BLAS_LIBDIRS =	(dirs with cBLAS libs)
# BLAS_LIBS =		(cBLAS lib files to link against)
# BLAS_HEADER =		(name of the cBLAS header file in brackets, e.g. <cblas.h>)
#
# Available make targets are:
#
# simple_blas_unit
# 		Compiles a unit test for the naive BLAS implementation
#		The unit test will exit with status 0 if the test passes


# ******* COMPILATION SETTINGS ******* #

# list of directories with headers
INCDIRS = 

# list of directories with libraries
LIBDIRS = 

# list of libraries to link against
LIBS = -lm -lrt

# list of object files to compile with
OBJS =

# list of macros for the compiler
DEFS = -D_POSIX_C_SOURCE=199309 

# compiler
CC = gcc

# debug flags
DEBUG = -g

# compiler flags
CFLAGS = -Wall -std=c99 -fPIC -O3 -march=native

# MKL
BLAS_INCDIRS = -I/opt/intel/mkl/include
BLAS_LIBDIRS = -L/opt/intel/mkl/lib/intel64
BLAS_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lmkl_mc3 -lmkl_def -lpthread -fopenmp
# lmkl_mc3 and -lmkl_def are necessary because of a bug when loading .so from Python
BLAS_HEADER = <mkl_cblas.h>

# if TYPE=blas, link against external BLAS library
ifeq ($(TYPE),blas)
INCDIRS += $(BLAS_INCDIRS)
LIBDIRS += $(BLAS_LIBDIRS)
LIBS += $(BLAS_LIBS)
DEFS += -DASGD_BLAS -DBLAS_HEADER='$(BLAS_HEADER)'
else
OBJS += simple_blas.o
endif

asgd_unit: asgd.o asgd_core.o simple_blas.o tests/asgd_unit.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIRS) $(LIBDIRS) $(LIBS) $(DEFS) -o bin/asgd_unit $(OBJS) asgd.o asgd_core.o tests/asgd_unit.c

asgd.o: simple_blas.o asgd_core.o asgd.c asgd.h
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIRS) $(LIBDIRS) $(LIBS) $(DEFS) -c -o asgd.o $(OBJS) asgd.c

asgd_core.o: simple_blas.o asgd_core.c asgd_core.h
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIRS) $(LIBDIRS) $(LIBS) $(DEFS) -c -o asgd_core.o $(OBJS) asgd_core.c
	$(CC) -shared $(CFLAGS) $(DEBUG) $(INCDIRS) $(LIBDIRS) $(LIBS) $(DEFS) -o asgd_core.so $(OBJS) asgd_core.c

simple_blas_unit: simple_blas.o tests/simple_blas_unit.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIRS) $(LIBDIRS) $(LIBS) $(DEFS) -o bin/simple_blas_unit $(OBJS) tests/simple_blas_unit.c

simple_blas.o: simple_blas.c simple_blas.h
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIRS) $(LIBDIRS) $(LIBS) $(DEFS) -c -o simple_blas.o simple_blas.c

# clean up compilation byproducts
.PHONY: clean
clean:
	rm -fR *.o
	rm -fR *.so
	rm -fR bin/*

