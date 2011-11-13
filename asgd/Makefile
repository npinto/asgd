CC = gcc
DEBUG = -g
CFLAGS = -Wall -std=c99

LIB = -lm -lcblas
INCDIR = -I/usr/include/
LIBDIR = -L/usr/lib64/atlas

# decide which library to use for the linear algebra
ifeq ($(TYPE),sse)
LTYPE = -DASGD_SSE
endif
ifeq ($(TYPE),blas)
LTYPE = -DASGD_BLAS
endif
ifeq ($(TYPE),)
LTYPE = simple_blas.c
endif

asgd_unit: tests/asgd_unit.c asgd.c simple_blas.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIB) -o bin/asgd_unit $(LTYPE) tests/asgd_unit.c asgd.c

simple_blas_unit: tests/simple_blas_unit.c simple_blas.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIB) -o bin/simple_blas_unit tests/simple_blas_unit.c simple_blas.c

# clean up the compilation byproducts
.PHONY: clean
clean:
	rm -fR *.o
	rm -f bin/asgd_unit
