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
LTYPE =
endif

asgd_unit: tests/asgd_unit.c asgd.c
	$(CC) $(CFLAGS) $(DEBUG) $(LTYPE) $(INCDIR) $(LIBDIR) $(LIB) -o bin/asgd_unit tests/asgd_unit.c asgd.c

# clean up the compilation byproducts
.PHONY: clean
clean:
	rm -fR *.o
	rm -f bin/asgd_unit
