CC = gcc
DEBUG = -g
CFLAGS = -Wall -std=c99 -O3

# these variables are to compile with OpenBLAS
LIB = -lm -lgfortran -lopenblas
INCDIR = -I/opt/OpenBLAS/include
LIBDIR = -L/opt/OpenBLAS/lib
LDLIB = /opt/OpenBLAS/lib

# these variables are to compile with ACML
#LIB = -lm -lacml
#INCDIR = -I/opt/acml5.0.0/gfortran64_int64/include
#LIBDIR = -I/opt/acml5.0.0/gfortran64_int64/lib

blas_asgd: main.o blas_asgd.o
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIB) -o blas_asgd main.o blas_asgd.o

main.o: main.c
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIB) -c -o main.o main.c

blas_asgd.o: blas_asgd.c blas_asgd.h
	$(CC) $(CFLAGS) $(DEBUG) $(INCDIR) $(LIBDIR) $(LIB) -c -o blas_asgd.o blas_asgd.c



# run the compiled program
.ONESHELL: run
.PHONY: run
run:
	export LD_LIBRARY_PATH=$(LDLIB)
	./blas_asgd


# debug the compiled program
.ONESHELL: debug
.PHONY: debug
debug:
	export LD_LIBRARY_PATH=$(LDLIB)
	gdb ./blas_asgd


# clean up the compilation byproducts
.PHONY: clean
clean:
	rm -f *.o
	rm -f blas_asgd
