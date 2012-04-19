
// determine which BLAS implementation to use

#if defined ASGD_BLAS
#include BLAS_HEADER
#else
#include "simple_blas.h"
#endif

