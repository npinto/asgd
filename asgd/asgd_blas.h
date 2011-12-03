
// determine which BLAS implementation to use

#if defined ASGD_BLAS
#include <cblas.h>
#elif defined ASGD_SSE
#include "sse_blas.h"
#else
#include "simple_blas.h"
#endif

