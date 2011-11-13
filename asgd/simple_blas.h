#ifndef _SIMPLE_BLAS_
#define _SIMPLE_BLAS_

#include <stddef.h>

enum CBLAS_ORDER
{
	CblasRowMajor = 1,
	CblasColMajor = 2
};

enum CBLAS_TRANSPOSE
{
	CblasNoTrans = 1,
	CblasTrans = 2,
	CblasConjTrans = 3
};

float cblas_sdsdot(size_t n, float alpha, const float *x, size_t incx, const float *y, size_t incy);

void cblas_sscal(size_t n, float alpha, float *x, size_t incx);

void cblas_saxpy(size_t n, float alpha, const float *x, size_t incx, float *y, size_t incy);

void cblas_sgemm(
		enum CBLAS_ORDER order,
		enum CBLAS_TRANSPOSE transa,
		enum CBLAS_TRANSPOSE transb,
		size_t m,
		size_t n,
		size_t k,
		float alpha,
		const float *a,
		size_t lda,
		const float *b,
		size_t ldb,
		float beta,
		float *c,
		size_t ldc);

#endif

