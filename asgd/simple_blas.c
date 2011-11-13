#include "simple_blas.h"

#include <stdio.h>
#include <stdlib.h>

float cblas_sdsdot(size_t n, float alpha, const float *x, size_t incx, const float *y, size_t incy)
{
	while (n--)
	{
		alpha += x[n*incx] * y[n*incy];
	}
	return alpha;
}

void cblas_sscal(size_t n, float alpha, float *x, size_t incx)
{
	while (n--)
	{
		x[n*incx] *= alpha;
	}
}

void cblas_saxpy(size_t n, float alpha, const float *x, size_t incx, float *y, size_t incy)
{
	while (n--)
	{
		y[n*incy] += x[n*incx] * alpha;
	}
}

inline static void _cblas_sgemm_row_not_not(
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
		size_t ldc)
{
	size_t N = n;
	size_t K = k;
	while (m--)
	{
		n = N;
		while (n--)
		{
			c[m*ldc + n] *= beta;
			k = K;
			while (k--)
			{
				c[m*ldc + n] += alpha * a[m*lda + k] * b[k*ldb + n];
			}
		}
	}
}

inline static void _cblas_sgemm_row_t_not(
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
		size_t ldc)
{
	size_t N = n;
	size_t K = k;
	while (m--)
	{
		n = N;
		while (n--)
		{
			c[m*ldc + n] *= beta;
			k = K;
			while (k--)
			{
				c[m*ldc + n] += alpha * a[k*lda + m] * b[k*ldb + n];
			}
		}
	}
}

inline static void _cblas_sgemm_row_not_t(
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
		size_t ldc)
{
	size_t N = n;
	size_t K = k;
	while (m--)
	{
		n = N;
		while (n--)
		{
			c[m*ldc + n] *= beta;
			k = K;
			while (k--)
			{
				c[m*ldc + n] += alpha * a[m*lda + k] * b[n*ldb + k];
			}
		}
	}
}

inline static void _cblas_sgemm_row_t_t(
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
		size_t ldc)
{
	size_t N = n;
	size_t K = k;
	while (m--)
	{
		n = N;
		while (n--)
		{
			c[m*ldc + n] *= beta;
			k = K;
			while (k--)
			{
				c[m*ldc + n] += alpha * a[k*lda + m] * b[n*ldb + k];
			}
		}
	}
}

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
		size_t ldc)
{
	if (__builtin_expect(order == CblasColMajor, 0))
	{
		fprintf(stderr,
			"column major order not supported by ASGD BLAS implementation\n");
		exit(EXIT_FAILURE);
	}

	switch (transa*0x10 + transb)
	{
		case CblasNoTrans*0x10 + CblasNoTrans:
			_cblas_sgemm_row_not_not(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
			break;

		case CblasTrans*0x10 + CblasNoTrans:
			_cblas_sgemm_row_t_not(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
			break;

		case CblasNoTrans*0x10 + CblasTrans:
			_cblas_sgemm_row_not_t(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
			break;

		case CblasTrans*0x10 + CblasTrans:
			_cblas_sgemm_row_t_t(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
			break;

		default:
			fprintf(stderr,
				"invalid transposition parameters\n");
			exit(EXIT_FAILURE);
	}

}

