#include "simple_blas.h"

#include <stdio.h>
#include <stdlib.h>

void cblas_sscal(size_t n, float alpha, float *x, size_t incx)
{
	while (n--)
	{
		x[n*incx] *= alpha;
	}
}

void cblas_scopy(size_t n, float *x, size_t incx, float *y, size_t incy)
{
	while (n--)
	{
		y[n*incy] = x[n*incx];
	}
}

void cblas_saxpy(size_t n, float alpha, const float *x, size_t incx, float *y, size_t incy)
{
	while (n--)
	{
		y[n*incy] += x[n*incx] * alpha;
	}
}

float cblas_sdsdot(size_t n, float alpha, const float *x, size_t incx, const float *y, size_t incy)
{
	while (n--)
	{
		alpha += x[n*incx] * y[n*incy];
	}
	return alpha;
}

inline static void _cblas_sgemv_n(
		size_t m,
		size_t n,
		float alpha,
		float *A,
		size_t lda,
		float *x,
		size_t incx,
		float beta,
		float *y,
		size_t incy)
{
	while (m--)
	{
		y[m*incy] *= beta;
		size_t N = n;
		while (N--)
		{
			y[m*incy] += alpha * A[m*lda + N] * x[N*incx];
		}
	}
}

inline static void _cblas_sgemv_t(
		size_t m,
		size_t n,
		float alpha,
		float *A,
		size_t lda,
		float *x,
		size_t incx,
		float beta,
		float *y,
		size_t incy)
{
	while (n--)
	{
		y[n*incy] *= beta;
		size_t M = m;
		while (M--)
		{
			y[n*incy] += alpha * A[M*lda + n] * x[M*incx];
		}
	}
}

void cblas_sgemv(
		enum CBLAS_ORDER order,
		enum CBLAS_TRANSPOSE trans,
		size_t m,
		size_t n,
		float alpha,
		float *A,
		size_t lda,
		float *x,
		size_t incx,
		float beta,
		float *y,
		size_t incy)
{
	if (order == CblasColMajor)
	{
		fprintf(stderr, "column-major matrices not supported\n");
		exit(EXIT_FAILURE);
	}

	switch (trans)
	{
		case CblasTrans:
			_cblas_sgemv_t(m, n, alpha, A, lda, x, incx, beta, y, incy);
			break;

		case CblasNoTrans:
			_cblas_sgemv_n(m, n, alpha, A, lda, x, incx, beta, y, incy);
			break;

		default:
			fprintf(stderr, "invalid transposition parameters\n");
			exit(EXIT_FAILURE);
	}
}

void cblas_sger(
		enum CBLAS_ORDER order,
		size_t m,
		size_t n,
		float alpha,
		float *x,
		size_t incx,
		float *y,
		size_t incy,
		float *A,
		size_t lda)
{
	if (order == CblasColMajor)
	{
		fprintf(stderr, "column-major matrices not supported\n");
		exit(EXIT_FAILURE);
	}

	while (m--)
	{
		size_t N = n;
		while (N--)
		{
			A[m*lda + N] += alpha * x[m*incx] * y[N*incy];
		}
	}
}

inline static void _cblas_sgemm_row_n_n(
		size_t m,
		size_t n,
		size_t k,
		float alpha,
		const float *A,
		size_t lda,
		const float *B,
		size_t ldb,
		float beta,
		float *C,
		size_t ldc)
{
	size_t N = n;
	size_t K = k;
	while (m--)
	{
		n = N;
		while (n--)
		{
			C[m*ldc + n] *= beta;
			k = K;
			while (k--)
			{
				C[m*ldc + n] += alpha * A[m*lda + k] * B[k*ldb + n];
			}
		}
	}
}

inline static void _cblas_sgemm_row_t_n(
		size_t m,
		size_t n,
		size_t k,
		float alpha,
		const float *A,
		size_t lda,
		const float *B,
		size_t ldb,
		float beta,
		float *C,
		size_t ldc)
{
	size_t N = n;
	size_t K = k;
	while (m--)
	{
		n = N;
		while (n--)
		{
			C[m*ldc + n] *= beta;
			k = K;
			while (k--)
			{
				C[m*ldc + n] += alpha * A[k*lda + m] * B[k*ldb + n];
			}
		}
	}
}

inline static void _cblas_sgemm_row_n_t(
		size_t m,
		size_t n,
		size_t k,
		float alpha,
		const float *A,
		size_t lda,
		const float *B,
		size_t ldb,
		float beta,
		float *C,
		size_t ldc)
{
	size_t N = n;
	size_t K = k;
	while (m--)
	{
		n = N;
		while (n--)
		{
			C[m*ldc + n] *= beta;
			k = K;
			while (k--)
			{
				C[m*ldc + n] += alpha * A[m*lda + k] * B[n*ldb + k];
			}
		}
	}
}

inline static void _cblas_sgemm_row_t_t(
		size_t m,
		size_t n,
		size_t k,
		float alpha,
		const float *A,
		size_t lda,
		const float *B,
		size_t ldb,
		float beta,
		float *C,
		size_t ldc)
{
	size_t N = n;
	size_t K = k;
	while (m--)
	{
		n = N;
		while (n--)
		{
			C[m*ldc + n] *= beta;
			k = K;
			while (k--)
			{
				C[m*ldc + n] += alpha * A[k*lda + m] * B[n*ldb + k];
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
		const float *A,
		size_t lda,
		const float *B,
		size_t ldb,
		float beta,
		float *C,
		size_t ldc)
{
	if (order == CblasColMajor)
	{
		fprintf(stderr, "column major order not supported\n");
		exit(EXIT_FAILURE);
	}

	switch (transa*0x10 + transb)
	{
		case CblasNoTrans*0x10 + CblasNoTrans:
			_cblas_sgemm_row_n_n(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;

		case CblasTrans*0x10 + CblasNoTrans:
			_cblas_sgemm_row_t_n(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;

		case CblasNoTrans*0x10 + CblasTrans:
			_cblas_sgemm_row_n_t(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;

		case CblasTrans*0x10 + CblasTrans:
			_cblas_sgemm_row_t_t(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
			break;

		default:
			fprintf(stderr, "invalid transposition parameters\n");
			exit(EXIT_FAILURE);
	}

}

