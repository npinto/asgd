#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "../simple_blas.h"

/**
 * float *exp		Array of expected values
 * float *got		Array to compare against exp
 * size_t length	How many items in each array
 * size_t row		After how many entries to add a newline
 * const char *name	The name of the test (for the printed report)
 * double tolerance	The maximum acceptable diff between each couple of values
 */
static bool compare(
		float *exp,
		float *got,
		size_t length,
		size_t row,
		const char *name,
		double tolerance)
{
	bool succ = true;
	bool last;
	
	for (size_t i = 0; i*row < length; ++i)
	{
		printf("exp: ");
		for (size_t j = 0; j < row; ++j)
		{
			printf("%9.4f ", exp[i*row+j]);
		}
		printf("\ngot: ");
		for (size_t j = 0; j < row; ++j)
		{
			printf("%9.4f ", got[i*row+j]);
		}
		printf("\nres: ");
		for (size_t j = 0; j < row; ++j)
		{
			last = abs(exp[i*row+j] - got[i*row+j]) <= tolerance;
			succ = succ ? last : false;
			printf("    %c     ", last ? '=' : '!');
		}
		printf("\n");
	}

	if (succ)
	{
		fprintf(stdout, "%s passed\n\n", name);
	}
	else
	{
		fprintf(stderr, "%s failed\n\n", name);
	}

	return succ;
}

bool test_sscal(double tolerance)
{
	float x[1][5] = {{3,5,7,11,13}};
	float alpha = 37;

	float exp_x[1][5] = {{111, 185, 259, 407, 481}};

	printf("sscal started\n");
	cblas_sscal(5, alpha, (float *)x, 1);
	printf("sscal ended\n");

	return compare((float *)exp_x, (float *)x, 5, 5, "sscal", tolerance);
}

bool test_scopy(double tolerance)
{
	float x[8] = {2,3,5,7,11,13,17,19};
	float y[8];
	
	printf("scopy started\n");
	cblas_scopy(8, x, 1, y, 1);
	printf("scopy ended\n");
	
	return compare((float *)x, (float *)y, 8, 8, "scopy", tolerance);
}

bool test_saxpy(double tolerance)
{
	float x[1][5] = {{3.f,5.f,7.f,11.f,13.f}};
	float y[1][5] = {{17.f,19.f,23.f,29.f,31.f}};
	float alpha = 37.f;

	float exp_y[1][5] = {{128.f, 204.f, 282.f, 436.f, 512.f}};

	printf("saxpy started\n");
	cblas_saxpy(5, alpha, (float *)x, 1, (float *)y, 1);
	printf("saxpy ended\n");

	return compare((float *)exp_y, (float *)y, 5, 5, "saxpy", tolerance);
}

bool test_sdsdot(double tolerance)
{
	float x[1][5] = {{3.f,5.f,7.f,11.f,13.f}};
	float y[5][1] = {{17.f},{19.f},{23.f},{29.f},{31.f}};
	float alpha = 37.f;
	
	printf("sdsdot started\n");
	float res = cblas_sdsdot(5, alpha, (float *)x, 1, (float *)y, 1);
	printf("sdsdot ended\n");
	float exp = 1066;

	return compare(&exp, &res, 1, 1, "sdsdot", tolerance);
}

bool test_sgemv(double tolerance)
{
	float alpha = 3.f;
	float beta = 5.f;
	
	float a_n[4][3] = {
		{3.f,5.f,7.f},
		{11.f,13.f,17.f},
		{19.f,23.f,29.f},
		{31.f,37.f,41.f}};
	float x_n[3][1] = {{2.f},{4.f},{6.f}};
	float y_n[4][1] = {{1.f},{2.f},{3.f},{4.f}};
	
	float a_t[4][3] = {
		{3.f,5.f,7.f},
		{11.f,13.f,17.f},
		{19.f,23.f,29.f},
		{31.f,37.f,41.f}};
	float x_t[4][1] = {{2.f},{4.f},{6.f},{8.f}};
	float y_t[3][1] = {{1.f},{2.f},{3.f}};

	float exp_y_n[4][1] = {{209.f}, {538.f}, {927.f}, {1388.f}};
	float exp_y_t[3][1] = {{1241.f}, {1498.f}, {1767.f}};

	printf("sgemv started\n");
	cblas_sgemv(CblasRowMajor, CblasNoTrans,
			4, 3,
			alpha,
			(float *)a_n, 3,
			(float *)x_n, 1,
			beta,
			(float *)y_n, 1);
	cblas_sgemv(CblasRowMajor, CblasTrans,
			4, 3,
			alpha,
			(float *)a_t, 3,
			(float *)x_t, 1,
			beta,
			(float *)y_t, 1);
	printf("sgemv ended\n");

	return compare((float *)exp_y_n, (float *)y_n, 4, 4, "sgemv_n", tolerance)
		& compare((float *)exp_y_t, (float *)y_t, 3, 3, "sgemv_t", tolerance);
}

bool test_sger(double tolerance)
{
	float a[4][3] = {
		{3.f,5.f,7.f},
		{11.f,13.f,17.f},
		{19.f,23.f,29.f},
		{31.f,37.f,41.f}};
	float alpha = 3.f;
	float x[4][1] = {{1.f},{2.f},{3.f},{4.f}};
	float y[1][3] = {{2.f,4.f,6.f}};

	float exp_a[4][3] = {
		{9.f,17.f,25.f},
		{23.f,37.f,53.f},
		{37.f,59.f,83.f},
		{55.f,85.f,113.f}};

	printf("sger started\n");
	cblas_sger(CblasRowMajor,
			4, 3,
			alpha,
			(float *)x, 1,
			(float *)y, 1,
			(float *)a, 3);
	printf("sger ended\n");

	return compare((float *)exp_a, (float *)a, 12, 3, "sger", tolerance);
}

bool test_sgemm(double tolerance)
{
	float alpha = 2.f;
	float beta = 53.f;
	
	float a_n_n[2][3] = {{3.f,5.f,7.f},{11.f,13.f,17.f}};
	float b_n_n[3][2] = {{19.f,23.f},{29.f,31.f},{37.f,41.f}};
	float c_n_n[2][2] = {{59.f,61.f},{67.f,71.f}};
	float exp_c_n_n[2][2] = {{4049.f, 4255.f}, {5981.f, 6469.f}};

	float a_t_t[3][2] = {{19.f,23.f},{29.f,31.f},{37.f,41.f}};
	float b_t_t[2][3] = {{3.f,5.f,7.f},{11.f,13.f,17.f}};
	float c_t_t[2][2] = {{59.f,61.f},{67.f,71.f}};
	float exp_c_t_t[2][2] = {{4049.f, 5663.f}, {4573.f, 6469.f}};
	
	printf("sgemm started\n");
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			2, 2, 3,
			alpha,
			(float *)a_n_n, 3,
			(float *)b_n_n, 2,
			beta,
			(float *)c_n_n, 2);
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
			2, 2, 3,
			alpha,
			(float *)a_t_t, 2,
			(float *)b_t_t, 3,
			beta,
			(float *)c_t_t, 2);
	printf("sgemm ended\n");

	return compare((float *)exp_c_n_n, (float *)c_n_n, 4, 2, "sgemm_n_n", tolerance)
		& compare((float *)exp_c_t_t, (float *)c_t_t, 4, 2, "sgemm_t_t", tolerance);
}

int main(void)
{
	double tolerance = 1e-7;
	int res = true;

	res &= test_sscal(tolerance);
	res &= test_scopy(tolerance);
	res &= test_saxpy(tolerance);
	res &= test_sdsdot(tolerance);
	res &= test_sgemv(tolerance);
	res &= test_sger(tolerance);
	res &= test_sgemm(tolerance);

	if (res)
	{
		printf("\n *** unit passed *** \n");
		return EXIT_SUCCESS;
	}
	else
	{
		printf("\n *** unit failed *** \n");
		return EXIT_FAILURE;
	}
}

