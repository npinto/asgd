#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "../simple_blas.h"

bool test_sdsdot(double tolerance)
{
	float x[1][5] = {{3,5,7,11,13}};
	float y[5][1] = {{17},{19},{23},{29},{31}};
	float alpha = 37;
	
	printf("testing sdsdot\n");
	float result = cblas_sdsdot(5, alpha, (float *)x, 1, (float *)y, 1);
	printf("exp: %f got: %f\n", 1066.0, result);
	printf("sdsdot done\n");
	if (fabs(result - 1066.0) > tolerance)
	{
		fprintf(stderr, "sdsdot failed\n\n");
		return false;
	}
	else
	{
		fprintf(stdout, "sdsdot passed\n\n");
		return true;
	}
}

bool test_sscal(double tolerance)
{
	float x[1][5] = {{3,5,7,11,13}};
	float alpha = 37;

	float out_x[1][5] = {{111, 185, 259, 407, 481}};

	printf("testing sscal\n");
	cblas_sscal(5, alpha, (float *)x, 1);

	bool res = true;
	for (size_t i = 0; i < 5; ++i)
	{
		printf("exp: %f got: %f\n", out_x[0][i], x[0][i]);
		if (fabs(x[0][i] - out_x[0][i]) > tolerance)
		{
			res = false;
		}
	}
	printf("sscal done\n");
	
	if (res)
	{
		fprintf(stdout, "sscal passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "sscal failed\n\n");
		return false;
	}
}

bool test_saxpy(double tolerance)
{
	float x[1][5] = {{3.f,5.f,7.f,11.f,13.f}};
	float y[1][5] = {{17.f,19.f,23.f,29.f,31.f}};
	float alpha = 37.f;

	float out_y[1][5] = {{128.f, 204.f, 282.f, 436.f, 512.f}};

	printf("testing saxpy\n");
	cblas_saxpy(5, alpha, (float *)x, 1, (float *)y, 1);

	bool res = true;
	for (size_t i = 0; i < 5; ++i)
	{
		printf("exp: %f got: %f\n", out_y[0][i], y[0][i]);
		if (fabs(y[0][i] - out_y[0][i]) > tolerance)
		{
			res = false;
		}
	}
	printf("saxpy done\n");
	
	if (res)
	{
		fprintf(stdout, "saxpy passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "saxpy failed\n\n");
		return false;
	}
}

bool test_sgemm(double tolerance)
{
	float a[2][3] = {{3.f,5.f,7.f},{11.f,13.f,17.f}};
	float b[3][2] = {{19.f,23.f},{29.f,31.f},{37.f,41.f}};
	float alpha = 2.f;
	float beta = 53.f;
	float c[2][2] = {{59.f,61.f},{67.f,71.f}};

	float out_c[2][2] = {{4049.f, 4255.f}, {5981.f, 6469.f}};

	printf("testing sgemm\n");
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			2, 2, 3,
			alpha,
			(float *)a, 3,
			(float *)b, 2,
			beta,
			(float *)c, 2);

	bool res = true;
	for (size_t i = 0; i < 2; ++i)
	{
		for (size_t j = 0; j < 2; ++j)
		{
			printf("exp: %f got: %f\n", out_c[i][j], c[i][j]);
			if (fabs(c[i][j] - out_c[i][j]) > tolerance)
			{
				res = false;
			}
		}
	}
	printf("sgemm done\n");
	
	if (res)
	{
		fprintf(stdout, "sgemm passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "sgemm failed\n\n");
		return false;
	}
}

int main(void)
{
	float tolerance = 1e-7;
	int res = true;

	res &= test_sdsdot(tolerance);
	res &= test_sscal(tolerance);
	res &= test_saxpy(tolerance);
	res &= test_sgemm(tolerance);

	if (res)
	{
		printf("test passed\n");
		return EXIT_SUCCESS;
	}
	else
	{
		printf("test failed\n");
		return EXIT_FAILURE;
	}
}

