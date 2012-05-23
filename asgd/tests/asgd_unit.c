#include "stdio.h"
#include "stdlib.h"

#include "../asgd.h"
#include "../asgd_core.h"

bool test_swap()
{
	float in[3][5] = {{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}};
	float out[3][5] = {{15,2,3,4,11}, {6,7,8,9,10}, {5,12,13,14,1}};
	matrix_t *m = matrix_init(3, 5, 0.0f);
	memcpy(m->data, in, 3*5*sizeof(*m->data));
	matrix_swap(m, 0, 0, 2, 4);
	matrix_swap(m, 0, 4, 2, 0);
	bool res = !memcmp(m->data, out, 3*5*sizeof(*m->data));
	matrix_destr(m);
	if (res)
	{
		fprintf(stdout, "matrix_swap passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "matrix_Fswap failed\n\n");
		return false;
	}
}

bool test_row_shuffle()
{
	float in[5][3] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}};
	int r[5] = {3,2,1,0};
	float out[5][3] = {{13,14,15},{10,11,12},{7,8,9},{4,5,6},{1,2,3}};
	
	matrix_t *m = matrix_init(5, 3, 0.0f);
	memcpy(m->data, in, 5*3*sizeof(*m->data));
	matrix_row_shuffle(m, r);
	bool res = !memcmp(m->data, out, sizeof(out));
	matrix_destr(m);

	if (res)
	{
		fprintf(stdout, "durstenfeld_shuffle passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "durstenfeld_shuffle failed\n\n");
		return false;
	}
}

bool test_core_partial_fit()
{

	size_t n_points = 5;
	size_t n_feats = 3;
	size_t n_classes = 3;
	float X[5][3] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}};
	float y[5][1] = {{16},{17},{18},{19},{20}};
	size_t perm[5] = {0,1,2,3,4};
	float sgd_weights[3][3] = {{0}};
	float asgd_weights[3][3] = {{0}};
	float sgd_bias[3][1] = {{0}};
	float asgd_bias[3][1] = {{0}};

	// TODO parameters need to be tuned
	unsigned long n_observs = 0;
	float sgd_step_size = 1.f;
	float asgd_step_size = 1.f;
	core_partial_fit(
			2,
			&n_observs,
			&sgd_step_size,
			&asgd_step_size,

			0.01,
			1.0,
			1.0,
			1.0,
			
			(float *)sgd_weights,
			n_feats,
			n_classes,
			
			(float *)sgd_bias,
			n_classes,
			1,

			(float *)asgd_weights,
			n_feats,
			n_classes,
			
			(float *)asgd_bias,
			n_classes,
			1,

			(float *)X,
			n_points,
			n_feats,

			(float *)y,
			n_points,
			1,
			(size_t *)perm);

	return false;
}

bool test_partial_fit(double tolerance)
{
	float Xd[5][3] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}};
	matrix_t *Xm = matrix_init(5, 3, 0.0f);
	memcpy(Xm->data, Xd, sizeof(Xd));

	float yd[5][1] = {{16},{17},{18},{19},{20}};
	matrix_t *ym = matrix_init(5, 1, 0.0f);
	memcpy(ym->data, yd, sizeof(yd));

	float out_weights[3][1] =
	{
		{0.016f}, {0.032f}, {0.048f}
	};
	float out_bias[1][1] = {{0.016f}};

	nb_asgd_t *clf = nb_asgd_init(3, 1e-3f, 1e-6f, 4, false);
	partial_fit(clf, Xm, ym, NULL, 1);

	bool res = true;
	printf("testing partial_fit\n");
	for (size_t i = 0; i < 3; ++i)
	{
		for (size_t j = 0; j < 1; ++j)
		{
			printf("sgd_weights %zu %zu exp: %f got: %f\n",
				i, j,
				out_weights[i][j],
				matrix_get(clf->sgd_weights, i, j));
			if(fabs(matrix_get(clf->sgd_weights, i, j) - out_weights[i][j]) > tolerance)
			{
				res = false;
			}
		}
	}

	printf("sgd_bias exp: %f got: %f\n", out_bias[0][0], clf->sgd_bias->data[0]);
	if (fabs(clf->sgd_bias->data[0] - out_bias[0][0]) > tolerance)
	{
		res = false;
	}

	printf("asgd_bias exp: %f got: %f\n", out_bias[0][0], clf->asgd_bias->data[0]);
	if (fabs(clf->asgd_bias->data[0] - out_bias[0][0]) > tolerance)
	{
		res = false;
	}


	nb_asgd_destr(clf);
	matrix_destr(Xm);
	matrix_destr(ym);

	if (res)
	{
		fprintf(stdout, "partial_fit passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "partial_fit failed\n\n");
		return false;
	}
}

bool test_fit(double tolerance)
{
	float Xd[5][3] = {{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}};
	matrix_t *Xm = matrix_init(5, 3, 0.0f);
	memcpy(Xm->data, Xd, sizeof(Xd));

	float yd[5][1] = {{16},{17},{18},{19},{20}};
	matrix_t *ym = matrix_init(5, 1, 0.0f);
	memcpy(ym->data, yd, sizeof(yd));
	int r[4][4] = {{1,1,0,2},{1,0,0,0},{1,1,2,3},{0,2,1,2}};

	float out_weights[3][1] =
	{
		{0.19f}, {0.209f}, {0.228f}
	};
	float out_bias[1][1] = {{0.019f}};

	nb_asgd_t *clf = nb_asgd_init(3, 1e-3f, 1e-6f, 4, false);
	fit(clf, Xm, ym, (int *)r, 1);

	bool res = true;
	printf("testing fit\n");
	for (size_t i = 0; i < 3; ++i)
	{
		for (size_t j = 0; j < 1; ++j)
		{
			printf("sgd_weights %zu %zu exp: %f got: %f\n",
				i, j,
				out_weights[i][j],
				matrix_get(clf->sgd_weights, i, j));
			if(fabs(matrix_get(clf->sgd_weights, i, j) - out_weights[i][j]) > tolerance)
			{
				res = false;
			}
		}
	}

	printf("sgd_bias exp: %f got: %f\n", out_bias[0][0], clf->sgd_bias->data[0]);
	if (fabs(clf->sgd_bias->data[0] - out_bias[0][0]) > tolerance)
	{
		res = false;
	}

	printf("asgd_bias exp: %f got: %f\n", out_bias[0][0], clf->asgd_bias->data[0]);
	if (fabs(clf->asgd_bias->data[0] - out_bias[0][0]) > tolerance)
	{
		res = false;
	}


	nb_asgd_destr(clf);
	matrix_destr(Xm);
	matrix_destr(ym);

	if (res)
	{
		fprintf(stdout, "fit passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "fit failed\n\n");
		return false;
	}
}

bool test_decision_function(double tolerance)
{
	float Xd[5][3] = {{1,2,3},{4,5,6},{7,8,-9},{10,11,12},{13,14,15}};
	matrix_t *Xm = matrix_init(5, 3, 0.0f);
	memcpy(Xm->data, Xd, sizeof(Xd));

	float yd[5][1] = {{16},{17},{18},{19},{20}};
	matrix_t *ym = matrix_init(5, 1, 0.0f);
	memcpy(ym->data, yd, sizeof(yd));

	matrix_t *rm = matrix_init(5, 1, 0.0f);

	float in_weights[3][1] =
	{
		{0.016f}, {0.032f}, {0.048f}
	};
	float in_bias[1][1] = {{0.016f}};
	float out_r[5][1] = {{0.24f}, {0.528f}, {-0.0479999f}, {1.104f}, {1.392f}};

	nb_asgd_t *clf = nb_asgd_init(3, 1e-3f, 1e-6f, 4, false);
	memcpy(clf->asgd_weights->data, in_weights, sizeof(in_weights));
	memcpy(clf->asgd_bias->data, in_bias, sizeof(in_bias));

	decision_function(clf, Xm, rm);
	bool res = true;
	printf("testing decision_function\n");
	for (size_t i = 0; i < 5; ++i)
	{
		for (size_t j = 0; j < 1; ++j)
		{
			printf("exp: %f got: %f\n",
				out_r[i][j],
				matrix_get(rm, i, j));
			if(fabs(matrix_get(rm, i, j) - out_r[i][j]) > tolerance)
			{
				res = false;
			}
		}
	}

	nb_asgd_destr(clf);
	matrix_destr(Xm);
	matrix_destr(ym);
	matrix_destr(rm);

	if (res)
	{
		fprintf(stdout, "decision_function passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "decision_function failed\n\n");
		return false;
	}
}

bool test_predict(double tolerance)
{
	float Xd[5][3] = {{1,2,3},{4,5,6},{7,8,-9},{10,11,12},{13,14,15}};
	matrix_t *Xm = matrix_init(5, 3, 0.0f);
	memcpy(Xm->data, Xd, sizeof(Xd));

	float yd[5][1] = {{16},{17},{18},{19},{20}};
	matrix_t *ym = matrix_init(5, 1, 0.0f);
	memcpy(ym->data, yd, sizeof(yd));

	matrix_t *rm = matrix_init(5, 1, 0.0f);

	float in_weights[3][1] =
	{
		{0.016f}, {0.032f}, {0.048f}
	};
	float in_bias[1][1] = {{0.016f}};
	float out_r[5][1] = {{1.f}, {1.f}, {-1.f}, {1.f}, {1.f}};

	nb_asgd_t *clf = nb_asgd_init(3, 1e-3f, 1e-6f, 4, false);
	memcpy(clf->asgd_weights->data, in_weights, sizeof(in_weights));
	memcpy(clf->asgd_bias->data, in_bias, sizeof(in_bias));

	predict(clf, Xm, rm);
	bool res = true;
	printf("testing decision_function\n");
	for (size_t i = 0; i < 5; ++i)
	{
		for (size_t j = 0; j < 1; ++j)
		{
			printf("exp: %f got: %f\n",
				out_r[i][j],
				matrix_get(rm, i, j));
			if(fabs(matrix_get(rm, i, j) - out_r[i][j]) > tolerance)
			{
				res = false;
			}
		}
	}

	nb_asgd_destr(clf);
	matrix_destr(Xm);
	matrix_destr(ym);
	matrix_destr(rm);

	if (res)
	{
		fprintf(stdout, "decision_function passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "decision_function failed\n\n");
		return false;
	}
}

int main(void)
{
	double tolerance = 1e-5;
	int res = true;
	res &= test_swap();
	res &= test_row_shuffle();
	res &= test_decision_function(tolerance);
	res &= test_predict(tolerance);
	res &= test_core_partial_fit();
	res &= test_partial_fit(tolerance);
	res &= test_fit(tolerance);

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

