#include "stdio.h"
#include "stdlib.h"

#include "../asgd.h"

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
	matrix_t *m = matrix_init(5, 3, 0.0f);
	memcpy(m->data, in, 5*3*sizeof(*m->data));
	matrix_row_shuffle(m);
	bool got_row0 = false;
	bool got_row1 = false;
	bool got_row2 = false;
	bool got_row3 = false;
	bool got_row4 = false;
	for (size_t i = 0; i < 5; ++i)
	{
		if (!memcmp(m->data+i*3, in[0], 3)) got_row0 = true;
		if (!memcmp(m->data+i*3, in[1], 3)) got_row1 = true;
		if (!memcmp(m->data+i*3, in[2], 3)) got_row2 = true;
		if (!memcmp(m->data+i*3, in[3], 3)) got_row3 = true;
		if (!memcmp(m->data+i*3, in[4], 3)) got_row4 = true;
	}
	matrix_destr(m);

	bool res = got_row0 && got_row1 && got_row2 && got_row3 && got_row4;
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
	partial_fit(clf, Xm, ym);

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
	for (size_t i = 0; i < 1; ++i)
	{
		for (size_t j = 0; j < 5; ++j)
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
	float in_r[5][5] = {
		{0.4324, -0.0297, 243.0920, -98389, 8939843},
		{0.5405, 5489.459, 9845.845, 454.25, -858.954},
		{343.53, 9345.98, 4.5954, 98.455, -80.934},
		{9630.49, 98.2304, -34.0495, 10974.5, 8.7457},
		{-350.785, 30.73407, -14.353, -34.0340, 983.453}};
	matrix_t *mr = matrix_init(5, 5, 0.0f);
	memcpy(mr->data, in_r, sizeof(in_r));
	float out_r[5][5] = {
		{1.0, -1.0, 1.0, -1, 1},
		{1, 1, 1, 1, -1},
		{1, 1, 1, 1, -1},
		{1, 1, -1, 1, 1},
		{-1, 1, -1, -1, 1}};

	predict(mr);

	printf("testing predict\n");
	bool res = true;
	for (size_t i = 0; i < mr->rows; ++i)
	{
		for (size_t j = 0; j < mr->cols; ++j)
		{
			printf("exp: %f got: %f\n",
				out_r[i][j],
				matrix_get(mr, i, j));
			if (fabs(matrix_get(mr, i, j) - out_r[i][j]) > tolerance)
			{
				res = false;
			}
		}
	}

	matrix_destr(mr);
	if (res)
	{
		fprintf(stdout, "predict passed\n\n");
		return true;
	}
	else
	{
		fprintf(stderr, "predict failed\n\n");
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
	res &= test_partial_fit(tolerance);

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

