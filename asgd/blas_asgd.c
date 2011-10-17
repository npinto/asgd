#include "blas_asgd.h"

#include <cblas.h>

matrix_t *matrix_init(
		size_t rows,
		size_t cols,
		float val)
{
	matrix_t *m = malloc(sizeof(*m));
	m->rows = rows;
	m->cols = cols;
	m->data = malloc(rows*cols*sizeof(*m->data));

	for (long i = 0; i < rows * cols; ++i)
	{
		m->data[i] = val;
	}

	return m;
}

void matrix_destr(matrix_t *m)
{
	free(m->data);
	free(m);
}

float matrix_get(matrix_t *m, size_t i, size_t j)
{
	return m->data[i*m->cols + j];
}

void matrix_set(matrix_t *m, size_t i, size_t j, float val)
{
	m->data[i*m->cols + j] = val;
}

float *matrix_row(matrix_t *m, size_t i)
{
	return m->data + i*m->cols;
}

void matrix_copy(matrix_t *dst, const matrix_t *src)
{
	dst->rows = src->rows;
	dst->cols = src->cols;
	
	memcpy(dst->data, src->data, src->cols * src->rows * sizeof(*src->data));
}

matrix_t *matrix_clone(matrix_t *m)
{
	matrix_t *r = malloc(sizeof(*r));
	memcpy(r, m, sizeof(*m));
	r->data = malloc(m->rows * m->cols * sizeof(*m->data));
	memcpy(r->data, m->data, m->rows * m->cols * sizeof(*m->data));
	return r;
}

static void swap(matrix_t *m, size_t j, size_t k, size_t x, size_t y)
{
	float buff = matrix_get(m, j, k);
	matrix_set(m, j, k, matrix_get(m, x, y));
	matrix_set(m, x, y, buff);
}

/**
 * Exits with an error unless a given condition is true
 * @param cond The condition to check
 * @param mex A string to print if the condition is false
 */
void mex_assert(bool cond, const char *mex)
{
	if (__builtin_expect(!cond, false)) {
		fprintf(stderr, "%s\n", mex);
		exit(EXIT_FAILURE);
	}
}

static void durstenfeld_shuffle(matrix_t *m)
{
	srand(time(NULL));
	for (size_t i = m->rows-1; i > 0; --i) {
		size_t j = rand() % (i+1);
		// flip current row with a random row among remaining ones
		for (size_t k = 0; k < m->cols; ++k) {
			swap(m, i, k, j, k);
		}
	}
}

/**
 * Constructor for the Binary ASGD structure
 */
nb_asgd_t *nb_asgd_init(
	uint64_t n_feats,
	float sgd_step_size0,
	float l2_reg,
	uint64_t n_iters,
	bool feedback)
{
	nb_asgd_t *data = malloc(sizeof(*data));
	mex_assert(data != NULL, "cannot allocate nb_asgd");
	data->n_feats = n_feats;
	data->n_iters = n_iters;
	data->feedback = feedback;

	mex_assert(__builtin_expect(l2_reg > 0, true), "invalid l2 regularization");
	data->l2_reg = l2_reg;

	data->sgd_weights = matrix_init(n_feats, 1, 0.0f);
	data->sgd_bias = matrix_init(1, 1, 0.0f);
	data->sgd_step_size = sgd_step_size0;
	data->sgd_step_size0 = sgd_step_size0;

	data->sgd_step_size_scheduling_exp = 2. / 3.;
	data->sgd_step_size_scheduling_mul = l2_reg;

	data->asgd_weights = matrix_init(n_feats, 1, 0.0f);
	data->asgd_bias = matrix_init(1, 1, 0.0f);
	data->asgd_step_size = 1;
	data->asgd_step_size0 = 1;

	data->n_observs = 0;
	return data;
}

/**
 * Destructor for the Binary ASGD structure
 */
void nb_asgd_destr(
		nb_asgd_t *data)
{
	matrix_destr(data->sgd_weights);
	matrix_destr(data->sgd_bias);
	matrix_destr(data->asgd_weights);
	matrix_destr(data->asgd_bias);
	free(data);
}

void partial_fit(
		nb_asgd_t *data,
		matrix_t *X,
		matrix_t *y)
{

	for (size_t i = 0; i < X->rows; ++i) {
		
		// compute margin //
		// TODO sgd_weights will become a matrix
		// notice that each row in X is also a column because of the stride
		float margin = matrix_get(y, i, 0) * 
			cblas_sdsdot(X->cols, matrix_get(data->sgd_bias, 0, 0),
				matrix_row(X, i), 1,
				matrix_row(data->sgd_weights, i), 1);

		// update sgd //
		if (data->l2_reg != 0)
		{
			// TODO sgd_weights will become a matrix
			cblas_sscal(data->sgd_weights->rows,
					1 - data->l2_reg * data->sgd_step_size,
					data->sgd_weights->data, 1);
		}

		if (margin < 1)
		{
			// TODO sgd_weights will become a matrix
			// TODO may be faster to leave sgd_weights on the stack
			matrix_t *sgd_weights = matrix_clone(data->sgd_weights);
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					1, data->sgd_weights->cols, X->cols, 
					data->sgd_step_size * matrix_get(y, i, 0),
					matrix_row(X, i), X->cols,
					sgd_weights->data, sgd_weights->cols,
					1.0f,
					data->sgd_weights->data, 1); // interpret as row
			matrix_destr(sgd_weights);

			// TODO sgd_bias will become a vector
			matrix_set(data->sgd_bias, 0, 0,
					data->sgd_step_size * matrix_get(y, i, 0));
		}

		// update asgd //
		//matrix_t *asgd_weights = matrix_clone(data->asgd_weights);
		cblas_sscal(data->asgd_weights->rows,
				1 - data->asgd_step_size,
				data->asgd_weights->data, 1);
		cblas_saxpy(data->asgd_weights->rows,
				data->asgd_step_size,
				data->sgd_weights->data, 1,
				data->asgd_weights->data, 1);

		matrix_set(data->asgd_bias, 0, 0,
				1 - data->asgd_step_size * matrix_get(data->asgd_bias, 0, 0) +
				data->asgd_step_size * matrix_get(data->sgd_bias, 0, 0));
		
		// update step_sizes //
		data->n_observs += 1;
		float sgd_step_size_scheduling = 1 + data->sgd_step_size0 * data->n_observs
			* data->sgd_step_size_scheduling_mul;
		data->sgd_step_size = data->sgd_step_size0 /
			powf(sgd_step_size_scheduling, data->sgd_step_size_scheduling_exp);
		data->asgd_step_size = 1.0f / data->n_observs;
	}
}

void fit(
	nb_asgd_t *data,
	matrix_t *X,
	matrix_t *y)
{
	mex_assert(X->rows > 1, "fit: X should be a matrix");
	mex_assert(y->cols == 1, "fit: y should be a column vector");

	for (uint64_t i = 0; i < data->n_iters; ++i) {
		durstenfeld_shuffle(X);
		durstenfeld_shuffle(y);
		partial_fit(data, X, y);

		if (data->feedback) {
			matrix_copy(data->sgd_weights, data->asgd_weights);
			matrix_copy(data->sgd_bias, data->asgd_bias);
		}
	}
}

void predict(
	nb_asgd_t *data,
	matrix_t *X,
	matrix_t *r)
{
	// r must be inited to 1s
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
		r->rows, r->cols, data->asgd_weights->rows, 1.0f,
		data->asgd_weights->data, data->asgd_weights->cols,
		X->data, X->cols,
		matrix_get(data->asgd_bias, 0, 0),
		r->data, r->cols);

	for (size_t i = 0; i < r->rows; ++i)
	{
		for (size_t j = 0; j < r->cols; ++j)
		{
			// if the entry is zero, leave it zero
			// otherwise, take the sign
			if (matrix_get(r, i, j) > 0.0f)
			{
				matrix_set(r, i, j, 1.0f);
			}
			else if (matrix_get(r, i, j) < 0.0f)
			{
				matrix_set(r, i, j, -1.0f);
			}
		}
	}
}

