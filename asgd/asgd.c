#include "asgd.h"

#include "asgd_blas.h"
#include "asgd_core.h"

matrix_t *matrix_init(
		size_t rows,
		size_t cols,
		float val)
{
	matrix_t *m = malloc(sizeof(*m));
	m->rows = rows;
	m->cols = cols;
	m->data = malloc(rows*cols*sizeof(*m->data));
	if (m->data == NULL)
	{
		fprintf(stderr, "cannot allocate %zd x %zd  matrix\n", rows, cols);
		exit(EXIT_FAILURE);
	}

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

void matrix_swap(matrix_t *m, size_t j, size_t k, size_t x, size_t y)
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

/**
 * @param r As many random integers (any integer value) as the # of rows - 1
 */
void matrix_row_shuffle(matrix_t *m, int *r)
{
	// do a Durstenfeld shuffle
	for (size_t i = m->rows-1; i > 0; --i) {
		size_t j = r[i-1] % (i+1);
		// flip current row with a random row among remaining ones
		for (size_t k = 0; k < m->cols; ++k) {
			matrix_swap(m, i, k, j, k);
		}
	}
}

/**
 * Constructor for the Binary ASGD structure
 */
nb_asgd_t *nb_asgd_init(
	long n_feats,
	float sgd_step_size0,
	float l2_reg,
	long n_iters,
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
		matrix_t *y,
		size_t *perm,
		size_t batch_size)
{
	core_partial_fit(
			batch_size,
			&data->n_observs,
			&data->sgd_step_size,
			&data->asgd_step_size,
			
			data->l2_reg,
			data->sgd_step_size0,
			data->sgd_step_size_scheduling_exp,
			data->sgd_step_size_scheduling_mul,
			
			data->sgd_weights->data,
			data->sgd_weights->rows,
			data->sgd_weights->cols,

			data->sgd_bias->data,
			data->sgd_bias->rows,
			data->sgd_bias->cols,

			data->asgd_weights->data,
			data->asgd_weights->rows,
			data->asgd_weights->cols,

			data->asgd_bias->data,
			data->asgd_bias->rows,
			data->asgd_bias->cols,

			X->data,
			X->rows,
			X->cols,

			y->data,
			y->rows,
			y->cols,
			perm);
}

void fit(
	nb_asgd_t *data,
	matrix_t *X,
	matrix_t *y,
	int *r,
	size_t batch_size)
{
	mex_assert(X->rows > 1, "fit: X should be a matrix");
	mex_assert(y->cols == 1, "fit: y should be a column vector");

	for (long i = 0; i < data->n_iters; ++i)
	{
		matrix_t *Xb = matrix_clone(X);
		matrix_row_shuffle(Xb, r+i*Xb->rows);
		matrix_t *yb = matrix_clone(y);
		matrix_row_shuffle(yb, r+i*Xb->rows);
		partial_fit(data, Xb, yb, NULL, batch_size);
		matrix_destr(Xb);
		matrix_destr(yb);

		if (data->feedback)
		{
			matrix_copy(data->sgd_weights, data->asgd_weights);
			matrix_copy(data->sgd_bias, data->asgd_bias);
		}
	}
}

void decision_function(
	nb_asgd_t *data,
	matrix_t *X,
	matrix_t *r)
{
	// set the result vector to one, so that
	// when we multiply it becomes the bias vector
	for (size_t i = 0; i < r->rows; ++i)
	{
		for (size_t j = 0; j < r->cols; ++j)
		{
			matrix_set(r, i, j, 1.0f);
		}
	}

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		r->rows, r->cols, data->asgd_weights->rows,
		1.0f,
		X->data, X->cols,
		data->asgd_weights->data, data->asgd_weights->cols,
		matrix_get(data->asgd_bias, 0, 0),
		r->data, r->cols);
}


void predict(
	nb_asgd_t *data,
	matrix_t *X,
	matrix_t *r)
{
	decision_function(data, X, r);

	for (size_t i = 0; i < r->rows; ++i)
	{
		for (size_t j = 0; j < r->cols; ++j)
		{
			// take positive as +1
			// and nonpositive as -1
			if (matrix_get(r, i, j) > 0.0f)
			{
				matrix_set(r, i, j, 1.0f);
			}
			else
			{
				matrix_set(r, i, j, -1.0f);
			}
		}
	}
}

