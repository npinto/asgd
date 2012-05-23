#include "asgd_blas.h"
#include "asgd_core.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// MACRO_PARTIAL_FIT_PARAMS_DEF is defined in asgd_core.h
// MACRO_PARTIAL_FIT_PARAMS_VAL is defined in asgd_core.h

static void core_partial_fit_stochastic_binary(
	MACRO_PARTIAL_FIT_PARAMS_DEF
	)
{
	// v x v
	// sgd_weights = n_feats x 1
	// sgd_bias = 1 x 1
	// X = n_points x n_feats
	// y = n_points x 1
	
	float *obs;
	for (size_t i = 0; i < X_rows; ++i)
	{
		obs = X + i * X_cols;
		
		// compute margin //
		// margin = label (obs * sgd_weights + sgd_bias)
		float margin = y[i] * 
			cblas_sdsdot(
				X_cols,
				*sgd_bias,
				obs, 1,
				sgd_weights, 1);

		// update sgd //
		if (l2_reg != 0.f)
		{
			// sgd_weights *= (1 - l2_reg * sgd_step_size)
			cblas_sscal(sgd_weights_rows * sgd_weights_cols,
					1 - l2_reg * *sgd_step_size,
					sgd_weights, 1);
		}

		if (margin < 1)
		{
			// sgd_weights += sgd_step_size * label * obs
			cblas_saxpy(
					sgd_weights_rows, 
					*sgd_step_size * y[i],
					obs, 1,
					sgd_weights, 1);

			// sgd_bias += sgd_step_size * label
			*sgd_bias += *sgd_step_size * y[i];
		}

		// update asgd //
		// asgd_weights = (1 - asgd_step_size) * asgd_weights + asgd_step_size * sgd_weights
		cblas_sscal(asgd_weights_rows,
				1 - *asgd_step_size,
				asgd_weights, 1);
		
		cblas_saxpy(asgd_weights_rows,
				*asgd_step_size,
				sgd_weights, 1,
				asgd_weights, 1);

		// asgd_bias = (1 - asgd_step_size) * asgd_bias + asgd_step_size * sgd_bias
		*asgd_bias =
			(1 - *asgd_step_size) * *asgd_bias
			+ *asgd_step_size * *sgd_bias;

		// update step_sizes //
		*n_observs += 1;

		float sgd_step_size_scheduling =
			1 + sgd_step_size0 * *n_observs * sgd_step_size_scheduling_mul;

		*sgd_step_size = sgd_step_size0 /
			pow(sgd_step_size_scheduling, sgd_step_size_scheduling_exp);

		*asgd_step_size = 1.0f / *n_observs;
	}
}


static void core_partial_fit_stochastic_ova(
	MACRO_PARTIAL_FIT_PARAMS_DEF
	)
{
	// v x M
	// sgd_weights = n_feats x n_classes
	// sgd_bias = n_classes x 1
	// X = n_points x n_feats
	// y = n_points x 1
	
	size_t margin_rows = 1;
	size_t margin_cols = sgd_weights_cols;		
	float *margin = malloc(margin_rows * margin_cols * sizeof(*margin));
	float *obs;
	assert(margin != NULL);

	for (size_t i = 0; i < X_rows; ++i)
	{
		obs = X + i * X_cols;

		// compute unlabeled margin //
		// margin = obs * sgd_weights + sgd_bias
		cblas_scopy(sgd_weights_cols, sgd_bias, 1, margin, 1);
		cblas_sgemv(CblasRowMajor, CblasTrans,
				sgd_weights_rows, sgd_weights_cols,
				1.f,
				sgd_weights, sgd_weights_cols,
				obs, 1,
				1.f,
				margin, 1);

		// update sgd //
		if (l2_reg != 0.f)
		{
			// sgd_weights *= (1 - l2_reg * sgd_step_size)
			cblas_sscal(sgd_weights_rows * sgd_weights_cols,
					1 - l2_reg * *sgd_step_size,
					sgd_weights, 1);
		}

		for (size_t j = 0; j < sgd_weights_cols; ++j)
		{
			float label = (y[i] == j) ? 1.f : -1.f;
			if (label * margin[j] < 1.f)
			{
				// sgd_weights += sgd_step_size * label * obs
				// sgd_bias += sgd_step_size * label
				cblas_saxpy(
						sgd_weights_rows,
						*sgd_step_size * label,
						obs, 1,
						sgd_weights+j, sgd_weights_cols);

				sgd_bias[j] += *sgd_step_size * label;
			}
		}

		// update asgd //
		// asgd_weights = (1 - asgd_step_size) * asgd_weights + asgd_step_size * sgd_weights
		cblas_sscal(asgd_weights_rows * asgd_weights_cols,
				1.f - *asgd_step_size,
				asgd_weights, 1);

		cblas_saxpy(asgd_weights_rows * asgd_weights_cols,
				*asgd_step_size,
				sgd_weights, 1,
				asgd_weights, 1);

		// asgd_bias = (1 - asgd_step_size) * asgd_bias + asgd_step_size * sgd_bias
		cblas_sscal(asgd_bias_rows * asgd_bias_cols,
				1.f - *asgd_step_size,
				asgd_bias, 1);

		cblas_saxpy(asgd_bias_rows * asgd_bias_cols,
				*asgd_step_size,
				sgd_bias, 1,
				asgd_bias, 1);

		// update step_sizes //
		*n_observs += 1;

		float sgd_step_size_scheduling =
			1.f + sgd_step_size0 * *n_observs * sgd_step_size_scheduling_mul;

		*sgd_step_size = sgd_step_size0 /
			pow(sgd_step_size_scheduling, sgd_step_size_scheduling_exp);

		*asgd_step_size = 1.f / *n_observs;
	}

	free(margin);
}



static void core_partial_fit_minibatch_binary(
	MACRO_PARTIAL_FIT_PARAMS_DEF
	)
{
	// M x v
	// sgd_weights = n_feats x 1
	// sgd_bias = 1 x 1
	// X = n_points x n_feats
	// y = n_points x 1
	
	size_t margin_rows = batch_size;
	size_t margin_cols = 1;		
	float *margin = malloc(margin_rows * margin_cols * sizeof(*margin));
	float *obs;
	assert(margin != NULL);

	for (size_t i = 0; i < X_rows; i += batch_size)
	{
		// the last iteration might require a smaller batch
		// in case X_rows % batch_size != 0
		if (i + batch_size > X_rows)
		{
			batch_size = X_rows - i;
		}

		obs = X + i * X_cols;
		
		// compute margin //
		// margin = label * (obs * sgd_weights + sgd_bias)
		for (size_t j = 0; j < batch_size; ++j)
		{
			margin[j] = *sgd_bias;
		}
		cblas_sgemv(CblasRowMajor, CblasNoTrans,
				batch_size, X_cols,
				1.f,
				obs, X_cols,
				sgd_weights, 1,
				1.f,
				margin, 1);
		
		for (size_t j = 0; j < batch_size; ++j)
		{
			margin[j] *= y[i + j];
		}

		// update sgd //
		if (l2_reg != 0.f)
		{
			// sgd_weights *= (1 - l2_reg * sgd_step_size)
			cblas_sscal(sgd_weights_rows * sgd_weights_cols,
					1 - l2_reg * *sgd_step_size,
					sgd_weights, 1);
		}

		for (size_t j = 0; j < batch_size; ++j)
		{
			if (margin[j] < 1)
			{
				// sgd_weights += (sgd_step_size * label / batch) * obs
				cblas_saxpy(
						sgd_weights_rows, 
						*sgd_step_size * y[i+j] / batch_size,
						obs + j * X_cols, 1,
						sgd_weights, 1);

				// sgd_bias += sgd_step_size * label / batch
				*sgd_bias += *sgd_step_size * y[i+j] / batch_size;
			}
		}

		// update asgd //
		// asgd_weights = (1 - asgd_step_size) * asgd_weights + asgd_step_size * sgd_weights
		cblas_sscal(asgd_weights_rows,
				1 - *asgd_step_size,
				asgd_weights, 1);
		
		cblas_saxpy(asgd_weights_rows,
				*asgd_step_size,
				sgd_weights, 1,
				asgd_weights, 1);

		// asgd_bias = (1 - asgd_step_size) * asgd_bias + asgd_step_size * sgd_bias
		*asgd_bias =
			(1 - *asgd_step_size) * *asgd_bias
			+ *asgd_step_size * *sgd_bias;

		// update step_sizes //
		*n_observs += 1;

		float sgd_step_size_scheduling =
			1 + sgd_step_size0 * *n_observs * sgd_step_size_scheduling_mul;

		*sgd_step_size = sgd_step_size0 /
			pow(sgd_step_size_scheduling, sgd_step_size_scheduling_exp);

		*asgd_step_size = 1.0f / *n_observs;
	}

	free(margin);
}

static void core_partial_fit_minibatch_ova(
	MACRO_PARTIAL_FIT_PARAMS_DEF
	)
{
	// M x M
	// sgd_weights = n_feats x n_classes
	// sgd_bias = n_classes x 1
	// X = n_points x n_feats
	// y = n_points x 1
	
	assert(sgd_weights_rows == asgd_weights_rows);
	assert(sgd_weights_cols == asgd_weights_cols);
	assert(sgd_bias_rows == asgd_bias_rows);
	assert(sgd_bias_cols == asgd_bias_cols);
	
	assert(sgd_weights_cols == sgd_bias_rows); // classes
	assert(X_rows == y_rows); // points
	assert(X_cols == sgd_weights_rows); // feats
	size_t n_points = X_rows;
	size_t n_feats = X_cols;
	size_t n_classes = sgd_weights_cols;
	assert(batch_size <= n_points);

	float *margin = malloc(batch_size * n_classes * sizeof(*margin));
	float *obs;
	assert(margin != NULL);

	for (size_t i=0; i < n_points; i += batch_size)
	{
		// the last iteration might require a smaller batch
		// in case X_rows % batch_size != 0
		if (i + batch_size > n_points)
		{
			batch_size = n_points - i;
		}

		obs = X + i * X_cols;
		
		// compute margin //
		// margin = label * (obs * sgd_weights + sgd_bias)
		for (size_t j = 0; j < batch_size; ++j)
		{
			float *margin_row = margin + j * n_classes;
			cblas_scopy(n_classes, sgd_bias, 1, margin_row, 1);
		}
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				batch_size, n_classes, n_feats,
				1.f,
				obs, n_feats,
				sgd_weights, n_classes,
				1.f,
				margin, n_classes);


		// update sgd //
		if (l2_reg != 0.f)
		{
			// sgd_weights *= (1 - l2_reg * sgd_step_size)
			cblas_sscal(n_feats * n_classes,
					1 - l2_reg * *sgd_step_size,
					sgd_weights, 1);
		}

		for (size_t k = 0; k < batch_size; ++k)
		{
			for (size_t j = 0; j < n_classes; ++j)
			{
				size_t index = k * n_classes + j;
				float label = y[i+k] == j ? 1.f : -1.f;
				label = label * margin[index] < 1.f ? label : 0.f;
	
				if(fabs(label) > 0.f)
				{
					// sgd_weights += sgd_step_size * label * obs
					// sgd_bias += sgd_step_size * label
					cblas_saxpy(
							n_feats,
							*sgd_step_size * label / batch_size,
							obs + k * n_feats, 1,
							sgd_weights + j, n_classes);
	
					sgd_bias[j] += *sgd_step_size * label / batch_size;
				}
			}
		}

		// update asgd //
		// asgd_weights = (1 - asgd_step_size) * asgd_weights + asgd_step_size * sgd_weights
		cblas_sscal(n_feats * n_classes,
				1 - *asgd_step_size,
				asgd_weights, 1);
		
		cblas_saxpy(n_feats * n_classes,
				*asgd_step_size,
				sgd_weights, 1,
				asgd_weights, 1);

		// asgd_bias = (1 - asgd_step_size) * asgd_bias + asgd_step_size * sgd_bias
		cblas_sscal(n_classes,
				1.f - *asgd_step_size,
				asgd_bias, 1);

		cblas_saxpy(n_classes,
				*asgd_step_size,
				sgd_bias, 1,
				asgd_bias, 1);

		// update step_sizes //
		*n_observs += 1;

		float sgd_step_size_scheduling =
			1 + sgd_step_size0 * *n_observs * sgd_step_size_scheduling_mul;

		*sgd_step_size = sgd_step_size0 /
			pow(sgd_step_size_scheduling, sgd_step_size_scheduling_exp);

		*asgd_step_size = 1.0f / *n_observs;
	}

	free(margin);
}

int time_count = 0;
static void core_partial_fit_minibatch_ova_shuffle(
	MACRO_PARTIAL_FIT_PARAMS_DEF
	)
{
	// M x M
	// sgd_weights = n_feats x n_classes
	// sgd_bias = n_classes x 1
	// X = n_points x n_feats
	// y = n_points x 1
	
	assert(sgd_weights_rows == asgd_weights_rows);
	assert(sgd_weights_cols == asgd_weights_cols);
	assert(sgd_bias_rows == asgd_bias_rows);
	assert(sgd_bias_cols == asgd_bias_cols);
	
	assert(sgd_weights_cols == sgd_bias_rows); // classes
	assert(X_rows == y_rows); // points
	assert(X_cols == sgd_weights_rows); // feats
	size_t n_points = X_rows;
	size_t n_feats = X_cols;
	size_t n_classes = sgd_weights_cols;
	assert(batch_size <= n_points);

	float *margin = malloc(batch_size * n_classes * sizeof(*margin));
	float *obs = malloc(batch_size * n_feats * sizeof(*obs));
	assert(margin != NULL);
	assert(obs != NULL);

	time_count = 0;
	struct timespec tp1,tp2;
	for (size_t i = 0; i < n_points; i += batch_size)
	{
		// the last iteration might require a smaller batch
		// in case X_rows % batch_size != 0
		if (i + batch_size > n_points)
		{
			batch_size = n_points - i;
		}
		
		// compute margin //
		// margin = label * (obs * sgd_weights + sgd_bias)
		for (size_t j = 0; j < batch_size; ++j)
		{
			cblas_scopy(n_feats, X + perm[i+j] * n_feats, 1, obs + j * n_feats, 1);
		}
		
		clock_gettime(CLOCK_MONOTONIC, &tp1);
		for (size_t j = 0; j < batch_size; ++j)
		{
			float *margin_row = margin + j * n_classes;
			cblas_scopy(n_classes, sgd_bias, 1, margin_row, 1);
		}
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				batch_size, n_classes, n_feats,
				1.f,
				obs, n_feats,
				sgd_weights, n_classes,
				1.f,
				margin, n_classes);

		// update sgd //
		if (l2_reg != 0.f)
		{
			// sgd_weights *= (1 - l2_reg * sgd_step_size)
			cblas_sscal(n_feats * n_classes,
					1 - l2_reg * *sgd_step_size,
					sgd_weights, 1);
		}

		for (size_t k = 0; k < batch_size; ++k)
		{
			for (size_t j = 0; j < n_classes; ++j)
			{
				size_t index = k * n_classes + j;
				float label = y[perm[i+k]] == j ? 1.f : -1.f;
				label = label * margin[index] < 1.f ? label : 0.f;
	
				if(fabs(label) > 0.f)
				{
					// sgd_weights += sgd_step_size * label * obs
					// sgd_bias += sgd_step_size * label
					cblas_saxpy(
							n_feats,
							*sgd_step_size * label / batch_size,
							obs + k * n_feats, 1,
							sgd_weights + j, n_classes);
	
					sgd_bias[j] += *sgd_step_size * label / batch_size;
				}
			}
		}

		// update asgd //
		// asgd_weights = (1 - asgd_step_size) * asgd_weights + asgd_step_size * sgd_weights
		cblas_sscal(n_feats * n_classes,
				1 - *asgd_step_size,
				asgd_weights, 1);
		
		cblas_saxpy(n_feats * n_classes,
				*asgd_step_size,
				sgd_weights, 1,
				asgd_weights, 1);

		// asgd_bias = (1 - asgd_step_size) * asgd_bias + asgd_step_size * sgd_bias
		cblas_sscal(n_classes,
				1.f - *asgd_step_size,
				asgd_bias, 1);

		cblas_saxpy(n_classes,
				*asgd_step_size,
				sgd_bias, 1,
				asgd_bias, 1);

		// update step_sizes //
		*n_observs += 1;

		float sgd_step_size_scheduling =
			1 + sgd_step_size0 * *n_observs * sgd_step_size_scheduling_mul;

		*sgd_step_size = sgd_step_size0 /
			pow(sgd_step_size_scheduling, sgd_step_size_scheduling_exp);

		*asgd_step_size = 1.0f / *n_observs;
		
		clock_gettime(CLOCK_MONOTONIC, &tp2);
		time_count += (tp2.tv_sec - tp1.tv_sec);
	}

	free(margin);
	free(obs);
}

void core_partial_fit(
	MACRO_PARTIAL_FIT_PARAMS_DEF
	)
{
	if (batch_size == 1 && sgd_weights_cols == 1 && perm == NULL)
	{
		core_partial_fit_stochastic_binary(
				MACRO_PARTIAL_FIT_PARAMS_VAL
				);
		return;
	}
	else
	if (batch_size == 1 && sgd_weights_cols == 1 && perm == NULL)
	{
		core_partial_fit_stochastic_ova(
				MACRO_PARTIAL_FIT_PARAMS_VAL
				);
		return;
	}
	else
	if (batch_size > 1 && sgd_weights_cols == 1 && perm == NULL)
	{
		core_partial_fit_minibatch_binary(
				MACRO_PARTIAL_FIT_PARAMS_VAL
				);
		return;
	}
	else
	if (batch_size > 1 && sgd_weights_cols > 1 && perm == NULL)
	{
		core_partial_fit_minibatch_ova(
				MACRO_PARTIAL_FIT_PARAMS_VAL
				);
		return;
	}
	else
	if (batch_size > 1 && sgd_weights_cols > 1 && perm != NULL)
	{
		printf("auto-shuffle\n");
		core_partial_fit_minibatch_ova_shuffle(
				MACRO_PARTIAL_FIT_PARAMS_VAL
				);
		return;
	}

	fprintf(stderr, "unsupported parameters to core_partial_fit\n");
	fprintf(stderr, "batch size: %zu n_classes: %zu permutation: %p\n",
			batch_size, sgd_weights_cols, perm);
}

