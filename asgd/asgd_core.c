#include "asgd_blas.h"
#include "asgd_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
	
	float *margin = malloc(sgd_weights_cols*sizeof(*margin));
	float *label = malloc(sgd_weights_cols*sizeof(*label));
	float *obs;
	
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

		// zero out entries that don't violate the margin
		// and turn the remaining ones into a label coefficient
		for (size_t j = 0; j < sgd_weights_cols; ++j)
		{
			label[j] = y[i] == j ? 1.f : -1.f;
			label[j] = label[j] * margin[j] < 1.f ? label[j] : 0.f;
		
			if(abs(label[j]) > 0.f)
			{
				// sgd_weights += sgd_step_size * label * obs
				// sgd_bias += sgd_step_size * label
				cblas_saxpy(
						sgd_weights_rows,
						*sgd_step_size * label[j],
						obs, 1,
						sgd_weights+j, sgd_weights_cols);

				sgd_bias[j] += *sgd_step_size * label[j];
			}
		}
		
		// sgd_weights += sgd_step_size * label * obs
		// sgd_bias += sgd_step_size * label
		/*cblas_sger(CblasRowMajor,
				sgd_weights_rows, sgd_weights_cols,
				*sgd_step_size,
				obs, 1,
				label, 1,
				sgd_weights, sgd_weights_cols);
		
		cblas_saxpy(sgd_weights_cols, 
				*sgd_step_size,
				label, 1,
				sgd_bias, 1);*/

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
	free(label);
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
	
	float *margin = malloc(batch_size*sizeof(*margin));
	float *obs;

	batch_size = 2;
	
	if (asgd_weights_rows % batch_size != 0)
	{
		fprintf(stderr, "%s:%s: %s\n",
				"asgd_core.c",
				"core_partial_fit_minibatch_binary",
				"batch_size must divide n_feats");
		return;
	}

	for (size_t i = 0; i < X_rows; i+=batch_size)
	{
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
						*sgd_step_size * y[i+j],
						obs + j * X_cols, 1,
						sgd_weights, 1);

				// sgd_bias += sgd_step_size * label / batch
				*sgd_bias += *sgd_step_size * y[i+j];
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



void core_partial_fit(
	MACRO_PARTIAL_FIT_PARAMS_DEF
	)
{
	if (batch_size < 2)
	{
		// purely stochastic
		if (sgd_weights_cols == 1)
		{
			core_partial_fit_stochastic_binary(
					MACRO_PARTIAL_FIT_PARAMS_VAL
					);
		}
		else
		{
			core_partial_fit_stochastic_ova(
					MACRO_PARTIAL_FIT_PARAMS_VAL
					);
		}
	}
	else
	{
		// minibatch
		if (sgd_weights_cols == 1)
		{
			core_partial_fit_minibatch_binary(
					MACRO_PARTIAL_FIT_PARAMS_VAL
					);
		}
		else
		{
			/* TODO core_partial_fit_minibatch_ova(
					MACRO_PARTIAL_FIT_PARAMS_VAL
					);*/
		}
	}
}

