#include "asgd_core.h"

#include <math.h>

#include "asgd_blas.h"

void core_partial_fit(
		long *n_observs,
		float *sgd_step_size,
		float *asgd_step_size,

		float l2_reg,
		float sgd_step_size0,
		float sgd_step_size_scheduling_exp,
		float sgd_step_size_scheduling_mul,

		float* sgd_weights,
		size_t sgd_weights_rows,
		size_t sgd_weights_cols,

		float* sgd_bias,
		size_t sgd_bias_rows,
		size_t sgd_bias_cols,

		float *asgd_weights,
		size_t asgd_weights_rows,
		size_t asgd_weights_cols,

		float *asgd_bias,
		size_t asgd_bias_rows,
		size_t asgd_bias_cols,
		
		float *X,
		size_t X_rows,
		size_t X_cols,
		
		float *y,
		size_t y_rows,
		size_t y_cols)
{

	for (size_t i = 0; i < X_rows; ++i) {

		// compute margin //
		// TODO sgd_weights will become a matrix
		// notice that each row in X is also a column because of the stride
		float margin = y[y_cols*i] * 
			cblas_sdsdot(
				X_cols,
				sgd_bias[0],
				X+X_cols*i, 1,
				sgd_weights, 1);

		// update sgd //
		if (l2_reg != 0)
		{
			// TODO sgd_weights will become a matrix
			cblas_sscal(sgd_weights_rows,
					1 - l2_reg * *sgd_step_size,
					sgd_weights, 1);
		}

		if (margin < 1)
		{
			// TODO sgd_weights will become a matrix
			// TODO may be faster to leave sgd_weights on the stack
			cblas_saxpy(
					sgd_weights_rows, 
					*sgd_step_size * y[y_cols*i],
					X+X_cols*i, 1,
					sgd_weights, 1);

			// TODO sgd_bias will become a vector
			sgd_bias[0] = *sgd_step_size * y[y_cols*i];
		}

		// update asgd //
		cblas_sscal(asgd_weights_rows,
				1 - *asgd_step_size,
				asgd_weights, 1);
		
		cblas_saxpy(asgd_weights_rows,
				*asgd_step_size,
				sgd_weights, 1,
				asgd_weights, 1);

		asgd_bias[0] =
			(1 - *asgd_step_size) * asgd_bias[0]
			+ *asgd_step_size * sgd_bias[0];

		// update step_sizes //
		*n_observs += 1;
		
		float sgd_step_size_scheduling =
			1 + sgd_step_size0 * *n_observs * sgd_step_size_scheduling_mul;
		
		*sgd_step_size = sgd_step_size0 /
			pow(sgd_step_size_scheduling, sgd_step_size_scheduling_exp);
		
		*asgd_step_size = 1.0f / *n_observs;
	}
}

