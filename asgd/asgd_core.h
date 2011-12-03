#ifndef _ASGD_CORE_H_
#define _ASGD_CORE_H_

#include <stddef.h>

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
		size_t y_cols);

#endif

