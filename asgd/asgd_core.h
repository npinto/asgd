#ifndef _ASGD_CORE_H_
#define _ASGD_CORE_H_

#include <stddef.h>

/* define the parameters of the function as a macro */
/******** BEGIN MACRO_PARTIAL_FIT_PARAMS_DEF ********/
#define MACRO_PARTIAL_FIT_PARAMS_DEF \
		size_t batch_size, \
		unsigned long *n_observs, \
		float *sgd_step_size, \
		float *asgd_step_size, \
\
		float l2_reg, \
		float sgd_step_size0, \
		float sgd_step_size_scheduling_exp, \
		float sgd_step_size_scheduling_mul, \
\
		float* sgd_weights, \
		size_t sgd_weights_rows, \
		size_t sgd_weights_cols, \
\
		float* sgd_bias, \
		size_t sgd_bias_rows, \
		size_t sgd_bias_cols, \
\
		float *asgd_weights, \
		size_t asgd_weights_rows, \
		size_t asgd_weights_cols, \
\
		float *asgd_bias, \
		size_t asgd_bias_rows, \
		size_t asgd_bias_cols, \
\
		float *X, \
		size_t X_rows, \
		size_t X_cols, \
\
		float *y, \
		size_t y_rows, \
		size_t y_cols, \
		size_t *perm

/********* END MACRO_PARTIAL_FIT_PARAMS_DEF *********/

/******** BEGIN MACRO_PARTIAL_FIT_PARAMS_VAL ********/
#define MACRO_PARTIAL_FIT_PARAMS_VAL \
		batch_size, \
		n_observs, \
		sgd_step_size, \
		asgd_step_size, \
\
		l2_reg, \
		sgd_step_size0, \
		sgd_step_size_scheduling_exp, \
		sgd_step_size_scheduling_mul, \
\
		sgd_weights, \
		sgd_weights_rows, \
		sgd_weights_cols, \
\
		sgd_bias, \
		sgd_bias_rows, \
		sgd_bias_cols, \
\
		asgd_weights, \
		asgd_weights_rows, \
		asgd_weights_cols, \
\
		asgd_bias, \
		asgd_bias_rows, \
		asgd_bias_cols, \
\
		X, \
		X_rows, \
		X_cols, \
\
		y, \
		y_rows, \
		y_cols, \
		perm

/********* END MACRO_PARTIAL_FIT_PARAMS_VAL *********/

void core_partial_fit(
	MACRO_PARTIAL_FIT_PARAMS_DEF
	);

#endif

