#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include "blas_asgd.h"

int main(
	int argc,
	char *argv[])
{
	srand(time(NULL));
	size_t n_points = 1000;
	size_t n_feats = 100;

	matrix_t *X = matrix_init(n_points, n_feats, 0.0f);
	matrix_t *y = matrix_init(n_points, 1, 0.0f);
	matrix_t *Xtst = matrix_init(n_points, n_feats, 0.0f);
	matrix_t *ytst = matrix_init(n_points, 1, 0.0f);
	for (size_t i = 0; i < n_points; ++i)
	{
		float last0 = powf(-1.0f, rand());
		matrix_set(y, i, 0, last0);
		
		float last1 = powf(-1.0f, rand());
		matrix_set(ytst, i, 0, last1);
		
		for (size_t j = 0; j < n_feats; ++j)
		{
			float val0 = 1.0f * rand() / RAND_MAX;
			val0 = last0 == 1.0f ? val0 + 0.1f : val0;
			matrix_set(X, i, j, val0);
			
			float val1 = 1.0f * rand() / RAND_MAX;
			val1 = last1 == 1.0f ? val1 + 0.1f : val1;
			matrix_set(Xtst, i, j, val1);
		}
	}
	
	nb_asgd_t *clf = nb_asgd_init(n_feats, 1e-3f, 1e-6f, 4, false);
	fit(clf, X, y);
	matrix_t *ytrn_preds = matrix_init(1, n_points, 1.0f);
	predict(clf, X, ytrn_preds);
	matrix_t *ytst_preds = matrix_init(1, n_points, 1.0f);
	predict(clf, Xtst, ytst_preds);

	float ytrn_acc = 0.0f;
	for (size_t i = 0; i < n_points; ++i) {
		if (matrix_get(ytrn_preds, 0, i) == matrix_get(y, i, 0)) {
			ytrn_acc += 1.0f;
		}
	}
	ytrn_acc /= n_points;

	float ytst_acc = 0.0f;
	for (size_t i = 0; i < n_points; ++i) {
		if (matrix_get(ytst_preds, 0, i) == matrix_get(y, i, 0)) {
			ytst_acc += 1.0f;
		}
	}
	ytst_acc /= n_points;
	printf("1st %f (0.723)\n2nd %f (0.513)\n", ytrn_acc, ytst_acc);
	mex_assert(ytrn_acc == 0.723f, "first assert failed");
	mex_assert(ytst_acc == 0.513f, "second assert failed");

	nb_asgd_destr(clf);
	matrix_destr(ytrn_preds);
	matrix_destr(ytst_preds);
	matrix_destr(X);
	matrix_destr(y);
	matrix_destr(Xtst);
	matrix_destr(ytst);
}

