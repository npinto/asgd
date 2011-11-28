from libc.stdint cimport uint64_t
from libcpp cimport bool
from libc cimport stdlib

cimport numpy as np
np.import_array()

cdef extern from "asgd.h":

	ctypedef struct matrix_t:
		size_t rows
		size_t cols
		float* data
	
	ctypedef struct nb_asgd_t:
		matrix_t* asgd_weights

	nb_asgd_t* nb_asgd_init(uint64_t n_feats, float sgd_step_size0, float l2_reg, uint64_t n_iters, bool feedback)

	void nb_asgd_destr(nb_asgd_t* data)

	matrix_t* matrix_init(size_t rows, size_t cols, float val)

	matrix_t* matrix_destr(matrix_t* m)

	void fit(nb_asgd_t* data, matrix_t* X, matrix_t* y, int* r)

	void partial_fit(nb_asgd_t* data, matrix_t* X, matrix_t* y)

	void decision_function(nb_asgd_t* data, matrix_t* X, matrix_t* r)

	void predict(matrix_t* r)

cdef class ASGD:

	cdef nb_asgd_t* data
	
	def __cinit__(self, uint64_t n_feats, float sgd_step_size0, float l2_reg, uint64_t n_iters, bool feedback):
		self.data = nb_asgd_init(n_feats, sgd_step_size0, l2_reg, n_iters, feedback)
	
	def __dealloc__(self):
		nb_asgd_destr(self.data)
	
	def decision_function(self, np.ndarray X):

		cdef matrix_t Xc
		cdef np.npy_intp* X_dims = np.PyArray_DIMS(X)
		Xc.rows = X_dims[0]
		Xc.cols = X_dims[1]
		Xc.data = <float*>np.PyArray_DATA(X)
		
		cdef np.npy_intp r_dims[2]
		r_dims[0] = X_dims[0]
		r_dims[1] = self.data[0].asgd_weights.cols
		cdef r = np.PyArray_SimpleNew(2, r_dims, np.NPY_FLOAT32)
		
		cdef matrix_t rc
		rc.rows = r_dims[0]
		rc.cols = r_dims[1]
		rc.data = <float*>np.PyArray_DATA(r)
		
		decision_function(self.data, &Xc, &rc);
		return r

