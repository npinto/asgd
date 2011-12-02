from libc.stdint cimport uint64_t
from libcpp cimport bool
from libc cimport stdlib

cimport numpy as np
import numpy as np
np.import_array()

cdef extern from "asgd.h":

	ctypedef struct matrix_t:
		size_t rows
		size_t cols
		float* data
	
	ctypedef struct nb_asgd_t:
		matrix_t* sgd_weights
		matrix_t* sgd_bias
		matrix_t* asgd_weights
		matrix_t* asgd_bias
		size_t n_iters

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
	
	def sgd_weights(self):
		cdef np.npy_intp r_dims[2]
		r_dims[0] = self.data[0].sgd_weights[0].rows
		r_dims[1] = self.data[0].sgd_weights[0].cols
		return np.PyArray_SimpleNewFromData(2, r_dims, np.NPY_FLOAT32, <void*>self.data[0].sgd_weights[0].data)

	def partial_fit(self, np.ndarray X, np.ndarray y):
		# ensure that the matrices are stored in contiguous C format
		if np.PyArray_FLAGS(X) & np.NPY_C_CONTIGUOUS == False:
			msg = 'decision_function: X should be in C contiguous format'
			raise Exception(msg)
		
		if np.PyArray_FLAGS(y) & np.NPY_C_CONTIGUOUS == False:
			msg = 'decision_function: y should be in C contiguous format'
			raise Exception(msg)

		cdef matrix_t Xc
		cdef np.npy_intp* X_dims = np.PyArray_DIMS(X)
		Xc.rows = X_dims[0]
		Xc.cols = X_dims[1]
		Xc.data = <float*>np.PyArray_DATA(X)
		print "np. ",(<float*>np.PyArray_DATA(X))[1]
		print "ar. ",X[1]

		cdef matrix_t yc
		cdef np.npy_intp* y_dims = np.PyArray_DIMS(y)
		yc.rows = y_dims[0]
		yc.cols = y_dims[1]
		yc.data = <float*>np.PyArray_DATA(y)

		partial_fit(self.data, &Xc, &yc)

	def fit(self, np.ndarray X, np.ndarray y):
		# ensure that the matrices are stored in contiguous C format
		if np.PyArray_FLAGS(X) & np.NPY_C_CONTIGUOUS == False:
			msg = 'decision_function: X should be in C contiguous format'
			raise Exception(msg)
		
		if np.PyArray_FLAGS(y) & np.NPY_C_CONTIGUOUS == False:
			msg = 'decision_function: y should be in C contiguous format'
			raise Exception(msg)

		cdef matrix_t Xc
		cdef np.npy_intp* X_dims = np.PyArray_DIMS(X)
		Xc.rows = X_dims[0]
		Xc.cols = X_dims[1]
		Xc.data = <float*>np.PyArray_DATA(X)

		cdef matrix_t yc
		cdef np.npy_intp* y_dims = np.PyArray_DIMS(y)
		yc.rows = y_dims[0]
		yc.cols = y_dims[1]
		yc.data = <float*>np.PyArray_DATA(y)

		cdef int* rc
		cdef np.ndarray r = np.random.random(self.data[0].n_iters*(X_dims[0]-1))
		rc = <int*>np.PyArray_DATA(r)

		fit(self.data, &Xc, &yc, rc)
	
	def decision_function(self, np.ndarray X):
		# ensure that the matrix is stored in contiguous C format
		if np.PyArray_FLAGS(X) & np.NPY_C_CONTIGUOUS == False:
			msg = 'decision_function: X should be in C contiguous format'
			raise Exception(msg)

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

	def predict(self, np.ndarray X):
		# ensure that the matrix is stored in contiguous C format
		if np.PyArray_FLAGS(X) & np.NPY_C_CONTIGUOUS == False:
			msg = 'predict: X should be in C contiguous format'
			raise Exception(msg)

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

