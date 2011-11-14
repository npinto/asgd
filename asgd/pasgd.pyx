from libc.stdint cimport uint64_t

from libcpp cimport bool

cdef extern from "asgd.h":

	ctypedef struct matrix_t:
		pass
	
	ctypedef struct nb_asgd_t:
		pass

	nb_asgd_t* nb_asgd_init(uint64_t n_feats, float sgd_step_size0, float l2_reg, uint64_t n_iters, bool feedback)

	void nb_asgd_destr(nb_asgd_t* data)

	void fit(nb_asgd_t* data, matrix_t* X, matrix_t* y)

	void partial_fit(nb_asgd_t* data, matrix_t* X, matrix_t* y)

	void decision_function(nb_asgd_t* data, matrix_t* X, matrix_t* r)

	void predict(matrix_t* r)

cdef class ASGD:

	cdef nb_asgd_t* data
	
	def __cinit__(self, uint64_t n_feats, float sgd_step_size0, float l2_reg, uint64_t n_iters, bool feedback):
		self.data = nb_asgd_init(n_feats, sgd_step_size0, l2_reg, n_iters, feedback)
	
	def __dealloc__(self):
		nb_asgd_destr(self.data)

