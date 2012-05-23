"""Averaging Stochastic Gradient Descent Classifier

naive, non-optimized implementation
"""
import ctypes as ct
import numpy as np
from numpy import dot
from itertools import izip


class CASGD(object):

	def __init__(
		self,
		n_classes,
		n_features,
		sgd_step_size0=1e-2,
		l2_regularization=1e-3,
		n_iterations=10,
		feedback=False,
		dtype=np.float32):
		
		self.n_classes = n_classes
		self.n_features = n_features
		self.n_iterations = n_iterations
		self.feedback = feedback
		
		assert l2_regularization > 0
		self.l2_regularization = l2_regularization
		self.dtype = dtype
		
		self.sgd_weights = np.zeros((n_features,n_classes), dtype=dtype)
		self.sgd_bias = np.zeros((n_classes), dtype=dtype)
		self.sgd_step_size0 = sgd_step_size0
		self.sgd_step_size = sgd_step_size0
		self.sgd_step_size_scheduling_exponent = 2. / 3
		self.sgd_step_size_scheduling_multiplier = l2_regularization
		
		self.asgd_weights = np.zeros((n_features,n_classes), dtype=dtype)
		self.asgd_bias = np.zeros((n_classes), dtype=dtype)
		self.asgd_step_size0 = 1
		self.asgd_step_size = self.asgd_step_size0
		
		self.n_observations = 0
		self.core_lib = ct.CDLL("./asgd_core.so")
	
	def __del__(self):
		#dl = ct.CDLL("libdl.so")
		#dl.dlclose(self.core_lib._handle)
		pass


	def partial_fit(self, X, y, perm, batch_size):
		
		# force ndarrays to point to different data
		if self.sgd_weights is self.asgd_weights:
			self.asgd_weights = self.asgd_weights.copy(order='C')
		if self.sgd_bias is self.asgd_bias:
			self.asgd_bias = self.asgd_bias.copy(order='C')
	
		# require that all arrays are in contiguous C format
		input_req = ['A', 'O', 'C']
		output_req = ['A', 'O', 'W', 'C']
		sgd_weights = np.require(
				self.sgd_weights,
				dtype=np.float32,
				requirements=output_req)
		sgd_bias = np.require(
				self.sgd_bias,
				dtype=np.float32,
				requirements=output_req)
		asgd_weights = np.require(
				self.asgd_weights,
				dtype=np.float32,
				requirements=output_req)
		asgd_bias = np.require(
				self.asgd_bias,
				dtype=np.float32,
				requirements=output_req)

		X = np.require(X, dtype=np.float32, requirements=input_req)
		y = np.require(y, dtype=np.float32, requirements=input_req)

		# convert all parameters to the right C type
		sgd_step_size0 = ct.c_float(self.sgd_step_size0)
		sgd_step_size = ct.c_float(self.sgd_step_size)
		sgd_step_size_scheduling_exponent = \
				ct.c_float(self.sgd_step_size_scheduling_exponent)
		sgd_step_size_scheduling_multiplier = \
				ct.c_float(self.sgd_step_size_scheduling_multiplier)
		asgd_step_size = ct.c_float(self.asgd_step_size)

		l2_regularization = ct.c_float(self.l2_regularization)
		n_observations = ct.c_long(self.n_observations)

		# get array sizes in the right format
		sgd_weights_rows = ct.c_size_t(sgd_weights.shape[0])
		sgd_weights_cols = ct.c_size_t(1)
		if sgd_weights.ndim == 2:
			sgd_weights_cols = ct.c_size_t(sgd_weights.shape[1])

		sgd_bias_rows = ct.c_size_t(sgd_bias.shape[0])
		sgd_bias_cols = ct.c_size_t(1)
		if sgd_bias.ndim == 2:
			sgd_bias_cols = ct.c_size_t(sgd_bias.shape[1])

		asgd_weights_rows = ct.c_size_t(asgd_weights.shape[0])
		asgd_weights_cols = ct.c_size_t(1)
		if asgd_weights.ndim == 2:
			asgd_weights_cols = ct.c_size_t(asgd_weights.shape[1])

		asgd_bias_rows = ct.c_size_t(asgd_bias.shape[0])
		asgd_bias_cols = ct.c_size_t(1)
		if asgd_bias.ndim == 2:
			asgd_bias_cols = ct.c_size_t(asgd_bias.shape[1])

		X_rows = ct.c_size_t(X.shape[0])
		X_cols = ct.c_size_t(1)
		if X.ndim == 2:
			X_cols = ct.c_size_t(X.shape[1])

		y_rows = ct.c_size_t(y.shape[0])
		y_cols = ct.c_size_t(1)
		if y.ndim == 2:
			y_cols = ct.c_size_t(y.shape[1])

		if perm != None:
			size_t_length = str(ct.sizeof(ct.c_size_t))
			perm = np.require(perm, dtype=np.dtype('u'+size_t_length), requirements=input_req)
		
		self.core_lib.core_partial_fit(
				batch_size,
				ct.byref(n_observations),
				ct.byref(sgd_step_size),
				ct.byref(asgd_step_size),
				l2_regularization,
				sgd_step_size0,
				sgd_step_size_scheduling_exponent,
				sgd_step_size_scheduling_multiplier,
				sgd_weights.ctypes.data_as(ct.POINTER(ct.c_float)),
				sgd_weights_rows,
				sgd_weights_cols,
				sgd_bias.ctypes.data_as(ct.POINTER(ct.c_float)),
				sgd_bias_rows,
				sgd_bias_cols,
				asgd_weights.ctypes.data_as(ct.POINTER(ct.c_float)),
				asgd_weights_rows,
				asgd_weights_cols,
				asgd_bias.ctypes.data_as(ct.POINTER(ct.c_float)),
				asgd_bias_rows,
				asgd_bias_cols,
				X.ctypes.data_as(ct.POINTER(ct.c_float)),
				X_rows,
				X_cols,
				y.ctypes.data_as(ct.POINTER(ct.c_float)),
				y_rows,
				y_cols,
				perm.ctypes.data_as(ct.POINTER(ct.c_size_t)))

		# --
		self.sgd_weights = sgd_weights
		self.sgd_bias = sgd_bias
		self.sgd_step_size = sgd_step_size.value

		self.asgd_weights = asgd_weights
		self.asgd_bias = asgd_bias
		self.asgd_step_size = asgd_step_size.value

		self.n_observations = n_observations.value

		return ct.c_int.in_dll(self.core_lib, 'time_count');


	def fit(self, X, y, batch_size):
		
		assert X.ndim == 2
		assert y.ndim == 1
		
		n_points, n_features = X.shape
		assert n_features == self.n_features
		assert n_points == y.size
		
		n_iterations = self.n_iterations
		
		for i in xrange(n_iterations):
			
			idx = np.random.permutation(n_points)
			Xb = X[idx]
			yb = y[idx]
			
			self.partial_fit(Xb, yb, batch_size)
			
			if self.feedback:
				self.sgd_weights = self.asgd_weights
				self.sgd_bias = self.asgd_bias

	def decision_function(self, X):
		return dot(X, self.asgd_weights) + self.asgd_bias

	def predict(self, X):
		return self.decision_function(X).argmax(1)


