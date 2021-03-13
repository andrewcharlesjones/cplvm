import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from scipy.stats import multivariate_normal
from cplvm.approximations.approx_model import ApproximateModel



class CPLVMLogNormalApprox(ApproximateModel):

	def __init__(self, X, Y, k_shared, k_foreground, offset_term=True, num_test_genes=0, is_H0=False):


		self.data_dim, self.num_datapoints_x = X.shape
		self.num_datapoints_y = Y.shape[1]
		self._k_shared = k_shared
		self._k_foreground = k_foreground

		if offset_term:
			# delta
			self.qdeltax_mean = tf.Variable(tf.random.normal([self.data_dim, 1]))
			self.qdeltax_stddv = tfp.util.TransformedVariable(
			  1e-4 * tf.ones([self.data_dim, 1]),
			  bijector=tfb.Softplus())

		self.qsize_factor_x_mean = tf.Variable(tf.random.normal([1, self.num_datapoints_x]))
		self.qsize_factor_x_stddv = tfp.util.TransformedVariable(
			1e-4 * tf.ones([1, self.num_datapoints_x]),
			bijector=tfb.Softplus())

		self.qsize_factor_y_mean = tf.Variable(tf.random.normal([1, self.num_datapoints_y]))
		self.qsize_factor_y_stddv = tfp.util.TransformedVariable(
			1e-4 * tf.ones([1, self.num_datapoints_y]),
			bijector=tfb.Softplus())

		# S
		self.qs_mean = tf.Variable(tf.random.normal([self.data_dim, self._k_shared]))
		self.qw_mean = tf.Variable(tf.random.normal([self.data_dim - num_test_genes, self._k_foreground]))
		self.qzx_mean = tf.Variable(tf.random.normal([self._k_shared, self.num_datapoints_x]))
		self.qzy_mean = tf.Variable(tf.random.normal([self._k_shared, self.num_datapoints_y]))
		self.qty_mean = tf.Variable(tf.random.normal([self._k_foreground, self.num_datapoints_y]))

		self.qs_stddv = tfp.util.TransformedVariable(
		  1e-4 * tf.ones([self.data_dim, self._k_shared]),
		  bijector=tfb.Softplus())
		self.qw_stddv = tfp.util.TransformedVariable(
		  1e-4 * tf.ones([self.data_dim - num_test_genes, self._k_foreground]),
		  bijector=tfb.Softplus())
		self.qzx_stddv = tfp.util.TransformedVariable(
			1e-4 * tf.ones([self._k_shared, self.num_datapoints_x]),
			bijector=tfb.Softplus())
		self.qzy_stddv = tfp.util.TransformedVariable(
			1e-4 * tf.ones([self._k_shared, self.num_datapoints_y]),
			bijector=tfb.Softplus())
		self.qty_stddv = tfp.util.TransformedVariable(
			1e-4 * tf.ones([self._k_foreground, self.num_datapoints_y]),
			bijector=tfb.Softplus())

		def factored_normal_variational_model():

			if offset_term:
				qdeltax = yield tfd.LogNormal(loc=self.qdeltax_mean, scale=self.qdeltax_stddv, name="qdeltax")

			qsize_factor_x = yield tfd.LogNormal(loc=self.qsize_factor_x_mean,
						 scale=self.qsize_factor_x_stddv,
						 name="qsize_factor_x")

			qsize_factor_y = yield tfd.LogNormal(loc=self.qsize_factor_y_mean,
						 scale=self.qsize_factor_y_stddv,
						 name="qsize_factor_y")

			qs = yield tfd.LogNormal(loc=self.qs_mean, scale=self.qs_stddv, name="qs")
			qzx = yield tfd.LogNormal(loc=self.qzx_mean, scale=self.qzx_stddv, name="qzx")
			qzy = yield tfd.LogNormal(loc=self.qzy_mean, scale=self.qzy_stddv, name="qzy")

			if not is_H0:
				qw = yield tfd.LogNormal(loc=self.qw_mean, scale=self.qw_stddv, name="qw")
				qty = yield tfd.LogNormal(loc=self.qty_mean, scale=self.qty_stddv, name="qty")

		# Surrogate posterior that we will try to make close to p
		surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
			factored_normal_variational_model)

		self.approximate_posterior = surrogate_posterior


