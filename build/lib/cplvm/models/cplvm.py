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
from cplvm.models.model import ContrastiveModel

tf.enable_v2_behavior()

warnings.filterwarnings('ignore')

NUM_VI_ITERS = 2000
LEARNING_RATE_VI = 0.05



class CPLVM(ContrastiveModel):

	def __init__(self, k_shared, k_foreground):

		super().__init__(k_shared, k_foreground)


	def model(self, data_dim, num_datapoints_x, num_datapoints_y, counts_per_cell_X, counts_per_cell_Y, is_H0=False, num_test_genes=0, offset_term=True):

		if offset_term:
			deltax = yield tfd.LogNormal(loc=tf.zeros([data_dim, 1]),
						 scale=tf.ones([data_dim, 1]),
						 name="deltax")

		else:
			deltax = tf.ones([data_dim, 1])

		size_factor_x = yield tfd.LogNormal(loc=np.mean(np.log(counts_per_cell_X)) * tf.ones([1, num_datapoints_x]),
											scale=np.std(np.log(counts_per_cell_X)) * tf.ones([1, num_datapoints_x]),
											name="size_factor_x")

		size_factor_y = yield tfd.LogNormal(loc=np.mean(np.log(counts_per_cell_Y)) * tf.ones([1, num_datapoints_y]),
											scale=np.std(np.log(counts_per_cell_Y)) * tf.ones([1, num_datapoints_y]),
											name="size_factor_y")

		s = yield tfd.Gamma(concentration=tf.ones([data_dim, self._k_shared]),
					 rate=tf.ones([data_dim, self._k_shared]),
					 name="w")

		zx = yield tfd.Gamma(concentration=tf.ones([self._k_shared, num_datapoints_x]),
					   rate=tf.math.multiply(1 * tf.ones([self._k_shared, num_datapoints_x]), 1),
					   name="zx")

		zy = yield tfd.Gamma(concentration=tf.ones([self._k_shared, num_datapoints_y]),
					   rate=tf.math.multiply(1 * tf.ones([self._k_shared, num_datapoints_y]), 1),
					   name="zy")

		# Null
		if is_H0:

			x = yield tfd.Poisson(rate=tf.math.multiply(tf.math.multiply(tf.math.multiply(tf.matmul(s, zx), 1), deltax), size_factor_x),
							 name="x")

			y = yield tfd.Poisson(rate=tf.math.multiply(tf.math.multiply(tf.matmul(s, zy), 1), size_factor_y),
								 name="y")


		else:

			w = yield tfd.Gamma(concentration=tf.ones([data_dim - num_test_genes, self._k_foreground]),
					   rate=tf.ones([data_dim - num_test_genes, self._k_foreground]),
					   name="w")

			ty = yield tfd.Gamma(concentration=tf.ones([self._k_foreground, num_datapoints_y]),
					   rate=tf.math.multiply(1 * tf.ones([self._k_foreground, num_datapoints_y]), 1),
					   name="ty")

			x = yield tfd.Poisson(rate=tf.math.multiply(tf.math.multiply(tf.math.multiply(tf.matmul(s, zx), 1), deltax), size_factor_x),
							 name="x")

			
			if num_test_genes != 0:
				w_padding = tf.zeros([num_test_genes, self._k_foreground])
				w = tf.concat([w, w_padding], axis=0)
				
			y = yield tfd.Poisson(rate=tf.math.multiply(tf.math.multiply(tf.matmul(s, zy) + tf.matmul(w, ty), 1), size_factor_y),
								 name="y")


	def _fit_model_vi(self, X, Y, approximate_model, compute_size_factors = True, is_H0 = False, num_test_genes=0, offset_term=True):

		assert X.shape[0] == Y.shape[0]
		data_dim=X.shape[0]
		num_datapoints_x, num_datapoints_y=X.shape[1], Y.shape[1]

		if compute_size_factors:
			counts_per_cell_X=np.mean(X, axis = 0)
			counts_per_cell_X=np.expand_dims(counts_per_cell_X, 0)
			counts_per_cell_Y=np.mean(Y, axis = 0)
			counts_per_cell_Y=np.expand_dims(counts_per_cell_Y, 0)
		else:
			counts_per_cell_X=1.0
			counts_per_cell_Y=1.0

		# ------- Specify model ---------

		concrete_clvm_model=functools.partial(self.model,
												data_dim = data_dim,
												num_datapoints_x = num_datapoints_x,
												num_datapoints_y = num_datapoints_y,
												counts_per_cell_X = counts_per_cell_X,
												counts_per_cell_Y = counts_per_cell_Y,
												is_H0 = is_H0,
												num_test_genes=num_test_genes,
												offset_term=offset_term)

		model=tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

		if is_H0:

			if offset_term:
				def target_log_prob_fn(deltax, size_factor_x, size_factor_y, s, zx, zy): return model.log_prob(
					(deltax, size_factor_x, size_factor_y, s, zx, zy, X, Y))

			else:

				def target_log_prob_fn(size_factor_x, size_factor_y, s, zx, zy): return model.log_prob(
					(size_factor_x, size_factor_y, s, zx, zy, X, Y))

		else:

			if offset_term:
				def target_log_prob_fn(deltax, size_factor_x, size_factor_y, s, zx, zy, w, ty): return model.log_prob(
					(deltax, size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y))
				

			else:
				def target_log_prob_fn(size_factor_x, size_factor_y, s, zx, zy, w, ty): return model.log_prob(
					(size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y))


		# --------- Fit variational inference model using MC samples and gradient descent ----------

		losses = tfp.vi.fit_surrogate_posterior(
			target_log_prob_fn,
			surrogate_posterior=approximate_model.approximate_posterior,
			optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI),
			num_steps=NUM_VI_ITERS)

		if is_H0:
			return_dict = {
				'loss_trace': losses,
				'qs_mean': qs_mean,
				'qzx_mean': qzx_mean,
				'qzy_mean': qzy_mean,
				'qs_stddv': qs_stddv,
				'qzx_stddv': qzx_stddv,
				'qzy_stddv': qzy_stddv,
				'qdeltax_mean': qdeltax_mean,
				'qdeltax_stddv': qdeltax_stddv,
			}
		else:

			if offset_term:
				return_dict = {
					'loss_trace': losses,
					'qs_mean': qs_mean,
					'qw_mean': qw_mean,
					'qzx_mean': qzx_mean,
					'qzy_mean': qzy_mean,
					'qty_mean': qty_mean,
					'qs_stddv': qs_stddv,
					'qw_stddv': qw_stddv,
					'qzx_stddv': qzx_stddv,
					'qzy_stddv': qzy_stddv,
					'qty_stddv': qty_stddv,
					'qdeltax_mean': qdeltax_mean,
					'qdeltax_stddv': qdeltax_stddv,
					'qsize_factors_x_mean': qsize_factor_x_mean,
					'qsize_factor_x_stddv': qsize_factor_x_stddv,
					'qsize_factors_y_mean': qsize_factor_y_mean,
					'qsize_factor_y_stddv': qsize_factor_y_stddv
				}
			else:
				return_dict = {
					'loss_trace': losses,
					'qs_mean': qs_mean,
					'qw_mean': qw_mean,
					'qzx_mean': qzx_mean,
					'qzy_mean': qzy_mean,
					'qty_mean': qty_mean,
					'qs_stddv': qs_stddv,
					'qw_stddv': qw_stddv,
					'qzx_stddv': qzx_stddv,
					'qzy_stddv': qzy_stddv,
					'qty_stddv': qty_stddv,
					'qsize_factors_x_mean': qsize_factor_x_mean,
					'qsize_factor_x_stddv': qsize_factor_x_stddv,
					'qsize_factors_y_mean': qsize_factor_y_mean,
					'qsize_factor_y_stddv': qsize_factor_y_stddv
				}

		return return_dict


	def _fit_model_map(self, X, Y, compute_size_factors = True, is_H0 = False, num_test_genes=0, offset_term=True):

		assert X.shape[0] == Y.shape[0]
		data_dim=X.shape[0]
		num_datapoints_x, num_datapoints_y=X.shape[1], Y.shape[1]

		if compute_size_factors:
			counts_per_cell_X=np.mean(X, axis = 0)
			counts_per_cell_X=np.expand_dims(counts_per_cell_X, 0)
			counts_per_cell_Y=np.mean(Y, axis = 0)
			counts_per_cell_Y=np.expand_dims(counts_per_cell_Y, 0)
		else:
			counts_per_cell_X=1.0
			counts_per_cell_Y=1.0

		# ------- Specify model ---------

		concrete_clvm_model=functools.partial(clvm,
												data_dim = data_dim,
												latent_dim_shared = latent_dim_shared,
												latent_dim_target = latent_dim_target,
												num_datapoints_x = num_datapoints_x,
												num_datapoints_y = num_datapoints_y,
												counts_per_cell_X = counts_per_cell_X,
												counts_per_cell_Y = counts_per_cell_Y,
												is_H0 = is_H0,
												num_test_genes=num_test_genes,
												offset_term=offset_term)

		model=tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

		if is_H0:

			if offset_term:
				def target_log_prob_fn(size_factor_x, size_factor_y, s, zx, zy): return model.log_prob(
					(size_factor_x, size_factor_y, s, zx, zy, X, Y))

			else:

				def target_log_prob_fn(deltax, size_factor_x, size_factor_y, s, zx, zy): return model.log_prob(
					(deltax, size_factor_x, size_factor_y, s, zx, zy, X, Y))

		else:

			if offset_term:
				def target_log_prob_fn(deltax, size_factor_x, size_factor_y, s, zx, zy, w, ty): return model.log_prob(
					(deltax, size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y))
				

			else:
				def target_log_prob_fn(size_factor_x, size_factor_y, s, zx, zy, w, ty): return model.log_prob(
					(size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y))



		qzy_stddv = tfp.util.TransformedVariable(
			1e-4 * tf.ones([self._k_shared, num_datapoints_y]),
			bijector=tfb.Softplus())

		mu_x = tf.Variable(tf.random.normal([data_dim, 1]))
		mu_y = tf.Variable(tf.random.normal([data_dim, 1]))
		s = tf.Variable(tf.random.normal([data_dim, self._k_shared]))
		w = tf.Variable(tf.random.normal([data_dim, self._k_foreground]))
		zx = tf.Variable(tf.random.normal([self._k_shared, num_datapoints_x]))
		zy = tf.Variable(tf.random.normal([self._k_shared, num_datapoints_y]))
		ty = tf.Variable(tf.random.normal([self._k_foreground, num_datapoints_y]))

		target_log_prob_fn = lambda mu_x, mu_y, s, zx, zy, w, ty: model.log_prob((mu_x, mu_y, s, zx, zy, w, ty, X, Y))
		losses = tfp.math.minimize(
			lambda: -target_log_prob_fn(mu_x, mu_y, s, zx, zy, w, ty),
			optimizer=tf.optimizers.Adam(learning_rate=0.05),
			num_steps=1000)

		if is_H0:
			return_dict = {
				'loss_trace': losses,
				'qs_mean': qs_mean,
				'qzx_mean': qzx_mean,
				'qzy_mean': qzy_mean,
				'qs_stddv': qs_stddv,
				'qzx_stddv': qzx_stddv,
				'qzy_stddv': qzy_stddv,
			}
		else:
			return_dict = {
				'loss_trace': losses,
				'mu_x': mu_x,
				'mu_y': mu_y,
				's': s,
				'zx': zx,
				'zy': zy,
				'w': w,
				'ty': ty
			}

		return return_dict
