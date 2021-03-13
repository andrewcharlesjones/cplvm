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


tf.enable_v2_behavior()

warnings.filterwarnings('ignore')

NUM_VI_ITERS = 1000
LEARNING_RATE_VI = 0.01


class CGLVM:

    def __init__(self, k_shared, k_foreground):
        self.k_shared = k_shared
        self.k_foreground = k_foreground


    def model(self, data_dim, num_datapoints_x, num_datapoints_y, counts_per_cell_X, counts_per_cell_Y, is_H0=False):

        mu_x = yield tfd.Normal(loc=tf.zeros([data_dim, 1]),
                     scale=tf.ones([data_dim, 1]),
                     name="mu_x")

        mu_y = yield tfd.Normal(loc=tf.zeros([data_dim, 1]),
                     scale=tf.ones([data_dim, 1]),
                     name="mu_x")

        size_factor_x = yield tfd.LogNormal(loc=np.mean(np.log(counts_per_cell_X)) * tf.ones([1, num_datapoints_x]),
                                            scale=np.std(np.log(counts_per_cell_X)) * tf.ones([1, num_datapoints_x]),
                                            name="size_factor_x")

        size_factor_y = yield tfd.LogNormal(loc=np.mean(np.log(counts_per_cell_Y)) * tf.ones([1, num_datapoints_y]),
                                            scale=np.std(np.log(counts_per_cell_Y)) * tf.ones([1, num_datapoints_y]),
                                            name="size_factor_y")

        s = yield tfd.Normal(loc=tf.zeros([data_dim, self.k_shared]) + 0,
                     scale=tf.ones([data_dim, self.k_shared]),
                     name="w")

        zx = yield tfd.Normal(loc=tf.zeros([self.k_shared, num_datapoints_x]) + 0,
                       scale=tf.ones([self.k_shared, num_datapoints_x]),
                       name="zx")

        zy = yield tfd.Normal(loc=tf.zeros([self.k_shared, num_datapoints_y]) + 0,
                       scale=tf.ones([self.k_shared, num_datapoints_y]),
                       name="zy")

        # Null
        if is_H0:

            x = yield tfd.Poisson(rate=tf.math.multiply(tf.math.exp(tf.math.add(tf.math.multiply(tf.matmul(s, zx), 1), mu_x) + np.log(data_dim)), size_factor_x / data_dim),
                             name="x")

            y = yield tfd.Poisson(rate=tf.math.multiply(tf.math.exp(tf.math.add(tf.matmul(s, zy), mu_y) + np.log(data_dim)), size_factor_y / data_dim),
                                 name="y")


        else:

            w = yield tfd.Normal(loc=tf.zeros([data_dim, self.k_foreground]) + 0,
                       scale=tf.ones([data_dim, self.k_foreground]),
                       name="w")

            ty = yield tfd.Normal(loc=tf.zeros([self.k_foreground, num_datapoints_y]) + 0,
                       scale=tf.ones([self.k_foreground, num_datapoints_y]),
                       name="ty")

            x = yield tfd.Poisson(rate=tf.math.multiply(tf.math.exp(tf.math.add(tf.math.multiply(tf.matmul(s, zx), 1), mu_x)), size_factor_x / data_dim),
                             name="x")

            y = yield tfd.Poisson(rate=tf.math.multiply(tf.math.exp(tf.math.add(tf.matmul(s, zy) + tf.matmul(w, ty), mu_y)), size_factor_y / data_dim),
                                 name="y")


    def fit_model_vi(self, X, Y, approximate_model, compute_size_factors = False, is_H0 = False):

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
                                                is_H0 = is_H0)

        model=tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

        if is_H0:

            def target_log_prob_fn(mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy): return model.log_prob(
                (mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, X, Y))
            # def target_log_prob_fn(mu_x, mu_y, s, zx, zy): return model.log_prob(
            #     (mu_x, mu_y, s, zx, zy, X, Y))

        else:

            def target_log_prob_fn(mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, w, ty): return model.log_prob(
                (mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y))
            # def target_log_prob_fn(mu_x, mu_y, s, zx, zy, w, ty): return model.log_prob(
            #     (mu_x, mu_y, s, zx, zy, w, ty, X, Y))


        # --------- Fit variational inference model using MC samples and gradient descent ----------

        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            surrogate_posterior=approximate_model.approximate_posterior,
            optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI),
            num_steps=NUM_VI_ITERS)

        if is_H0:
            return_dict = {
                'loss_trace': losses,
                'approx_model': approximate_model
            }
        else:
            return_dict = {
                'loss_trace': losses,
                'approx_model': approximate_model
            }

        return return_dict

    def fit_model_map(self, X, Y, compute_size_factors = False, is_H0 = False):

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
                                                is_H0 = is_H0)

        model=tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

        if is_H0:

            def target_log_prob_fn(mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy): return model.log_prob(
                (mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, X, Y))

        else:

            def target_log_prob_fn(mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, w, ty): return model.log_prob(
                (mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y))
            # def target_log_prob_fn(mu_x, mu_y, s, zx, zy, w, ty): return model.log_prob(
            #     (mu_x, mu_y, s, zx, zy, w, ty, X, Y))
            
        # ------- Specify variational families -----------

        mu_x = tf.Variable(tf.random.normal([data_dim, 1]))
        mu_y = tf.Variable(tf.random.normal([data_dim, 1]))
        s = tf.Variable(tf.random.normal([data_dim, self.k_shared]))
        w = tf.Variable(tf.random.normal([data_dim, self.k_foreground]))
        zx = tf.Variable(tf.random.normal([self.k_shared, num_datapoints_x]))
        zy = tf.Variable(tf.random.normal([self.k_shared, num_datapoints_y]))
        ty = tf.Variable(tf.random.normal([self.k_foreground, num_datapoints_y]))

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



