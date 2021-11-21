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

warnings.filterwarnings("ignore")

NUM_VI_ITERS = 2000
LEARNING_RATE_VI = 0.05


class CPLVM(ContrastiveModel):
    def __init__(self, k_shared, k_foreground, compute_size_factors=True):

        super().__init__(k_shared, k_foreground)
        self.compute_size_factors = compute_size_factors

    def model(
        self,
        data_dim,
        num_datapoints_x,
        num_datapoints_y,
        counts_per_cell_X,
        counts_per_cell_Y,
        is_H0=False,
        num_test_genes=0,
        offset_term=True,
    ):

        if offset_term:
            deltax = yield tfd.LogNormal(
                loc=tf.zeros([data_dim, 1]), scale=tf.ones([data_dim, 1]), name="deltax"
            )

        else:
            deltax = tf.ones([data_dim, 1])

        if self.compute_size_factors:
            size_factor_x = yield tfd.LogNormal(
                loc=np.mean(np.log(counts_per_cell_X)) * tf.ones([1, num_datapoints_x]),
                scale=np.std(np.log(counts_per_cell_X)) * tf.ones([1, num_datapoints_x]),
                name="size_factor_x",
            )

            size_factor_y = yield tfd.LogNormal(
                loc=np.mean(np.log(counts_per_cell_Y)) * tf.ones([1, num_datapoints_y]),
                scale=np.std(np.log(counts_per_cell_Y)) * tf.ones([1, num_datapoints_y]),
                name="size_factor_y",
            )
        else:
            size_factor_x = counts_per_cell_X
            size_factor_y = counts_per_cell_Y

        s = yield tfd.Gamma(
            concentration=tf.ones([data_dim, self._k_shared]),
            rate=tf.ones([data_dim, self._k_shared]),
            name="s",
        )

        zx = yield tfd.Gamma(
            concentration=tf.ones([self._k_shared, num_datapoints_x]),
            rate=tf.math.multiply(1 * tf.ones([self._k_shared, num_datapoints_x]), 1),
            name="zx",
        )

        zy = yield tfd.Gamma(
            concentration=tf.ones([self._k_shared, num_datapoints_y]),
            rate=tf.math.multiply(1 * tf.ones([self._k_shared, num_datapoints_y]), 1),
            name="zy",
        )

        # Null
        if is_H0:

            x = yield tfd.Poisson(
                rate=tf.math.multiply(
                    tf.math.multiply(tf.math.multiply(tf.matmul(s, zx), 1), deltax),
                    size_factor_x,
                ),
                name="x",
            )

            y = yield tfd.Poisson(
                rate=tf.math.multiply(
                    tf.math.multiply(tf.matmul(s, zy), 1), size_factor_y
                ),
                name="y",
            )

        else:

            w = yield tfd.Gamma(
                concentration=tf.ones([data_dim - num_test_genes, self._k_foreground]),
                rate=tf.ones([data_dim - num_test_genes, self._k_foreground]),
                name="w",
            )

            ty = yield tfd.Gamma(
                concentration=tf.ones([self._k_foreground, num_datapoints_y]),
                rate=tf.math.multiply(
                    1 * tf.ones([self._k_foreground, num_datapoints_y]), 1
                ),
                name="ty",
            )

            x = yield tfd.Poisson(
                rate=tf.math.multiply(
                    tf.math.multiply(tf.math.multiply(tf.matmul(s, zx), 1), deltax),
                    size_factor_x,
                ),
                name="x",
            )

            if num_test_genes != 0:
                w_padding = tf.zeros([num_test_genes, self._k_foreground])
                w = tf.concat([w, w_padding], axis=0)

            y = yield tfd.Poisson(
                rate=tf.math.multiply(
                    tf.math.multiply(tf.matmul(s, zy) + tf.matmul(w, ty), 1),
                    size_factor_y,
                ),
                name="y",
            )

    def _fit_model_vi(
        self,
        X,
        Y,
        approximate_model,
        is_H0=False,
        num_test_genes=0,
        offset_term=True,
    ):

        assert X.shape[0] == Y.shape[0]
        data_dim = X.shape[0]
        num_datapoints_x, num_datapoints_y = X.shape[1], Y.shape[1]

        if self.compute_size_factors:
            # counts_per_cell_X = np.mean(X, axis=0)
            # counts_per_cell_X = np.expand_dims(counts_per_cell_X, 0)
            # counts_per_cell_Y = np.mean(Y, axis=0)
            # counts_per_cell_Y = np.expand_dims(counts_per_cell_Y, 0)
            counts_per_cell_X = 1.0
            counts_per_cell_Y = 1.0
        else:
            counts_per_cell_X = 1.0
            counts_per_cell_Y = 1.0

        # ------- Specify model ---------

        concrete_clvm_model = functools.partial(
            self.model,
            data_dim=data_dim,
            num_datapoints_x=num_datapoints_x,
            num_datapoints_y=num_datapoints_y,
            counts_per_cell_X=counts_per_cell_X,
            counts_per_cell_Y=counts_per_cell_Y,
            is_H0=is_H0,
            num_test_genes=num_test_genes,
            offset_term=offset_term,
        )

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

        def target_log_prob_fn(*args):
            return model.log_prob(args, X, Y)

        # log_prob_inputs = ()
        # if offset_term:
        #     log_prob_inputs += (deltax)
        # if self.compute_size_factors:
        #     log_prob_inputs += (size_factor_x, size_factor_y)
        # log_prob_inputs += (s, zx, zy)
        # if not is_H0:
        #     log_prob_inputs += (w, ty)

        # if is_H0:

        #     if offset_term:

        #         def target_log_prob_fn(deltax, size_factor_x, size_factor_y, s, zx, zy):
        #             return model.log_prob(
        #                 (deltax, size_factor_x, size_factor_y, s, zx, zy, X, Y)
        #             )

        #     else:

        #         def target_log_prob_fn(size_factor_x, size_factor_y, s, zx, zy):
        #             return model.log_prob(
        #                 (size_factor_x, size_factor_y, s, zx, zy, X, Y)
        #             )

        # else:

        #     if offset_term:

        #         if self.compute_size_factors:

        #             def target_log_prob_fn(
        #                 deltax, size_factor_x, size_factor_y, s, zx, zy, w, ty
        #             ):
        #                 return model.log_prob(
        #                     (deltax, size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y)
        #                 )

        #         else:

        #             def target_log_prob_fn(
        #                 deltax, s, zx, zy, w, ty
        #             ):
        #                 return model.log_prob(
        #                     (deltax, s, zx, zy, w, ty, X, Y)
        #                 )



        #     else:

        #         if self.compute_size_factors:
        #             def target_log_prob_fn(size_factor_x, size_factor_y, s, zx, zy, w, ty):
        #                 return model.log_prob(
        #                     (size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y)
        #                 )
        #         else:
        #             def target_log_prob_fn(s, zx, zy, w, ty):
        #                 return model.log_prob(
        #                     (s, zx, zy, w, ty, X, Y)
        #                 )

        # --------- Fit variational inference model using MC samples and gradient descent ----------

        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            surrogate_posterior=approximate_model.approximate_posterior,
            optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI),
            num_steps=NUM_VI_ITERS,
        )

        return_dict = {
            "loss_trace": losses,
            "approximate_model": approximate_model
        }

        return return_dict

    def _fit_model_map(
        self,
        X,
        Y,
        compute_size_factors=True,
        is_H0=False,
        num_test_genes=0,
        offset_term=True,
    ):

        assert X.shape[0] == Y.shape[0]
        data_dim = X.shape[0]
        num_datapoints_x, num_datapoints_y = X.shape[1], Y.shape[1]

        if compute_size_factors:
            counts_per_cell_X = np.mean(X, axis=0)
            counts_per_cell_X = np.expand_dims(counts_per_cell_X, 0)
            counts_per_cell_Y = np.mean(Y, axis=0)
            counts_per_cell_Y = np.expand_dims(counts_per_cell_Y, 0)
        else:
            counts_per_cell_X = 1.0
            counts_per_cell_Y = 1.0

        # ------- Specify model ---------

        concrete_clvm_model = functools.partial(
            clvm,
            data_dim=data_dim,
            latent_dim_shared=latent_dim_shared,
            latent_dim_target=latent_dim_target,
            num_datapoints_x=num_datapoints_x,
            num_datapoints_y=num_datapoints_y,
            counts_per_cell_X=counts_per_cell_X,
            counts_per_cell_Y=counts_per_cell_Y,
            is_H0=is_H0,
            num_test_genes=num_test_genes,
            offset_term=offset_term,
        )

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

        if is_H0:

            if offset_term:

                def target_log_prob_fn(size_factor_x, size_factor_y, s, zx, zy):
                    return model.log_prob(
                        (size_factor_x, size_factor_y, s, zx, zy, X, Y)
                    )

            else:

                def target_log_prob_fn(deltax, size_factor_x, size_factor_y, s, zx, zy):
                    return model.log_prob(
                        (deltax, size_factor_x, size_factor_y, s, zx, zy, X, Y)
                    )

        else:

            if offset_term:

                def target_log_prob_fn(
                    deltax, size_factor_x, size_factor_y, s, zx, zy, w, ty
                ):
                    return model.log_prob(
                        (deltax, size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y)
                    )

            else:

                def target_log_prob_fn(size_factor_x, size_factor_y, s, zx, zy, w, ty):
                    return model.log_prob(
                        (size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y)
                    )


        mu_x = tf.Variable(tf.random.normal([data_dim, 1]))
        mu_y = tf.Variable(tf.random.normal([data_dim, 1]))
        s = tf.Variable(tf.random.normal([data_dim, self._k_shared]))
        w = tf.Variable(tf.random.normal([data_dim, self._k_foreground]))
        zx = tf.Variable(tf.random.normal([self._k_shared, num_datapoints_x]))
        zy = tf.Variable(tf.random.normal([self._k_shared, num_datapoints_y]))
        ty = tf.Variable(tf.random.normal([self._k_foreground, num_datapoints_y]))

        target_log_prob_fn = lambda mu_x, mu_y, s, zx, zy, w, ty: model.log_prob(
            (mu_x, mu_y, s, zx, zy, w, ty, X, Y)
        )
        losses = tfp.math.minimize(
            lambda: -target_log_prob_fn(mu_x, mu_y, s, zx, zy, w, ty),
            optimizer=tf.optimizers.Adam(learning_rate=0.05),
            num_steps=1000,
        )

        if is_H0:
            return_dict = {
                "loss_trace": approximate_model.losses,
                "qs_mean": approximate_model.qs_mean,
                "qzx_mean": approximate_model.qzx_mean,
                "qzy_mean": approximate_model.qzy_mean,
                "qs_stddv": approximate_model.qs_stddv,
                "qzx_stddv": approximate_model.qzx_stddv,
                "qzy_stddv": approximate_model.qzy_stddv
            }
        else:
            return_dict = {
                "loss_trace": approximate_model.losses,
                "qs_mean": approximate_model.qs_mean,
                "q2_mean": approximate_model.q2_mean,
                "qzx_mean": approximate_model.qzx_mean,
                "qzy_mean": approximate_model.qzy_mean,
                "qty_mean": approximate_model.qty_mean,
                "qs_stddv": approximate_model.qs_stddv,
                "qzx_stddv": approximate_model.qzx_stddv,
                "qzy_stddv": approximate_model.qzy_stddv,
                "qty_stddv": approximate_model.qty_stddv,
                "qw_stddv": approximate_model.qw_stddv
            }

        return return_dict