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

warnings.filterwarnings("ignore")

NUM_VI_ITERS = 300
LEARNING_RATE_VI = 0.05
GAMMA_MULTIPLIER = 3


# ------- Specify model ---------


def clvm(
    data_dim,
    latent_dim_shared,
    latent_dim_target,
    num_datapoints_x,
    num_datapoints_y,
    counts_per_cell_X,
    counts_per_cell_Y,
    is_H0=False,
):

    mu_x = yield tfd.Normal(
        loc=tf.zeros([data_dim, 1]), scale=tf.ones([data_dim, 1]), name="mu_x"
    )

    mu_y = yield tfd.Normal(
        loc=tf.zeros([data_dim, 1]), scale=tf.ones([data_dim, 1]), name="mu_y"
    )

    s = yield tfd.Normal(
        loc=tf.zeros([data_dim, latent_dim_shared]),
        scale=tf.ones([data_dim, latent_dim_shared]),
        name="w",
    )

    zx = yield tfd.Normal(
        loc=tf.zeros([latent_dim_shared, num_datapoints_x]),
        scale=tf.ones([latent_dim_shared, num_datapoints_x]),
        name="zx",
    )

    zy = yield tfd.Normal(
        loc=tf.zeros([latent_dim_shared, num_datapoints_y]),
        scale=tf.ones([latent_dim_shared, num_datapoints_y]),
        name="zy",
    )

    # Null
    if is_H0:

        x = yield tfd.Gaussian(
            loc=tf.math.add(tf.matmul(s, zx), mu_x),
            scale=tf.ones([data_dim, num_datapoints_x]),
            name="x",
        )

        y = yield tfd.Gaussian(
            loc=tf.math.add(tf.matmul(s, zy), mu_y),
            scale=tf.ones([data_dim, num_datapoints_y]),
            name="y",
        )

    else:

        w = yield tfd.Normal(
            loc=tf.zeros([data_dim, latent_dim_target]),
            scale=tf.ones([data_dim, latent_dim_target]),
            name="w",
        )

        ty = yield tfd.Normal(
            loc=tf.zeros([latent_dim_target, num_datapoints_y]),
            scale=tf.ones([latent_dim_target, num_datapoints_y]),
            name="ty",
        )

        x = yield tfd.Normal(
            loc=tf.math.add(tf.matmul(s, zx), mu_x),
            scale=tf.ones([data_dim, num_datapoints_x]),
            name="x",
        )

        y = yield tfd.Normal(
            loc=tf.math.add(tf.matmul(s, zy) + tf.matmul(w, ty), mu_y),
            scale=tf.ones([data_dim, num_datapoints_y]),
            name="y",
        )


def fit_model(
    X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False
):

    assert X.shape[0] == Y.shape[0]
    data_dim = X.shape[0]
    num_datapoints_x, num_datapoints_y = X.shape[1], Y.shape[1]

    X = np.log(X + 1)
    Y = np.log(Y + 1)

    if compute_size_factors:
        counts_per_cell_X = np.sum(X, axis=0)
        counts_per_cell_X = np.expand_dims(counts_per_cell_X, 0)
        counts_per_cell_Y = np.sum(Y, axis=0)
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
    )

    model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

    if is_H0:

        def target_log_prob_fn(deltax, deltay, s, zx, zy):
            return model.log_prob((deltax, deltay, s, zx, zy, X, Y))

    else:

        def target_log_prob_fn(deltax, deltay, s, zx, zy, w, ty):
            return model.log_prob((deltax, deltay, s, zx, zy, w, ty, X, Y))

    # ------- Specify variational families -----------

    # Variational parmater means

    # mu
    qmu_x_mean = tf.Variable(tf.random.normal([data_dim, 1]))
    qmu_x_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, 1]), bijector=tfb.Softplus()
    )

    qmu_y_mean = tf.Variable(tf.random.normal([data_dim, 1]))
    qmu_y_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, 1]), bijector=tfb.Softplus()
    )

    # S
    qs_mean = tf.Variable(tf.random.normal([data_dim, latent_dim_shared]))
    qw_mean = tf.Variable(tf.random.normal([data_dim, latent_dim_target]))
    qzx_mean = tf.Variable(tf.random.normal([latent_dim_shared, num_datapoints_x]))
    qzy_mean = tf.Variable(tf.random.normal([latent_dim_shared, num_datapoints_y]))
    qty_mean = tf.Variable(tf.random.normal([latent_dim_target, num_datapoints_y]))

    qs_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, latent_dim_shared]), bijector=tfb.Softplus()
    )
    qw_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, latent_dim_target]), bijector=tfb.Softplus()
    )
    qzx_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([latent_dim_shared, num_datapoints_x]), bijector=tfb.Softplus()
    )
    qzy_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([latent_dim_shared, num_datapoints_y]), bijector=tfb.Softplus()
    )
    qty_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([latent_dim_target, num_datapoints_y]), bijector=tfb.Softplus()
    )

    def factored_normal_variational_model():
        qmu_x = yield tfd.LogNormal(loc=qmu_x_mean, scale=qmu_x_stddv, name="qmu_x")
        qmu_y = yield tfd.LogNormal(loc=qmu_y_mean, scale=qmu_y_stddv, name="qmu_y")
        qs = yield tfd.LogNormal(loc=qs_mean, scale=qs_stddv, name="qs")
        qzx = yield tfd.LogNormal(loc=qzx_mean, scale=qzx_stddv, name="qzx")
        qzy = yield tfd.LogNormal(loc=qzy_mean, scale=qzy_stddv, name="qzy")

        if not is_H0:
            qw = yield tfd.LogNormal(loc=qw_mean, scale=qw_stddv, name="qw")
            qty = yield tfd.LogNormal(loc=qty_mean, scale=qty_stddv, name="qty")

    # Surrogate posterior that we will try to make close to p
    surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
        factored_normal_variational_model
    )

    # --------- Fit variational inference model using MC samples and gradient descent ----------

    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI),
        num_steps=NUM_VI_ITERS,
    )

    # import ipdb
    # ipdb.set_trace()

    if is_H0:
        return_dict = {
            "loss_trace": losses,
            # 'qs_mean': qs_mean,
            # 'qzx_mean': qzx_mean,
            # 'qzy_mean': qzy_mean,
            # 'qs_stddv': qs_stddv,
            # 'qzx_stddv': qzx_stddv,
            # 'qzy_stddv': qzy_stddv,
            # 'qdeltax_mean': qdeltax_mean,
            # 'qdeltay_mean': qdeltay_mean,
            # 'qdeltax_stddv': qdeltax_stddv,
            # 'qdeltay_stddv': qdeltay_stddv,
        }
    else:
        return_dict = {
            "loss_trace": losses,
            # 'qs_mean': qs_mean,
            # 'qw_mean': qw_mean,
            # 'qzx_mean': qzx_mean,
            # 'qzy_mean': qzy_mean,
            # 'qty_mean': qty_mean,
            # 'qs_stddv': qs_stddv,
            # 'qw_stddv': qw_stddv,
            # 'qzx_stddv': qzx_stddv,
            # 'qzy_stddv': qzy_stddv,
            # 'qty_stddv': qty_stddv,
            # 'qdeltax_mean': qdeltax_mean,
            # 'qdeltay_mean': qdeltay_mean,
            # 'qdeltax_stddv': qdeltax_stddv,
            # 'qdeltay_stddv': qdeltay_stddv,
            # 'qsize_factors_x_mean': qsize_factors_x_mean,
            # 'qsize_factors_y_mean': qsize_factors_y_mean
        }

    return return_dict
