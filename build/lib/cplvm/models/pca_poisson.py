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

NUM_VI_ITERS = 1000
LEARNING_RATE_VI = 0.05


# ------- Specify model ---------


def pca(data_dim, latent_dim, num_datapoints, counts_per_cell):

    s = yield tfd.Gamma(
        concentration=tf.ones([data_dim, latent_dim]),
        rate=tf.ones([data_dim, latent_dim]),
        name="w",
    )

    zx = yield tfd.Gamma(
        concentration=tf.ones([latent_dim, num_datapoints]),
        rate=tf.math.multiply(
            tf.ones([latent_dim, num_datapoints]), 1 / counts_per_cell
        ),
        name="zx",
    )

    x = yield tfd.Poisson(rate=tf.matmul(s, zx), name="x")


def fit_model(X, latent_dim, compute_size_factors=True):

    data_dim = X.shape[0]
    num_datapoints = X.shape[1]

    if compute_size_factors:
        counts_per_cell = np.mean(X, axis=0)
        counts_per_cell = np.expand_dims(counts_per_cell, 0)
    else:
        counts_per_cell = 1.0

    # ------- Specify model ---------

    concrete_clvm_model = functools.partial(
        pca,
        data_dim=data_dim,
        latent_dim=latent_dim,
        num_datapoints=num_datapoints,
        counts_per_cell=counts_per_cell,
    )

    model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

    def target_log_prob_fn(s, zx):
        return model.log_prob((s, zx, X))

    # ------- Specify variational families -----------

    # Variational parmater means

    counts_per_cell = np.mean(X, axis=0)
    counts_per_cell = np.expand_dims(counts_per_cell, 0)

    # S
    qs_mean = tf.Variable(tf.random.normal([data_dim, latent_dim]))

    zx_mean = np.repeat(np.log(counts_per_cell), latent_dim, axis=0)
    qzx_mean = tf.Variable(tf.random.normal(mean=0, shape=[latent_dim, num_datapoints]))

    # import ipdb; ipdb.set_trace()

    qs_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, latent_dim]), bijector=tfb.Softplus()
    )
    qzx_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([latent_dim, num_datapoints]), bijector=tfb.Softplus()
    )

    def factored_normal_variational_model():
        qs = yield tfd.LogNormal(loc=qs_mean, scale=qs_stddv, name="qs")
        qzx = yield tfd.LogNormal(loc=qzx_mean, scale=qzx_stddv, name="qzx")

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

    return_dict = {
        "loss_trace": losses,
        "qs_mean": qs_mean,
        "qzx_mean": qzx_mean,
        # 'qzy_mean': qzy_mean,
        "qs_stddv": qs_stddv,
        "qzx_stddv": qzx_stddv,
        # 'qzy_stddv': qzy_stddv,
        # 'qdeltax_mean': qdeltax_mean,
        # 'qdeltay_mean': qdeltay_mean,
        # 'qdeltax_stddv': qdeltax_stddv,
        # 'qdeltay_stddv': qdeltay_stddv,
    }

    return return_dict
