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


# ------- Specify model ---------


def clvm(data_dim, num_datapoints, counts_per_cell, dummy, is_H0=False):

    mu = yield tfd.Normal(
        loc=tf.zeros([data_dim, 1]), scale=tf.ones([data_dim, 1]), name="mu"
    )

    beta = yield tfd.Normal(
        loc=tf.zeros([data_dim, 1]), scale=tf.ones([data_dim, 1]), name="beta"
    )

    # sigma = yield tfd.InverseGamma(concentration=tf.ones([data_dim, 1]),
    #                                 scale=1,
    #                                 name="sigma")
    data = yield tfd.Normal(
        loc=(tf.matmul(beta, dummy) + mu) + np.log(counts_per_cell), scale=1, name="x"
    )


def fit_model(X, Y, compute_size_factors=True, is_H0=False):

    assert X.shape[0] == Y.shape[0]
    data_dim = X.shape[0]
    num_datapoints_x, num_datapoints_y = X.shape[1], Y.shape[1]
    n = num_datapoints_x + num_datapoints_y

    dummy = np.zeros(n)
    dummy[num_datapoints_x:] = 1
    dummy = np.expand_dims(dummy, 0)

    data = np.concatenate([X, Y], axis=1)

    data = np.log(data + 1)

    if compute_size_factors:
        # counts_per_cell = np.sum(data, axis=0)
        # counts_per_cell = np.expand_dims(counts_per_cell, axis=0)
        counts_per_cell = np.sum(np.concatenate([X, Y], axis=1), axis=0)
        counts_per_cell = np.expand_dims(counts_per_cell, axis=0)
        assert counts_per_cell.shape[1] == X.shape[1] + Y.shape[1]
    else:
        counts_per_cell = 1.0

    # ------- Specify model ---------

    concrete_clvm_model = functools.partial(
        clvm,
        data_dim=data_dim,
        num_datapoints=n,
        counts_per_cell=counts_per_cell,
        dummy=dummy,
        is_H0=is_H0,
    )

    model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

    if is_H0:

        def target_log_prob_fn(mu, beta):
            return model.log_prob((mu, beta, data))

    else:

        def target_log_prob_fn(mu, beta):
            return model.log_prob((mu, beta, data))

    # ------- Specify variational families -----------

    # Variational parmater means

    # mu
    qmu_mean = tf.Variable(tf.random.normal([data_dim, 1]))
    qmu_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, 1]), bijector=tfb.Softplus()
    )

    # beta
    qbeta_mean = tf.Variable(tf.random.normal([data_dim, 1]))
    qbeta_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, 1]), bijector=tfb.Softplus()
    )

    # sigma
    # qsigma_concentration = tfp.util.TransformedVariable(
    #     tf.ones([data_dim, 1]),
    #     bijector=tfb.Softplus())

    def factored_normal_variational_model():

        qmu = yield tfd.Normal(loc=qmu_mean, scale=qmu_stddv, name="qmu")

        qbeta = yield tfd.Normal(loc=qbeta_mean, scale=qbeta_stddv, name="qbeta")

        # qsigma = yield tfd.InverseGamma(concentration=qsigma_concentration,
        #                 scale=1,
        #              name="qsigma")

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

    # d = np.log(data + 1)
    # d = data / data.sum(0)
    # from sklearn.linear_model import LinearRegression
    # plt.scatter(np.squeeze(LinearRegression().fit(dummy.T, d.T).coef_), np.squeeze(qbeta_mean.numpy()))
    # plt.show()

    # d = (d.T - d.mean(1)).T
    # x = np.mean(d[:, num_datapoints_x:], axis=1)
    # y = np.mean(d[:, :num_datapoints_x], axis=1)
    # from sklearn.linear_model import LinearRegression
    # import ipdb
    # ipdb.set_trace()

    # plt.scatter(x - y, np.squeeze(qbeta_mean.numpy()))
    # plt.show()
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
            # 'qdelta_mean': qdelta_mean,
            # 'qdelta_stddv': qdelta_stddv,
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
            # 'qdelta_mean': qdelta_mean,
            # 'qdelta_stddv': qdelta_stddv,
        }

    return return_dict
