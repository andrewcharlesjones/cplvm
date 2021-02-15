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

NUM_VI_ITERS = 500
LEARNING_RATE_VI = 0.05


# ------- Specify model ---------

def clvm(data_dim, num_datapoints, counts_per_cell, dummy, is_H0=False):

    mu = yield tfd.Normal(loc=tf.zeros([data_dim, 1]),
                 scale=tf.ones([data_dim, 1]),
                 name="mu")

    beta = yield tfd.Normal(loc=tf.zeros([data_dim, 1]),
                                        scale=tf.ones([data_dim, 1]),
                                        name="beta")

    # size_factor = yield tfd.LogNormal(loc=np.mean(np.log(counts_per_cell)) * tf.ones([1, num_datapoints]),
    #                                     scale=np.std(np.log(counts_per_cell)) * tf.ones([1, num_datapoints]),
    #                                     name="size_factor")

    data = yield tfd.Poisson(rate=tf.math.exp(tf.matmul(beta, dummy) + mu), # + tf.math.log(size_factor)),
                                          name="x")



def fit_model(X, Y, compute_size_factors=True, is_H0=False, sf_x=None, sf_y=None):

    assert X.shape[0] == Y.shape[0]
    data_dim = X.shape[0]
    num_datapoints_x, num_datapoints_y = X.shape[1], Y.shape[1]
    n = num_datapoints_x + num_datapoints_y

    dummy = np.zeros(n)
    dummy[num_datapoints_x:] = 1
    dummy = np.expand_dims(dummy, 0)

    data = np.concatenate([X, Y], axis=1)

    if (sf_x is not None) and (sf_y is not None):
        counts_per_cell_X = sf_x
        counts_per_cell_Y = sf_y
        counts_per_cell_X = np.expand_dims(counts_per_cell_X, 0)
        counts_per_cell_Y = np.expand_dims(counts_per_cell_Y, 0)
        counts_per_cell = np.concatenate([counts_per_cell_X, counts_per_cell_Y], axis=1)
        assert counts_per_cell.shape[1] == X.shape[1] + Y.shape[1]
    else:
        if compute_size_factors:
            counts_per_cell_X = np.sum(X, axis=0)
            counts_per_cell_X = np.expand_dims(counts_per_cell_X, 0)
            counts_per_cell_Y = np.sum(Y, axis=0)
            counts_per_cell_Y = np.expand_dims(counts_per_cell_Y, 0)
            counts_per_cell = np.concatenate([counts_per_cell_X, counts_per_cell_Y], axis=1)
            assert counts_per_cell.shape[1] == X.shape[1] + Y.shape[1]
        else:
            counts_per_cell = 1.0


    

    # ------- Specify model ---------


    concrete_clvm_model = functools.partial(clvm,
                                            data_dim=data_dim,
                                            num_datapoints=n,
                                            counts_per_cell=counts_per_cell,
                                            dummy=dummy,
                                            is_H0=is_H0)

    model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

    

    if is_H0:

        def target_log_prob_fn(mu, beta, size_factor): return model.log_prob(
            (mu, beta, size_factor, data))

    else:

        # def target_log_prob_fn(mu, beta, size_factor): return model.log_prob(
        #     (mu, beta, size_factor, data))
        def target_log_prob_fn(mu, beta): return model.log_prob(
            (mu, beta, data))
        
    # ------- Specify variational families -----------

    # mu = tf.Variable(tf.random.normal([data_dim, 1]))
    # beta = tf.Variable(tf.random.normal([data_dim, 1]))
    # size_factor = tfp.util.TransformedVariable(tf.math.exp(tf.random.normal([1, n])), bijector=tfb.Softplus())

    # losses = tfp.math.minimize(
    #     lambda: -target_log_prob_fn(mu, beta, size_factor),
    #     optimizer=tf.optimizers.Adam(learning_rate=0.05),
    #     num_steps=500)

    # d = np.log(data + 1)
    # d = d / d.sum(0)
    # from sklearn.linear_model import LinearRegression
    # plt.scatter(np.squeeze(LinearRegression().fit(dummy.T, d.T).coef_), np.squeeze(beta.numpy()))
    # plt.show()
    # import ipdb
    # ipdb.set_trace()


    # Variational parmater means

    # mu
    qmu_mean = tf.Variable(tf.random.normal([data_dim, 1]))
    qmu_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, 1]),
        bijector=tfb.Softplus())

    # beta
    qbeta_mean = tf.Variable(tf.random.normal([data_dim, 1]))
    qbeta_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([data_dim, 1]),
        bijector=tfb.Softplus())

    # size factor
    # qsize_factor_mean = tf.Variable(np.mean(np.log(counts_per_cell)) * tf.ones([1, n]))
    # qsize_factor_stddv = tfp.util.TransformedVariable(
    #     np.std(np.log(counts_per_cell)) * tf.ones([1, n]), # + tf.math.exp(tf.random.normal([1, n])),
    #     # 1e-4 * tf.ones([1, n]),
    #     bijector=tfb.Softplus())
    qsize_factor_mean = tf.Variable(tf.random.normal([1, n]))
    qsize_factor_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([1, n]),
        bijector=tfb.Softplus())

    def factored_normal_variational_model():

        qmu = yield tfd.Normal(loc=qmu_mean,
                 scale=qmu_stddv,
                 name="qmu")

        qbeta = yield tfd.Normal(loc=qbeta_mean,
                     scale=qbeta_stddv,
                     name="qbeta")

        # qsize_factor = yield tfd.LogNormal(loc=qsize_factor_mean,
        #              scale=qsize_factor_stddv,
        #              name="qsize_factor")

    # Surrogate posterior that we will try to make close to p
    surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
        factored_normal_variational_model)

    # --------- Fit variational inference model using MC samples and gradient descent ----------

    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI),
        num_steps=NUM_VI_ITERS)
    # ipdb.set_trace()

    if is_H0:
        return_dict = {
            'loss_trace': losses,
        }
    else:
        return_dict = {
            'loss_trace': losses,
            'qmu_mean': qmu_mean,
            'qmu_stddv': qmu_stddv,
            'qbeta_mean': qbeta_mean,
            'qbeta_stddv': qbeta_stddv,
            'qsize_factor_mean': qsize_factor_mean,
            'qsize_factor_stddv': qsize_factor_stddv
        }

    return return_dict

if __name__ == "__main__":

    data_dim = 10
    n = 5000
    dummy = np.zeros(n)
    dummy[3000:] = 1
    dummy = np.expand_dims(dummy, 0)
    mimosca_model = functools.partial(clvm,
                                            data_dim=data_dim,
                                            dummy=dummy,
                                            num_datapoints=n,
                                            counts_per_cell=1,
                                            is_H0=False)

    model = tfd.JointDistributionCoroutineAutoBatched(mimosca_model)

    samples = model.sample()
    data_sampled = samples[-1]

    data_sampled = data_sampled.numpy()
    import ipdb; ipdb.set_trace()
