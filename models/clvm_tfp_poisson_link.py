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


# ------- Specify model ---------

def clvm(data_dim, latent_dim_shared, latent_dim_target, num_datapoints_x, num_datapoints_y, counts_per_cell_X, counts_per_cell_Y, is_H0=False):

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
    # size_factor_x = yield tfd.Normal(loc=np.mean(counts_per_cell_X) * tf.ones([1, num_datapoints_x]),
    #                                     scale=np.std(counts_per_cell_X) * tf.ones([1, num_datapoints_x]),
    #                                     name="size_factor_x")

    # size_factor_y = yield tfd.Normal(loc=np.mean(counts_per_cell_Y) * tf.ones([1, num_datapoints_y]),
    #                                     scale=np.std(counts_per_cell_Y) * tf.ones([1, num_datapoints_y]),
    #                                     name="size_factor_y")

    s = yield tfd.Normal(loc=tf.zeros([data_dim, latent_dim_shared]),
                 scale=tf.ones([data_dim, latent_dim_shared]),
                 name="w")

    zx = yield tfd.Normal(loc=tf.zeros([latent_dim_shared, num_datapoints_x]),
                   scale=tf.ones([latent_dim_shared, num_datapoints_x]) * 0.01,
                   name="zx")

    zy = yield tfd.Normal(loc=tf.zeros([latent_dim_shared, num_datapoints_y]),
                   scale=tf.ones([latent_dim_shared, num_datapoints_y]) * 0.01,
                   name="zy")

    # Null
    if is_H0:

        x = yield tfd.Poisson(rate=tf.math.multiply(tf.math.exp(tf.math.add(tf.math.multiply(tf.matmul(s, zx), 1), mu_x) + np.log(data_dim)), size_factor_x / data_dim),
                         name="x")

        y = yield tfd.Poisson(rate=tf.math.multiply(tf.math.exp(tf.math.add(tf.matmul(s, zy), mu_y) + np.log(data_dim)), size_factor_y / data_dim),
                             name="y")


    else:

        w = yield tfd.Normal(loc=tf.zeros([data_dim, latent_dim_target]),
                   scale=tf.ones([data_dim, latent_dim_target]),
                   name="w")

        ty = yield tfd.Normal(loc=tf.zeros([latent_dim_target, num_datapoints_y]),
                   scale=tf.ones([latent_dim_target, num_datapoints_y]) * 0.01,
                   name="ty")

        # x = yield tfd.Poisson(rate=tf.math.multiply(tf.math.exp(tf.math.add(tf.math.multiply(tf.matmul(s, zx), 1), mu_x)), size_factor_x), # / data_dim),
        #                  name="x")

        # y = yield tfd.Poisson(rate=tf.math.multiply(tf.math.exp(tf.math.add(tf.matmul(s, zy) + tf.matmul(w, ty), mu_y)), size_factor_y), # / data_dim),
        #                      name="y")
        x = yield tfd.Poisson(rate=tf.math.exp(tf.matmul(s, zx) + mu_x + tf.math.log(size_factor_x)),
                         name="x")

        y = yield tfd.Poisson(rate=tf.math.exp(tf.matmul(s, zy) + tf.matmul(w, ty) + mu_y + tf.math.log(size_factor_y)),
                             name="y")


def fit_model(X, Y, latent_dim_shared, latent_dim_target, compute_size_factors = False, is_H0 = False):

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
                                            is_H0 = is_H0)

    model=tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)
    # import ipdb; ipdb.set_trace()

    if is_H0:

        def target_log_prob_fn(mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy): return model.log_prob(
            (mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, X, Y))

    else:

        def target_log_prob_fn(mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, w, ty): return model.log_prob(
            (mu_x, mu_y, size_factor_x, size_factor_y, s, zx, zy, w, ty, X, Y))
    # ------- Specify variational families -----------

    # Variational parmater means

    # delta
    qmu_x_mean = tf.Variable(tf.random.normal([data_dim, 1]))
    qmu_x_stddv = tfp.util.TransformedVariable(
      1e-4 * tf.ones([data_dim, 1]),
      bijector=tfb.Softplus())

    qmu_y_mean = tf.Variable(tf.random.normal([data_dim, 1]))
    qmu_y_stddv = tfp.util.TransformedVariable(
      1e-4 * tf.ones([data_dim, 1]),
      bijector=tfb.Softplus())

    qsize_factor_x_mean = tf.Variable(tf.random.normal([1, num_datapoints_x]))
    qsize_factor_x_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([1, num_datapoints_x]),
        bijector=tfb.Softplus())

    qsize_factor_y_mean = tf.Variable(tf.random.normal([1, num_datapoints_y]))
    qsize_factor_y_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([1, num_datapoints_y]),
        bijector=tfb.Softplus())

    # S
    qs_mean = tf.Variable(tf.random.normal([data_dim, latent_dim_shared]))
    qw_mean = tf.Variable(tf.random.normal([data_dim, latent_dim_target]))
    qzx_mean = tf.Variable(tf.random.normal([latent_dim_shared, num_datapoints_x]))
    qzy_mean = tf.Variable(tf.random.normal([latent_dim_shared, num_datapoints_y]))
    qty_mean = tf.Variable(tf.random.normal([latent_dim_target, num_datapoints_y]))

    qs_stddv = tfp.util.TransformedVariable(
      1e-4 * tf.ones([data_dim, latent_dim_shared]),
      bijector=tfb.Softplus())
    qw_stddv = tfp.util.TransformedVariable(
      1e-4 * tf.ones([data_dim, latent_dim_target]),
      bijector=tfb.Softplus())
    qzx_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([latent_dim_shared, num_datapoints_x]),
        bijector=tfb.Softplus())
    qzy_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([latent_dim_shared, num_datapoints_y]),
        bijector=tfb.Softplus())
    qty_stddv = tfp.util.TransformedVariable(
        1e-4 * tf.ones([latent_dim_target, num_datapoints_y]),
        bijector=tfb.Softplus())

    def factored_normal_variational_model():
      qmu_x = yield tfd.Normal(loc=qmu_x_mean, scale=qmu_x_stddv, name="qmu_x")
      qmu_y = yield tfd.Normal(loc=qmu_y_mean, scale=qmu_y_stddv, name="qmu_y")
      qsize_factor_x = yield tfd.LogNormal(loc=qsize_factor_x_mean,
                     scale=qsize_factor_x_stddv,
                     name="qsize_factor_x")

      qsize_factor_y = yield tfd.LogNormal(loc=qsize_factor_y_mean,
                     scale=qsize_factor_y_stddv,
                     name="qsize_factor_y")
      qs = yield tfd.Normal(loc=qs_mean, scale=qs_stddv, name="qs")
      qzx = yield tfd.Normal(loc=qzx_mean, scale=qzx_stddv, name="qzx")
      qzy = yield tfd.Normal(loc=qzy_mean, scale=qzy_stddv, name="qzy")

      if not is_H0:
        qw = yield tfd.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
        qty = yield tfd.Normal(loc=qty_mean, scale=qty_stddv, name="qty")

    # Surrogate posterior that we will try to make close to p
    surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
        factored_normal_variational_model)

    # --------- Fit variational inference model using MC samples and gradient descent ----------

    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE_VI),
        num_steps=NUM_VI_ITERS)

    # import ipdb
    # ipdb.set_trace()

    if is_H0:
        return_dict = {
            'loss_trace': losses,
            'qs_mean': qs_mean,
            'qzx_mean': qzx_mean,
            'qzy_mean': qzy_mean,
            'qs_stddv': qs_stddv,
            'qzx_stddv': qzx_stddv,
            'qzy_stddv': qzy_stddv,
            # 'qdeltax_mean': qdeltax_mean,
            # 'qdeltay_mean': qdeltay_mean,
            # 'qdeltax_stddv': qdeltax_stddv,
            # 'qdeltay_stddv': qdeltay_stddv,
        }
    else:
        return_dict = {
            'loss_trace': losses,
            'qs_mean': qs_mean,
            'qw_mean': qw_mean,
            'qmu_x_mean': qmu_x_mean,
            'qmu_y_mean': qmu_y_mean,
            'qzx_mean': qzx_mean,
            'qzy_mean': qzy_mean,
            'qty_mean': qty_mean,
            'qs_stddv': qs_stddv,
            'qw_stddv': qw_stddv,
            'qzx_stddv': qzx_stddv,
            'qzy_stddv': qzy_stddv,
            'qty_stddv': qty_stddv,
            # 'qdeltax_mean': qdeltax_mean,
            # 'qdeltay_mean': qdeltay_mean,
            # 'qdeltax_stddv': qdeltax_stddv,
            # 'qdeltay_stddv': qdeltay_stddv,
            # 'qsize_factors_x_mean': qsize_factors_x_mean,
            # 'qsize_factors_y_mean': qsize_factors_y_mean
        }

    return return_dict


if __name__ == "__main__":


    num_datapoints_x = 5000
    num_datapoints_y = 5000
    data_dim = 200
    latent_dim_shared = 10
    latent_dim_target = 10

    zx = np.random.normal(scale=0.4, size=(latent_dim_shared, num_datapoints_x))
    zy = np.random.normal(scale=0.4, size=(latent_dim_shared, num_datapoints_y))
    ty = np.random.normal(scale=0.4, size=(latent_dim_target, num_datapoints_y))

    S = np.random.normal(scale=0.4, size=(data_dim, latent_dim_shared))
    W = np.random.normal(scale=0.4, size=(data_dim, latent_dim_target))

    X = np.random.poisson(np.exp(S @ zx))
    Y = np.random.poisson(np.exp(S @ zy + W @ ty))

    sns.heatmap(X)
    plt.show()

    import ipdb; ipdb.set_trace()



    concrete_clvm_model = functools.partial(clvm,
                                            data_dim=data_dim,
                                            latent_dim_shared=latent_dim_shared,
                                            latent_dim_target=latent_dim_target,
                                            num_datapoints_x=num_datapoints_x,
                                            num_datapoints_y=num_datapoints_y,
                                            counts_per_cell_X=1,
                                            counts_per_cell_Y=1,
                                            is_H0=False)

    model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

    samples = model.sample()
    X_sampled, Y_sampled = samples[-2], samples[-1]

    X, Y = X_sampled.numpy(), Y_sampled.numpy()
    import ipdb; ipdb.set_trace()

