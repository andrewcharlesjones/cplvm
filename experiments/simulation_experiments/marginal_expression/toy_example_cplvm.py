import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from pcpca import PCPCA, CPCA

from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

n, m = 100, 100
p = 2


# DATA FOR NONNEGATIVE CLVM

# xs = np.random.normal(20, 5, size=n).astype(int)
# ys = np.random.poisson(4, size=n)
# X = np.vstack([xs, ys]).T

# xs = np.random.normal(20, 5, size=n // 2).astype(int)
# ys = np.random.poisson(4, size=n // 2)
# Y1 = np.vstack([xs, ys]).T

# ys = np.random.normal(20, 5, size=n // 2).astype(int)
# xs = np.random.poisson(4, size=n // 2)
# Y2 = np.vstack([xs, ys]).T
# Y = np.concatenate([Y1, Y2], axis=0)


############ Generate data ############

# Covariance of RVs
cov_mat = np.array([[2.7, 2.6], [2.6, 2.7]])

n, m = 1000, 1000
p = 2

Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m // 2)
Z_tilde = norm.cdf(Z)
Y1 = poisson.ppf(q=Z_tilde, mu=10)
Y1[:, 0] += 8

Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m // 2)
Z_tilde = norm.cdf(Z)
Y2 = poisson.ppf(q=Z_tilde, mu=10)
Y2[:, 1] += 8
Y = np.concatenate([Y1, Y2], axis=0)

# # Generate latent variables
Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=n)

# Pass through standard normal CDF
Z_tilde = norm.cdf(Z)

# Inverse of observed distribution function
X = poisson.ppf(q=Z_tilde, mu=10)
X += 4

# Pre-standardized data
X_standardized = (X - X.mean(0)) / X.std(0)
Y_standardized = (Y - Y.mean(0)) / Y.std(0)

# Labels of the foreground clusters
true_labels = np.zeros(m)
true_labels[m // 2 :] = 1


gene2_offset_list = [0, 1, 5, 20]
plt.figure(figsize=(len(gene2_offset_list) * 7, 7))

latent_dim_shared = 2
latent_dim_foreground = 2

for ii, gene2_offset in enumerate(gene2_offset_list):

    X_shifted = X.copy()
    X_shifted[:, 1] += gene2_offset

    print(np.mean(X_shifted, axis=0))
    print(np.mean(Y, axis=0))

    ############ CPLVM ############

    # Fit model

    cplvm = CPLVM(
        k_shared=latent_dim_shared,
        k_foreground=latent_dim_foreground,
        compute_size_factors=False,
    )

    approx_model = CPLVMLogNormalApprox(
        X_shifted.T,
        Y.T,
        latent_dim_shared,
        latent_dim_foreground,
        offset_term=True,
        compute_size_factors=False,
    )
    results = cplvm._fit_model_vi(
        X_shifted.T, Y.T, approx_model, is_H0=False, offset_term=True
    )

    W = np.exp(
        results["approximate_model"].qw_mean.numpy()
        + results["approximate_model"].qw_stddv.numpy() ** 2
    )
    S = np.exp(
        results["approximate_model"].qs_mean.numpy()
        + results["approximate_model"].qs_stddv.numpy() ** 2
    )

    # qdeltax_mean = results['approximate_model'].qdeltax_mean
    # qdeltax_stddv = results['approximate_model'].qdeltax_stddv
    # deltax_mean = np.exp(qdeltax_mean + 0.5 * qdeltax_stddv**2)

    plt.subplot(1, len(gene2_offset_list), ii + 1)

    # Plot
    plt.xlim([-3, 38])
    plt.ylim([-3, 38])
    plt.scatter(
        X_shifted[:, 0], X_shifted[:, 1], label="Background", color="gray", alpha=0.4
    )

    S_slope = S[1, 0] / S[0, 0]
    S_intercept = 0
    axes = plt.gca()
    xlims = np.array(axes.get_xlim())
    x_vals = np.linspace(xlims[0], xlims[1], 100)
    y_vals = S_slope * x_vals
    plt.plot(x_vals, y_vals, "--", label="S1", color="black", linewidth=3)

    W_slope = W[1, 0] / W[0, 0]
    W_intercept = 0
    plt.scatter(
        Y[: m // 2, 0],
        Y[: m // 2, 1],
        label="Foreground group 1",
        color="green",
        alpha=0.4,
    )
    plt.scatter(
        Y[m // 2 :, 0],
        Y[m // 2 :, 1],
        label="Foreground group 2",
        color="orange",
        alpha=0.4,
    )

    W_slope = W[1, 0] / W[0, 0]
    axes = plt.gca()
    xlims = np.array(axes.get_xlim())
    x_vals = np.linspace(xlims[0], xlims[1], 100)
    y_vals = W_slope * x_vals
    plt.plot(x_vals, y_vals, "--", label="W1", color="red", linewidth=3)

    W_slope = W[1, 1] / W[0, 1]
    y_vals = W_slope * x_vals
    plt.plot(x_vals, y_vals, "--", label="W2", color="red", linewidth=3)

    plt.xlabel("Gene 1")
    plt.ylabel("Gene 2")

    plt.legend(prop={"size": 20})
    # dx2 = round(deltax_mean.squeeze()[1], 2)
    # plt.title(r"$\delta={0:.2f}$".format(deltax_mean.squeeze()[1]))
    plt.tight_layout()


plt.show()
import ipdb

ipdb.set_trace()
