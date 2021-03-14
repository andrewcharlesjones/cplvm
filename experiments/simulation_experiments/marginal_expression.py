import ipdb
import sys

sys.path.append("../../models")
from clvm_tfp_poisson_link import fit_model as fit_clvm_link
from clvm_tfp_poisson import fit_model as fit_clvm_nonnegative
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from pcpca import PCPCA, CPCA
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import adjusted_rand_score, silhouette_score

# import matplotlib
# font = {'size': 30}
# matplotlib.rc('font', **font)
# matplotlib.rcParams['text.usetex'] = True


METHODS = ["PCA", "NMF", "CPCA", "PCPCA", "CGLVM", "CPLVM"]

############ Generate data ############

# Covariance of RVs
cov_mat = np.array([[2.7, 2.6], [2.6, 2.7]])

n, m = 1000, 1000
p = 2


for _ in range(10):

    Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m // 2)
    # Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m)
    Z_tilde = norm.cdf(Z)
    Y1 = poisson.ppf(q=Z_tilde, mu=10)
    Y1[:, 0] += 8

    Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m // 2)
    Z_tilde = norm.cdf(Z)
    Y2 = poisson.ppf(q=Z_tilde, mu=10)
    Y2[:, 1] += 8
    Y = np.concatenate([Y1, Y2], axis=0)
    # Y = Y1
    # Y[:, 1] += 10

    # # Generate latent variables
    Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=n)

    # Pass through standard normal CDF
    Z_tilde = norm.cdf(Z)

    # Inverse of observed distribution function
    X = poisson.ppf(q=Z_tilde, mu=10)
    X += 4
    X[:, 1] += 25

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.scatter(Y[:, 0], Y[:, 1])
    # plt.show()
    # import sys
    # sys.exit()

    ############ CPLVM ############

    # Fit model
    model_dict = fit_clvm_nonnegative(
        X.T, Y.T, 1, 1, compute_size_factors=False, is_H0=False, offset_term=True
    )

    W = np.exp(model_dict["qw_mean"].numpy() + model_dict["qw_stddv"].numpy() ** 2)
    S = np.exp(model_dict["qs_mean"].numpy() + model_dict["qs_stddv"].numpy() ** 2)

    zx = np.exp(model_dict["qzx_mean"].numpy() + model_dict["qzx_stddv"].numpy() ** 2)
    zy = np.exp(model_dict["qzy_mean"].numpy() + model_dict["qzy_stddv"].numpy() ** 2)
    ty = np.exp(model_dict["qty_mean"].numpy() + model_dict["qty_stddv"].numpy() ** 2)

    sf_x = np.exp(
        model_dict["qsize_factors_x_mean"].numpy()
        + model_dict["qsize_factor_x_stddv"].numpy() ** 2
    )
    sf_y = np.exp(
        model_dict["qsize_factors_y_mean"].numpy()
        + model_dict["qsize_factor_y_stddv"].numpy() ** 2
    )

    deltax = np.exp(
        model_dict["qdeltax_mean"].numpy() + model_dict["qdeltax_stddv"].numpy() ** 2
    )
    print(deltax)

    X_recons = S @ zx
    Y_recons = np.multiply(S @ zy + W @ ty, deltax)
    Y_recons_partial = S @ zy

    plt.scatter(X_recons[0, :], X_recons[1, :], label="Xr")
    plt.scatter(Y_recons[0, :], Y_recons[1, :], label="Yr")
    plt.scatter(Y_recons_partial[0, :], Y_recons_partial[1, :], label="Yrp")

    plt.scatter(X[:, 0], X[:, 1], label="X")
    plt.scatter(Y[:, 0], Y[:, 1], label="Y")
    plt.legend()
    plt.show()

    ipdb.set_trace()
