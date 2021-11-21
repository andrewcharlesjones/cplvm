import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import poisson
from scipy.special import logsumexp
from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import silhouette_score

from pcpca import CPCA, PCPCA

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from cplvm import CGLVM
from cplvm import CGLVMMFGaussianApprox

# import sys

# sys.path.append("../../models")

# from clvm_tfp_poisson import fit_model as fit_clvm
# from clvm_tfp_poisson_link import fit_model_map as fit_clvm_link

tf.enable_v2_behavior()

warnings.filterwarnings("ignore")

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


if __name__ == "__main__":

    N_REPEATS = 5
    sil_scores_clvm = []
    sil_scores_pca = []
    sil_scores_nmf = []
    sil_scores_cpca = []
    sil_scores_cglvm = []

    # for _ in range(N_REPEATS):

    num_datapoints_x = 1000
    num_datapoints_y = 1000
    data_dim = 100
    latent_dim_shared = 2
    latent_dim_target = 2

    a, b = 1, 1

    actual_s = np.random.gamma(a, 1 / b, size=(data_dim, latent_dim_shared))
    actual_w = np.random.gamma(a, 1 / b, size=(data_dim, latent_dim_target))
    # actual_w[-data_dim//frac_response:, 0] = np.random.gamma(40, 1/5, size=(data_dim//frac_response))

    actual_zx = np.random.gamma(a, 1 / b, size=(latent_dim_shared, num_datapoints_x))
    actual_zy = np.random.gamma(a, 1 / b, size=(latent_dim_shared, num_datapoints_y))
    actual_ty = np.random.gamma(a, 1 / b, size=(latent_dim_target, num_datapoints_y))

    actual_ty[0, : num_datapoints_y // 2] = np.random.gamma(
        1, 1 / 20, size=(num_datapoints_y // 2)
    )
    actual_ty[1, num_datapoints_y // 2 :] = np.random.gamma(
        1, 1 / 20, size=(num_datapoints_y // 2)
    )

    # actual_w[-data_dim//frac_response:, 0] = np.random.gamma(20, 1/5, size=(data_dim//frac_response))
    # actual_w[0, :] = np.random.gamma(20, 1/5, size=latent_dim_target)

    x_train = np.random.poisson(actual_s @ actual_zx)

    y_train = np.random.poisson(actual_s @ actual_zy + actual_w @ actual_ty)

    labs = np.zeros(num_datapoints_y)
    labs[num_datapoints_y // 2 :] = 1

    group1_idx = np.where(labs == 0)[0]
    group2_idx = np.where(labs == 1)[0]

    plt.figure(figsize=(14, 7))

    ######### NMF #########
    reduced_data = NMF(n_components=latent_dim_target).fit_transform(
        np.concatenate([x_train, y_train], axis=1).T
    )

    fg_reduced = reduced_data[num_datapoints_x:, :]

    plt.subplot(121)
    plt.scatter(
        fg_reduced[group1_idx, 0],
        fg_reduced[group1_idx, 1],
        color="green",
        label="Foreground group 1",
    )
    plt.scatter(
        fg_reduced[group2_idx, 0],
        fg_reduced[group1_idx, 1],
        color="orange",
        label="Foreground group 2",
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title("NMF")
    plt.legend(fontsize=20)

    ######### CGLVM #########

    cglvm = CGLVM(k_shared=2, k_foreground=2, compute_size_factors=False)
    approx_model = CGLVMMFGaussianApprox(
        X=x_train,
        Y=y_train,
        k_shared=2,
        k_foreground=2,
        num_test_genes=0,
        is_H0=False,
        compute_size_factors=False,
    )
    results = cglvm.fit_model_vi(x_train, y_train, approx_model, is_H0=False)
    fg_reduced = results["approx_model"].qty_mean.numpy().T

    plt.subplot(122)
    plt.scatter(
        fg_reduced[group1_idx, 0],
        fg_reduced[group1_idx, 1],
        color="green",
        label="Foreground group 1",
    )
    plt.scatter(
        fg_reduced[group2_idx, 0],
        fg_reduced[group1_idx, 1],
        color="orange",
        label="Foreground group 2",
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title("CGLVM")
    plt.legend(fontsize=20)

    plt.tight_layout()
    plt.savefig("../out/simulation_scatter_nmf_cglvm.png")
    plt.show()
    plt.close()

    import ipdb

    ipdb.set_trace()
