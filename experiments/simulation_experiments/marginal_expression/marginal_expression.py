import matplotlib
from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox

import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import poisson
from scipy.special import logsumexp


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    num_datapoints_x = 100
    num_datapoints_y = 100
    data_dim = 2
    latent_dim_shared = 2
    latent_dim_foreground = 2

    NUM_REPEATS = 5

    xrate_list = [0.5, 1, 2, 5, 10]
    delta_list_gene1 = np.empty((NUM_REPEATS, len(xrate_list)))
    delta_list_gene2 = np.empty((NUM_REPEATS, len(xrate_list)))

    for ii, xrate in enumerate(xrate_list):

        for jj in range(NUM_REPEATS):

            X = np.random.poisson(xrate, size=(data_dim, num_datapoints_x))
            Y = np.random.poisson(1, size=(data_dim, num_datapoints_x))

            cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground, compute_size_factors=False)

            approx_model = CPLVMLogNormalApprox(
                X, Y, latent_dim_shared, latent_dim_foreground, offset_term=True, compute_size_factors=False
            )
            results = cplvm._fit_model_vi(
                X, Y, approx_model, is_H0=False, offset_term=True
            )

            qdeltax_mean = results['approximate_model'].qdeltax_mean
            qdeltax_stddv = results['approximate_model'].qdeltax_stddv
            deltax_mean = np.exp(qdeltax_mean + 0.5 * qdeltax_stddv**2)

            delta_list_gene1[jj, ii] = deltax_mean.squeeze()[0]
            delta_list_gene2[jj, ii] = deltax_mean.squeeze()[1]

            print("X mean: ", np.mean(X, axis=1))
            print("Y mean: ", np.mean(Y, axis=1))
            print("delta: ", deltax_mean.squeeze())

    delta_gene1_df = pd.DataFrame(delta_list_gene1, columns=xrate_list)
    delta_gene2_df = pd.DataFrame(delta_list_gene2, columns=xrate_list)
    delta_gene1_df_melted = pd.melt(delta_gene1_df)
    delta_gene2_df_melted = pd.melt(delta_gene2_df)

    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    sns.lineplot(data=delta_gene1_df_melted, x="variable", y="value", err_style="bars", color="black")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\delta$")
    plt.title("Gene 1")
    plt.subplot(122)
    sns.lineplot(data=delta_gene2_df_melted, x="variable", y="value", err_style="bars", color="black")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\delta$")
    plt.title("Gene 2")
    plt.tight_layout()
    plt.savefig("../out/marginal_expression_test.png")
    plt.show()
    import ipdb

    ipdb.set_trace()
        
