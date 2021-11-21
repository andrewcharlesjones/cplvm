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


tf.enable_v2_behavior()

warnings.filterwarnings("ignore")

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


if __name__ == "__main__":

    N_REPEATS = 10

    gamma_range = np.linspace(0, 0.99, 10)
    sil_scores_cpca = np.empty((N_REPEATS, len(gamma_range)))

    for jj, gamma in enumerate(gamma_range):

        for ii in range(N_REPEATS):

            num_datapoints_x = 1000
            num_datapoints_y = 1000
            data_dim = 100
            latent_dim_shared = 2
            latent_dim_target = 2

            a, b = 1, 1

            actual_s = np.random.gamma(a, 1 / b, size=(data_dim, latent_dim_shared))
            actual_w = np.random.gamma(a, 1 / b, size=(data_dim, latent_dim_target))
            # actual_w[-data_dim//frac_response:, 0] = np.random.gamma(40, 1/5, size=(data_dim//frac_response))

            actual_zx = np.random.gamma(
                a, 1 / b, size=(latent_dim_shared, num_datapoints_x)
            )
            actual_zy = np.random.gamma(
                a, 1 / b, size=(latent_dim_shared, num_datapoints_y)
            )
            actual_ty = np.random.gamma(
                a, 1 / b, size=(latent_dim_target, num_datapoints_y)
            )

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

            ######### CPCA #########
            ## Run CPCA
            cpca = CPCA(n_components=2, gamma=0.8)
            cpca_reduced_Y, cpca_reduced_X = cpca.fit_transform(y_train, x_train)

            sil_score_cpca = silhouette_score(X=cpca_reduced_Y.T, labels=labs)
            sil_scores_cpca[ii, jj] = sil_score_cpca

    results_df = pd.DataFrame(sil_scores_cpca, columns=gamma_range)
    results_df_melted = pd.melt(results_df)
    sns.lineplot(data=results_df_melted, x="variable", y="value")
    plt.tight_layout()
    plt.show()

    import ipdb

    ipdb.set_trace()
