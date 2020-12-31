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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pcpca import CPCA, PCPCA

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import sys
sys.path.append("../models")

from clvm_tfp_poisson import fit_model as fit_clvm

tf.enable_v2_behavior()

warnings.filterwarnings('ignore')

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True



if __name__ == "__main__":


    N_REPEATS = 5
    sil_scores_clvm = []
    sil_scores_pca = []
    sil_scores_cpca = []

    for _ in range(N_REPEATS):

        num_datapoints_x = 1000
        num_datapoints_y = 1000
        data_dim = 100
        latent_dim_shared = 2
        latent_dim_target = 2

        a, b = 1, 1

        actual_s = np.random.gamma(a, 1/b, size=(data_dim, latent_dim_shared))
        actual_w = np.random.gamma(a, 1/b, size=(data_dim, latent_dim_target))
        # actual_w[-data_dim//frac_response:, 0] = np.random.gamma(40, 1/5, size=(data_dim//frac_response))

        actual_zx = np.random.gamma(a, 1/b, size=(latent_dim_shared, num_datapoints_x))
        actual_zy = np.random.gamma(a, 1/b, size=(latent_dim_shared, num_datapoints_y))
        actual_ty = np.random.gamma(a, 1/b, size=(latent_dim_target, num_datapoints_y))

        actual_ty[0, :num_datapoints_y//2] = np.random.gamma(1, 1/20, size=(num_datapoints_y//2))
        actual_ty[1, num_datapoints_y//2:] = np.random.gamma(1, 1/20, size=(num_datapoints_y//2))

        # actual_w[-data_dim//frac_response:, 0] = np.random.gamma(20, 1/5, size=(data_dim//frac_response))
        # actual_w[0, :] = np.random.gamma(20, 1/5, size=latent_dim_target)

        x_train = np.random.poisson(actual_s @ actual_zx)

        y_train = np.random.poisson(actual_s @ actual_zy + actual_w @ actual_ty)

        labs = np.zeros(num_datapoints_y)
        labs[num_datapoints_y//2:] = 1

        ## Run CPCA
        cpca = CPCA(n_components=2, gamma=0.8)
        cpca_reduced_Y, cpca_reduced_X = cpca.fit_transform(y_train, x_train)

        sil_score_cpca = silhouette_score(X=cpca_reduced_Y.T, labels=labs)
        sil_scores_cpca.append(sil_score_cpca)

        # plt.scatter(cpca_reduced_Y.T[:, 0], cpca_reduced_Y.T[:, 1], c=labs)
        # import ipdb; ipdb.set_trace()


        model_dict = fit_clvm(x_train, y_train, latent_dim_shared, latent_dim_target, compute_size_factors=True)


        


        zx_estimated = np.exp(model_dict['qzx_mean'].numpy() + model_dict['qzx_stddv'].numpy()**2 / 2)
        zy_estimated = np.exp(model_dict['qzy_mean'].numpy() + model_dict['qzy_stddv'].numpy()**2 / 2)
        ty_estimated = np.exp(model_dict['qty_mean'].numpy() + model_dict['qty_stddv'].numpy()**2 / 2)


        # Do the same for PCA
        data_for_pca = np.log(np.concatenate([x_train, y_train], axis=1) + 1).T
        data_for_pca -= np.mean(data_for_pca, axis=0)
        pca_reduced_data = PCA(n_components=latent_dim_target).fit_transform(data_for_pca)

        # Compute silhouette score
        sil_score_clvm = silhouette_score(X=ty_estimated.T, labels=labs)
        sil_score_pca = silhouette_score(X=pca_reduced_data[num_datapoints_x:, :], labels=labs)

        print("PCA: {}".format(round(sil_score_pca, 2)))
        print("cPCA: {}".format(round(sil_score_cpca, 2)))
        print("cLVM: {}".format(round(sil_score_clvm, 2)))

        sil_scores_clvm.append(sil_score_clvm)
        sil_scores_pca.append(sil_score_pca)

    plt.figure(figsize=(9, 7))
    sns.boxplot(np.arange(3), [sil_scores_pca, sil_scores_cpca, sil_scores_clvm])
    plt.xticks(np.arange(3), labels=["PCA", "CPCA", "CPLVM"])
    plt.ylabel("Silhouette score")
    plt.xlabel("")
    plt.title("Cluster quality")
    plt.tight_layout()
    plt.savefig("./out/scatter_subset_response_silhouette.png")
    plt.show()
    import ipdb; ipdb.set_trace()

