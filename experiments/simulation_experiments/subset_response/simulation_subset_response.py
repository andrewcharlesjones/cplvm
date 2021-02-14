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

from pcpca import PCPCA, CPCA

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import sys
sys.path.append("../../models")

from clvm_tfp_poisson import fit_model as fit_clvm

tf.enable_v2_behavior()

warnings.filterwarnings('ignore')

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True



if __name__ == "__main__":

    num_datapoints_x = 1000
    num_datapoints_y = 1000
    data_dim = 100
    latent_dim_shared = 2
    latent_dim_target = 2

    a, b = 1, 1
    frac_response = 10

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

    # plt.subplot(131)
    # plt.scatter(actual_zx[0, :], actual_zx[1, :], label="X")
    # plt.subplot(132)
    # plt.scatter(actual_zy[0, :], actual_zy[1, :], label="Y")
    # plt.subplot(133)
    # plt.scatter(actual_ty[0, :num_datapoints_y//2], actual_ty[1, :num_datapoints_y//2], label="ty1")
    # plt.scatter(actual_ty[0, num_datapoints_y//2:], actual_ty[1, num_datapoints_y//2:], label="ty2")
    # plt.legend()
    # plt.show()


    model_dict = fit_clvm(x_train, y_train, latent_dim_shared, latent_dim_target, compute_size_factors=True)


    labs = np.zeros(num_datapoints_y)
    labs[-num_datapoints_y//frac_response:] = 1


    zx_estimated = np.exp(model_dict['qzx_mean'].numpy() + model_dict['qzx_stddv'].numpy()**2 / 2)
    zy_estimated = np.exp(model_dict['qzy_mean'].numpy() + model_dict['qzy_stddv'].numpy()**2 / 2)
    ty_estimated = np.exp(model_dict['qty_mean'].numpy() + model_dict['qty_stddv'].numpy()**2 / 2)


    ## Run CPCA
    cpca = CPCA(n_components=2, gamma=2)
    cpca_reduced_Y, cpca_reduced_X = cpca.fit_transform(y_train, x_train)

    # plt.scatter(cpca_reduced_Y.T[:, 0], cpca_reduced_Y.T[:, 1], c=labs)
    # import ipdb; ipdb.set_trace()
    
    # Do the same for PCA
    data_for_pca = np.log(np.concatenate([x_train, y_train], axis=1) + 1).T
    data_for_pca -= np.mean(data_for_pca, axis=0)
    pca_reduced_data = PCA(n_components=latent_dim_target).fit_transform(data_for_pca)

    plt.figure(figsize=(28, 7))
    plt.subplot(141)
    plt.scatter(actual_ty[0, :num_datapoints_y//2], actual_ty[1, :num_datapoints_y//2], label="Foreground group 1", color="green")
    plt.scatter(actual_ty[0, num_datapoints_y//2:], actual_ty[1, num_datapoints_y//2:], label="Foreground group 2", color="orange")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend(prop={'size': 20})
    plt.title("Truth")

    plt.subplot(142)
    plt.scatter(pca_reduced_data[num_datapoints_x:][:num_datapoints_y//2, 0], pca_reduced_data[num_datapoints_x:][:num_datapoints_y//2, 1], label="Foreground group 1", color="green")
    plt.scatter(pca_reduced_data[num_datapoints_x:][num_datapoints_y//2:, 0], pca_reduced_data[num_datapoints_x:][num_datapoints_y//2:, 1], label="Foreground group 2", color="orange")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend(prop={'size': 20})
    plt.title("PCA")

    plt.subplot(143)
    plt.scatter(cpca_reduced_Y.T[:num_datapoints_y//2, 0], cpca_reduced_Y.T[:num_datapoints_y//2, 1], label="Foreground group 1", color="green")
    plt.scatter(cpca_reduced_Y.T[num_datapoints_y//2:, 0], cpca_reduced_Y.T[num_datapoints_y//2:, 1], label="Foreground group 2", color="orange")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend(prop={'size': 20})
    plt.title("CPCA")

    plt.subplot(144)
    plt.scatter(ty_estimated[0, :num_datapoints_y//2], ty_estimated[1, :num_datapoints_y//2], label="Foreground group 1", color="green")
    plt.scatter(ty_estimated[0, num_datapoints_y//2:], ty_estimated[1, num_datapoints_y//2:], label="Foreground group 2", color="orange")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.legend(prop={'size': 20})
    plt.title("CPLVM")
    plt.tight_layout()

    plt.savefig("./out/clvm_scatter_comparison.png")
    plt.show()


    # plt.figure(figsize=(7, 5))
    # plt.hist(nn_idx_in_pca, 30, label="PCA", alpha=0.5)
    # plt.hist(nn_idx_in_latent, 30, label="cLVM", alpha=0.5)
    # plt.legend()
    # plt.xlabel("Nearest neighbor index")
    # plt.ylabel("Count")
    # plt.tight_layout()
    # plt.savefig("./out/nn_idx.png")
    # plt.show()

    # acc = np.mean(nn_idx_data == nn_idx_latent)
    # print("NN accuracy: {}".format(acc))
    # import ipdb; ipdb.set_trace()

    # plt.figure(figsize=(21, 5))
    # plt.subplot(131)
    # plt.scatter(zx_estimated[0, :], zx_estimated[1, :], c=labs, alpha=0.1)
    # plt.title("zx")
    # plt.xlabel("Latent dim 1")
    # plt.ylabel("Latent dim 2")
    # plt.subplot(132)
    # plt.scatter(zy_estimated[0, :], zy_estimated[1, :], c=labs, alpha=0.1)
    # plt.title("zy")
    # plt.xlabel("Latent dim 1")
    # plt.ylabel("Latent dim 2")
    # plt.subplot(133)
    # labs = np.zeros(num_datapoints_y)
    # labs[num_datapoints_y//2:] = 1
    # plt.scatter(ty_estimated[0, :], ty_estimated[1, :], c=labs, alpha=0.1)
    # plt.title("ty")
    # plt.xlabel("Latent dim 1")
    # plt.ylabel("Latent dim 2")
    # plt.tight_layout()
    # plt.savefig("./out/scatter_subset_response.png")
    # plt.show()

    
    # s_estimated = np.exp(model_dict['qs_mean'].numpy() + model_dict['qs_stddv'].numpy()**2 / 2)
    # w_estimated = np.exp(model_dict['qw_mean'].numpy() + model_dict['qw_stddv'].numpy()**2 / 2)

    # plt.figure(figsize=(14, 5))
    # plt.subplot(121)
    # sns.heatmap(s_estimated)
    # plt.subplot(122)
    # sns.heatmap(w_estimated)
    # plt.show()

    # import ipdb; ipdb.set_trace()

