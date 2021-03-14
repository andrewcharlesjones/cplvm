import sys

sys.path.append("../../models")

from mimosca_gaussian import fit_model as fit_mimosca_gaussian
from mimosca_poisson_link import fit_model as fit_mimosca_poisson_link
from clvm_tfp_gaussian import fit_model as fit_clvm_gaussian
from clvm_tfp_poisson_link import fit_model as fit_clvm_link
from clvm_tfp_poisson_link import fit_model_map as fit_clvm_link_map
from clvm_tfp_poisson_link import clvm as clvm_link
from clvm_tfp_poisson import clvm as clvm_nonnegative
from clvm_tfp_poisson import fit_model as fit_clvm_nonnegative
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

from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA, NMF
from pcpca import CPCA


import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    num_datapoints_x = 100
    num_datapoints_y = 100
    data_dim = 200
    latent_dim_shared = 5
    latent_dim_target = 3

    NUM_REPEATS = 10

    error_list_pca = []
    error_list_nmf = []
    error_list_cpca = []
    error_list_cplvm = []
    error_list_cglvm = []

    error_list_pca_bg = []
    error_list_nmf_bg = []
    error_list_cpca_bg = []
    error_list_cplvm_bg = []
    error_list_cglvm_bg = []
    for ii in range(NUM_REPEATS):

        concrete_clvm_model = functools.partial(
            clvm_nonnegative,
            data_dim=data_dim,
            latent_dim_shared=latent_dim_shared,
            latent_dim_target=latent_dim_target,
            num_datapoints_x=num_datapoints_x,
            num_datapoints_y=num_datapoints_y,
            counts_per_cell_X=1,
            counts_per_cell_Y=1,
            is_H0=False,
        )

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

        samples = model.sample()
        X_sampled, Y_sampled = samples[-2], samples[-1]
        X, Y = X_sampled.numpy(), Y_sampled.numpy()
        ty_truth = samples[-3].numpy()
        zx_truth = samples[-6].numpy()

        # ------ RUN MODELS ---------

        ##### PCA #####

        # Foreground
        pca = PCA(n_components=latent_dim_target)
        ty = pca.fit_transform(Y)
        # Estimated distance matrix
        distance_mat = pairwise_distances(ty, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(ty_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_pca = np.mean(w_dists)

        print("Error PCA fg: ", w_dist_pca)
        error_list_pca.append(w_dist_pca)

        # Background
        pca = PCA(n_components=latent_dim_target)
        ty = pca.fit_transform(X)
        # Estimated distance matrix
        distance_mat = pairwise_distances(ty, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(zx_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_pca = np.mean(w_dists)

        print("Error PCA bg: ", w_dist_pca)
        error_list_pca_bg.append(w_dist_pca)

        ##### NMF #####

        # Foreground
        nmf = NMF(n_components=latent_dim_target)
        ty = nmf.fit_transform(Y)
        # Estimated distance matrix
        distance_mat = pairwise_distances(ty, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(ty_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_pca = np.mean(w_dists)

        print("Error NMF fg: ", w_dist_pca)
        error_list_nmf.append(w_dist_pca)

        # Background
        nmf = NMF(n_components=latent_dim_target)
        ty = nmf.fit_transform(X)
        # Estimated distance matrix
        distance_mat = pairwise_distances(ty, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(zx_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_pca = np.mean(w_dists)

        print("Error NMF bg: ", w_dist_pca)
        error_list_nmf_bg.append(w_dist_pca)

        ##### CPCA #####

        # Foreground
        cpca = CPCA(n_components=latent_dim_target, gamma=0.9)
        ty, zx = cpca.fit_transform((Y.T - Y.mean(1)).T, (X.T - X.mean(1)).T)
        # Estimated distance matrix
        distance_mat = pairwise_distances(ty.T, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(ty_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_cpca = np.mean(w_dists)

        print("Error CPCA: ", w_dist_cpca)
        error_list_cpca.append(w_dist_cpca)

        # Background
        cpca = CPCA(n_components=latent_dim_target, gamma=0.9)
        ty, zx = cpca.fit_transform((Y.T - Y.mean(1)).T, (X.T - X.mean(1)).T)
        # Estimated distance matrix
        distance_mat = pairwise_distances(zx.T, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(zx_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_cpca = np.mean(w_dists)

        print("Error CPCA: ", w_dist_cpca)
        error_list_cpca_bg.append(w_dist_cpca)

        ##### Nonnegative poisson clVM #####
        model_dict_clvm_nonnegative_poisson = fit_clvm_nonnegative(
            X,
            Y,
            latent_dim_shared,
            latent_dim_target,
            compute_size_factors=True,
            is_H0=False,
            offset_term=False,
        )
        ELBO_clvm_poisson_nonnegative = (
            -1
            * model_dict_clvm_nonnegative_poisson["loss_trace"][-1].numpy()
            / (num_datapoints_x + num_datapoints_y)
        )

        # Test LL

        # Get fitted loadings matrices
        zx = np.exp(
            model_dict_clvm_nonnegative_poisson["qzx_mean"].numpy()
            + model_dict_clvm_nonnegative_poisson["qzx_stddv"].numpy() ** 2
        )
        zy = np.exp(
            model_dict_clvm_nonnegative_poisson["qzy_mean"].numpy()
            + model_dict_clvm_nonnegative_poisson["qzy_stddv"].numpy() ** 2
        )
        ty = np.exp(
            model_dict_clvm_nonnegative_poisson["qty_mean"].numpy()
            + model_dict_clvm_nonnegative_poisson["qty_stddv"].numpy() ** 2
        )

        # Foreground
        # Estimated distance matrix
        distance_mat = pairwise_distances(ty.T, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(ty_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_cplvm = np.mean(w_dists)

        print("Error CPLVM: ", w_dist_cplvm)
        error_list_cplvm.append(w_dist_cplvm)

        # Background
        # Estimated distance matrix
        distance_mat = pairwise_distances(zx.T, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(zx_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_cplvm = np.mean(w_dists)

        print("Error CPLVM: ", w_dist_cplvm)
        error_list_cplvm_bg.append(w_dist_cplvm)
        # import ipdb; ipdb.set_trace()

        ##### Poisson link clVM #####

        model_dict_clvm_poisson_link = fit_clvm_link_map(
            X,
            Y,
            latent_dim_shared,
            latent_dim_target,
            compute_size_factors=True,
            is_H0=False,
        )

        # Get fitted loadings matrices
        W = model_dict_clvm_poisson_link["w"].numpy()
        S = model_dict_clvm_poisson_link["s"].numpy()
        mu_x = model_dict_clvm_poisson_link["mu_x"].numpy()
        mu_y = model_dict_clvm_poisson_link["mu_y"].numpy()

        zx = model_dict_clvm_poisson_link["zx"].numpy()
        zy = model_dict_clvm_poisson_link["zy"].numpy()
        ty = model_dict_clvm_poisson_link["ty"].numpy()

        # Foreground
        # Estimated distance matrix
        distance_mat = pairwise_distances(ty.T, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(ty_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_cglvm = np.mean(w_dists)

        print("Error CPLVM: ", w_dist_cglvm)
        error_list_cglvm.append(w_dist_cglvm)

        # Background
        # Estimated distance matrix
        distance_mat = pairwise_distances(zx.T, metric="euclidean")
        row_sums = distance_mat.sum(axis=1)
        distance_mat_normalized = distance_mat / row_sums[:, np.newaxis]

        # True distance matrix
        distance_mat_truth = pairwise_distances(zx_truth.T, metric="euclidean")
        row_sums = distance_mat_truth.sum(axis=1)
        distance_mat_normalized_truth = distance_mat_truth / row_sums[:, np.newaxis]

        # Compute Wasserstein distance
        w_dists = []
        for ii in range(num_datapoints_y):
            curr_dist = wasserstein_distance(
                distance_mat_normalized_truth[ii, :], distance_mat_normalized[ii, :]
            )
            w_dists.append(curr_dist)

        w_dist_cglvm = np.mean(w_dists)

        print("Error CPLVM: ", w_dist_cglvm)
        error_list_cglvm_bg.append(w_dist_cglvm)

        method_list = ["PCA", "NMF", "CPCA", "CGLVM", "CPLVM"]

        plt.figure(figsize=(18, 6))

        plt.subplot(121)
        error_list = [
            error_list_pca,
            error_list_nmf,
            error_list_cpca,
            error_list_cglvm,
            error_list_cplvm,
        ]
        sns.boxplot(np.arange(len(error_list)), error_list)
        plt.xticks(np.arange(len(error_list)), labels=method_list)
        plt.ylabel("Avg. Wasserstein distance")
        plt.title("Foreground")

        plt.subplot(122)
        error_list = [
            error_list_pca_bg,
            error_list_nmf_bg,
            error_list_cpca_bg,
            error_list_cglvm_bg,
            error_list_cplvm_bg,
        ]
        sns.boxplot(np.arange(len(error_list)), error_list)
        plt.xticks(np.arange(len(error_list)), labels=method_list)
        plt.ylabel("Avg. Wasserstein distance")
        plt.title("Background")
        plt.tight_layout()
        plt.savefig("./out/model_comparison_wasserstein.png")
        plt.close()
        # import ipdb; ipdb.set_trace()

        # ELBO_list = [ELBOlist_mimosca_poisson_link, ELBOlist_mimosca_gaussian]
        # plt.figure(figsize=(7, 5))
        # sns.boxplot(np.arange(len(ELBO_list)), ELBO_list)
        # plt.xticks(np.arange(len(ELBO_list)), labels=[
        #            "MIMOSCA, Poisson link", "MIMOSCA, Gaussian"])
        # plt.ylabel("ELBO")
        # plt.savefig("./out/model_comparison_elbo_mimosca.png")
        # plt.close()
