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


import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    num_datapoints_x = 2000
    num_datapoints_y = 2000
    data_dim = 200
    latent_dim_shared = 2
    latent_dim_target = 2

    NUM_REPEATS = 3
    # ELBOlist_clvm_poisson_nonnegative = []
    # ELBOlist_clvm_poisson_link = []
    # ELBOlist_clvm_gaussian = []
    # ELBOlist_mimosca_poisson_link = []
    # ELBOlist_mimosca_gaussian = []

    error_list_cplvm = []
    error_list_cglvm = []
    error_list_glm = []
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

        # X, Y = X_sampled.numpy()[:, :1500], Y_sampled.numpy()[:, :1500]
        # X_test, Y_test = X_sampled.numpy()[:, 1500:], Y_sampled.numpy()[:, 1500:]

        # num_datapoints_x_test = X_test.shape[1]
        # num_datapoints_y_test = Y_test.shape[1]

        # ------ RUN MODELS ---------

        # Nonnegative poisson clVM
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
        # print("ELBO_clvm_poisson_nonnegative: {}".format(
        #     ELBO_clvm_poisson_nonnegative))

        # ## Test LL

        # Get fitted loadings matrices
        W = np.exp(
            model_dict_clvm_nonnegative_poisson["qw_mean"].numpy()
            + model_dict_clvm_nonnegative_poisson["qw_stddv"].numpy() ** 2
        )
        S = np.exp(
            model_dict_clvm_nonnegative_poisson["qs_mean"].numpy()
            + model_dict_clvm_nonnegative_poisson["qs_stddv"].numpy() ** 2
        )

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

        sf_x = np.exp(
            model_dict_clvm_nonnegative_poisson["qsize_factors_x_mean"].numpy()
            + model_dict_clvm_nonnegative_poisson["qsize_factor_x_stddv"].numpy() ** 2
        )
        sf_y = np.exp(
            model_dict_clvm_nonnegative_poisson["qsize_factors_y_mean"].numpy()
            + model_dict_clvm_nonnegative_poisson["qsize_factor_y_stddv"].numpy() ** 2
        )

        # Compute reconstructions
        x_rate = np.multiply(S @ zx, sf_x)
        y_rate = np.multiply(S @ zy + W @ ty, sf_y)
        preds = np.concatenate([x_rate, y_rate], axis=1)

        truth = np.concatenate([X, Y], axis=1)

        # Compute reconstruction error
        error_cplvm = np.mean((truth - preds) ** 2)
        print("Error CPLVM: ", error_cplvm)

        # Poisson link clVM
        model_dict_clvm_poisson_link = fit_clvm_link_map(
            X,
            Y,
            latent_dim_shared,
            latent_dim_target,
            compute_size_factors=True,
            is_H0=False,
        )
        # ELBO_clvm_poisson_link = -1 * \
        #     model_dict_clvm_poisson_link['loss_trace'][-1].numpy() / \
        #     (num_datapoints_x + num_datapoints_y)
        # print("ELBO_clvm_poisson_link: {}".format(ELBO_clvm_poisson_link))

        ## Test LL

        # # Get fitted loadings matrices
        # W = model_dict_clvm_poisson_link['qw_mean'].numpy()
        # S = model_dict_clvm_poisson_link['qs_mean'].numpy()
        # mu_x = model_dict_clvm_poisson_link['qmu_x_mean'].numpy()
        # mu_y = model_dict_clvm_poisson_link['qmu_y_mean'].numpy()

        # zx = model_dict_clvm_poisson_link['qzx_mean'].numpy()
        # zy = model_dict_clvm_poisson_link['qzy_mean'].numpy()
        # ty = model_dict_clvm_poisson_link['qty_mean'].numpy()
        # sf_x = np.exp(model_dict_clvm_poisson_link['qsize_factor_x_mean'].numpy() + model_dict_clvm_poisson_link['qsize_factor_x_stddv'].numpy()**2)
        # sf_y = np.exp(model_dict_clvm_poisson_link['qsize_factor_y_mean'].numpy() + model_dict_clvm_poisson_link['qsize_factor_y_stddv'].numpy()**2)

        # Get fitted loadings matrices
        W = model_dict_clvm_poisson_link["w"].numpy()
        S = model_dict_clvm_poisson_link["s"].numpy()
        mu_x = model_dict_clvm_poisson_link["mu_x"].numpy()
        mu_y = model_dict_clvm_poisson_link["mu_y"].numpy()

        zx = model_dict_clvm_poisson_link["zx"].numpy()
        zy = model_dict_clvm_poisson_link["zy"].numpy()
        ty = model_dict_clvm_poisson_link["ty"].numpy()

        # Compute predictions
        x_rate = np.exp(S @ zx + mu_x)  # + np.log(sf_x))
        y_rate = np.exp(S @ zy + W @ ty + mu_y)  # + np.log(sf_y))

        preds = np.concatenate([x_rate, y_rate], axis=1)

        truth = np.concatenate([X, Y], axis=1)

        # Compute reconstruction error
        error_cglvm = np.mean((truth - preds) ** 2)
        print("Error CGLVM: ", error_cglvm)
        # import ipdb; ipdb.set_trace()

        # Gaussian cLVM
        # model_dict_clvm_gaussian = fit_clvm_gaussian(
        #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=False, is_H0=False)
        # ELBO_clvm_gaussian = -1 * \
        #     model_dict_clvm_gaussian['loss_trace'][-1].numpy() / \
        #     (num_datapoints_x + num_datapoints_y)
        # print("ELBO_clvm_gaussian: {}".format(ELBO_clvm_gaussian))

        # # Poisson MIMOSCA
        model_dict_mimosca_poisson_link = fit_mimosca_poisson_link(
            X, Y, compute_size_factors=True, is_H0=False
        )
        # ELBO_mimosca_poisson_link = -1 * \
        #     model_dict_mimosca_poisson_link['loss_trace'][-1].numpy() / (
        #         num_datapoints_x + num_datapoints_y)
        # print("ELBO_mimosca_poisson_link: {}".format(ELBO_mimosca_poisson_link))

        ## Test LL

        # Get fitted loadings matrices
        mu = model_dict_mimosca_poisson_link["qmu_mean"].numpy()
        beta = model_dict_mimosca_poisson_link["qbeta_mean"].numpy()
        sf = np.exp(
            model_dict_mimosca_poisson_link["qsize_factor_mean"].numpy()
            + model_dict_mimosca_poisson_link["qsize_factor_stddv"].numpy() ** 2
        )

        # Compute reconstructions
        dummy = np.zeros(X.shape[1] + Y.shape[1])
        dummy[X.shape[1] :] = 1
        dummy = np.expand_dims(dummy, 0)
        preds = np.exp(beta @ dummy + mu)  # + np.log(sf))

        # Compute error
        truth = np.concatenate([X, Y], axis=1)
        error_glm = np.mean((truth - preds) ** 2)
        print("Error Poisson GLM: ", error_glm)

        # import ipdb; ipdb.set_trace()
        # # Gaussian MIMOSCA
        # model_dict_mimosca_gaussian = fit_mimosca_gaussian(
        #     X, Y, compute_size_factors=False, is_H0=False)
        # ELBO_mimosca_gaussian = -1 * \
        #     model_dict_mimosca_gaussian['loss_trace'][-1].numpy() / \
        #     (num_datapoints_x + num_datapoints_y)
        # print("ELBO_mimosca_gaussian: {}".format(ELBO_mimosca_gaussian))

        # ELBOlist_clvm_poisson_nonnegative.append(ELBO_clvm_poisson_nonnegative)
        # ELBOlist_clvm_poisson_link.append(ELBO_clvm_poisson_link)
        # # ELBOlist_clvm_gaussian.append(ELBO_clvm_gaussian)
        # ELBOlist_mimosca_poisson_link.append(ELBO_mimosca_poisson_link)
        # # ELBOlist_mimosca_gaussian.append(ELBO_mimosca_gaussian)

        error_list_cplvm.append(error_cplvm)
        error_list_cglvm.append(error_cglvm)
        error_list_glm.append(error_glm)

        error_list = [error_list_cplvm, error_list_cglvm, error_list_glm]

        plt.figure(figsize=(9, 6))
        # import ipdb; ipdb.set_trace()
        sns.boxplot(np.arange(len(error_list)), error_list)
        plt.xticks(
            np.arange(len(error_list)), labels=["CPLVM", "CGLVM", "Poisson\nGLM"]
        )
        plt.ylabel("Error")
        plt.tight_layout()
        plt.savefig("./out/model_comparison.png")
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
