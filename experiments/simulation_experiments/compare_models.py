import sys
sys.path.append("../models")

from mimosca_gaussian import fit_model as fit_mimosca_gaussian
from mimosca_poisson_link import fit_model as fit_mimosca_poisson_link
from clvm_tfp_gaussian import fit_model as fit_clvm_gaussian
from clvm_tfp_poisson_link import fit_model as fit_clvm_link
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
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True



# from clvm_tfp_poisson_link import fit_model as fit_clvm
# from clvm_tfp_poisson_link import clvm


tf.enable_v2_behavior()

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    num_datapoints_x = 2000
    num_datapoints_y = 2000
    data_dim = 200
    latent_dim_shared = 5
    latent_dim_target = 5

    NUM_REPEATS = 2
    ELBOlist_clvm_poisson_nonnegative = []
    ELBOlist_clvm_poisson_link = []
    ELBOlist_clvm_gaussian = []
    ELBOlist_mimosca_poisson_link = []
    ELBOlist_mimosca_gaussian = []
    for ii in range(NUM_REPEATS):

        # Simulate data from nonnegative poisson model
        # counts_per_cell_X = np.random.randint(low=20, high=150, size=num_datapoints_x)
        # counts_per_cell_X = np.expand_dims(counts_per_cell_X, 0)
        # counts_per_cell_Y = np.random.randint(low=20, high=150, size=num_datapoints_y)
        # counts_per_cell_Y = np.expand_dims(counts_per_cell_Y, 0)

        concrete_clvm_model = functools.partial(clvm_nonnegative,
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

        X, Y = X_sampled.numpy()[:, :1500], Y_sampled.numpy()[:, :1500]
        X_test, Y_test = X_sampled.numpy()[:, 1500:], Y_sampled.numpy()[:, 1500:]

        num_datapoints_x_test = X_test.shape[1]
        num_datapoints_y_test = Y_test.shape[1]

        # import ipdb; ipdb.set_trace()


        

        # ------ RUN MODELS ---------

        # Nonnegative poisson clVM
        model_dict_clvm_nonnegative_poisson = fit_clvm_nonnegative(
            X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False)
        ELBO_clvm_poisson_nonnegative = -1 * \
            model_dict_clvm_nonnegative_poisson['loss_trace'][-1].numpy() / (
                num_datapoints_x + num_datapoints_y)
        print("ELBO_clvm_poisson_nonnegative: {}".format(
            ELBO_clvm_poisson_nonnegative))

        # ## Test LL

        # Get fitted loadings matrices
        W = np.exp(model_dict_clvm_nonnegative_poisson['qw_mean'].numpy() + model_dict_clvm_nonnegative_poisson['qw_stddv'].numpy()**2)
        S = np.exp(model_dict_clvm_nonnegative_poisson['qs_mean'].numpy() + model_dict_clvm_nonnegative_poisson['qs_stddv'].numpy()**2)

        # Size factors
        counts_per_cell_X=np.mean(X_test, axis = 0)
        counts_per_cell_Y=np.mean(Y_test, axis = 0)

        # Sample LVs randomly
        zx = np.exp(np.random.normal(size=(latent_dim_shared, num_datapoints_x_test)))
        zy = np.exp(np.random.normal(size=(latent_dim_shared, num_datapoints_y_test)))
        ty = np.exp(np.random.normal(size=(latent_dim_target, num_datapoints_y_test)))
        sf_x = np.exp(np.random.normal(loc=np.mean(np.log(counts_per_cell_X)), scale=np.std(np.log(counts_per_cell_X)), size=(1, num_datapoints_x_test)))
        sf_y = np.exp(np.random.normal(loc=np.mean(np.log(counts_per_cell_X)), scale=np.std(np.log(counts_per_cell_Y)), size=(1, num_datapoints_y_test)))

        # Compute predictions
        x_rate = np.multiply(S @ zx, sf_x)
        y_rate = np.multiply(S @ zy + W @ ty, sf_y)
        # import ipdb; ipdb.set_trace()

        # Compute LL
        X_LL = poisson.logpmf(X_test, x_rate)
        Y_LL = poisson.logpmf(Y_test, y_rate)
        LL_cplvm = np.mean(np.concatenate([X_LL, Y_LL], axis=1))
        print("LL CPLVM: ", LL_cplvm)




        # # Poisson link clVM
        model_dict_clvm_poisson_link = fit_clvm_link(
            X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False)
        ELBO_clvm_poisson_link = -1 * \
            model_dict_clvm_poisson_link['loss_trace'][-1].numpy() / \
            (num_datapoints_x + num_datapoints_y)
        print("ELBO_clvm_poisson_link: {}".format(ELBO_clvm_poisson_link))

        ## Test LL

        # Get fitted loadings matrices
        W = model_dict_clvm_poisson_link['qw_mean'].numpy()
        S = model_dict_clvm_poisson_link['qs_mean'].numpy()
        mu_x = model_dict_clvm_poisson_link['qmu_x_mean'].numpy()
        mu_y = model_dict_clvm_poisson_link['qmu_y_mean'].numpy()

        # Size factors
        counts_per_cell_X=np.mean(X_test, axis = 0)
        counts_per_cell_Y=np.mean(Y_test, axis = 0)

        # Sample LVs randomly
        zx = np.random.normal(size=(latent_dim_shared, num_datapoints_x_test))
        zy = np.random.normal(size=(latent_dim_shared, num_datapoints_y_test))
        ty = np.random.normal(size=(latent_dim_target, num_datapoints_y_test))
        sf_x = np.random.normal(loc=np.mean(np.log(counts_per_cell_X)), scale=np.std(np.log(counts_per_cell_X)), size=(1, num_datapoints_x_test))
        sf_y = np.random.normal(loc=np.mean(np.log(counts_per_cell_X)), scale=np.std(np.log(counts_per_cell_Y)), size=(1, num_datapoints_y_test))

        # Compute predictions
        x_rate = np.multiply(np.exp(S @ zx + mu_x), sf_x)
        y_rate = np.multiply(np.exp(S @ zy + W @ ty + mu_y), sf_y)

        # Compute LL
        X_LL = poisson.logpmf(X_test, x_rate)
        Y_LL = poisson.logpmf(Y_test, y_rate)
        LL_cglvm = np.mean(np.concatenate([X_LL, Y_LL], axis=1))
        print("LL CGLVM: ", LL_cglvm)


        # Gaussian cLVM
        # model_dict_clvm_gaussian = fit_clvm_gaussian(
        #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=False, is_H0=False)
        # ELBO_clvm_gaussian = -1 * \
        #     model_dict_clvm_gaussian['loss_trace'][-1].numpy() / \
        #     (num_datapoints_x + num_datapoints_y)
        # print("ELBO_clvm_gaussian: {}".format(ELBO_clvm_gaussian))

        # # Poisson MIMOSCA
        model_dict_mimosca_poisson_link = fit_mimosca_poisson_link(
            X, Y, compute_size_factors=True, is_H0=False)
        ELBO_mimosca_poisson_link = -1 * \
            model_dict_mimosca_poisson_link['loss_trace'][-1].numpy() / (
                num_datapoints_x + num_datapoints_y)
        print("ELBO_mimosca_poisson_link: {}".format(ELBO_mimosca_poisson_link))

        ## Test LL

        # Get fitted loadings matrices
        mu = model_dict_mimosca_poisson_link['qmu_mean'].numpy()
        beta = model_dict_mimosca_poisson_link['qbeta_mean'].numpy()

        data_test = np.concatenate([X_test, Y_test], axis=1)

        # Size factors
        counts_per_cell = np.mean(data_test, axis = 0)

        # Sample LVs randomly
        sf = np.random.normal(loc=np.mean(np.log(counts_per_cell)), scale=np.std(np.log(counts_per_cell)), size=(1, data_test.shape[1]))

        # Compute predictions
        dummy = np.zeros(data_test.shape[1])
        dummy[num_datapoints_x_test:] = 1
        dummy = np.expand_dims(dummy, 0)
        rate = np.exp(beta @ dummy + mu + np.log(sf))

        # Compute LL
        LL_cglvm = np.mean(poisson.logpmf(data_test, rate))
        print("LL GLM: ", LL_cglvm)

        import ipdb; ipdb.set_trace()
        # # Gaussian MIMOSCA
        # model_dict_mimosca_gaussian = fit_mimosca_gaussian(
        #     X, Y, compute_size_factors=False, is_H0=False)
        # ELBO_mimosca_gaussian = -1 * \
        #     model_dict_mimosca_gaussian['loss_trace'][-1].numpy() / \
        #     (num_datapoints_x + num_datapoints_y)
        # print("ELBO_mimosca_gaussian: {}".format(ELBO_mimosca_gaussian))

        ELBOlist_clvm_poisson_nonnegative.append(ELBO_clvm_poisson_nonnegative)
        ELBOlist_clvm_poisson_link.append(ELBO_clvm_poisson_link)
        # ELBOlist_clvm_gaussian.append(ELBO_clvm_gaussian)
        ELBOlist_mimosca_poisson_link.append(ELBO_mimosca_poisson_link)
        # ELBOlist_mimosca_gaussian.append(ELBO_mimosca_gaussian)

        ELBO_list = [ELBOlist_clvm_poisson_nonnegative,
                     ELBOlist_clvm_poisson_link, ELBOlist_mimosca_poisson_link]

        plt.figure(figsize=(10, 6))
        sns.boxplot(np.arange(len(ELBO_list)), ELBO_list)
        plt.xticks(np.arange(len(ELBO_list)), labels=[
                   "CPLVM", "CGLVM", "Poisson\nGLM"])
        plt.ylabel("ELBO")
        plt.tight_layout()
        plt.savefig("./out/model_comparison_elbo_clvm.png")
        plt.close()

        # ELBO_list = [ELBOlist_mimosca_poisson_link, ELBOlist_mimosca_gaussian]
        # plt.figure(figsize=(7, 5))
        # sns.boxplot(np.arange(len(ELBO_list)), ELBO_list)
        # plt.xticks(np.arange(len(ELBO_list)), labels=[
        #            "MIMOSCA, Poisson link", "MIMOSCA, Gaussian"])
        # plt.ylabel("ELBO")
        # plt.savefig("./out/model_comparison_elbo_mimosca.png")
        # plt.close()
