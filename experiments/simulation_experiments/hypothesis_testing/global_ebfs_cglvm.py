import matplotlib

from cplvm import CPLVM, CGLVM
from cplvm import CGLVMMFGaussianApprox

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

    num_datapoints_x = 1000
    num_datapoints_y = 1000
    data_dim = 200
    latent_dim_shared = 3
    latent_dim_foreground = 3

    actual_a, actual_b = 3, 3

    NUM_REPEATS = 20
    bfs_experiment = []
    bfs_control = []
    bfs_shuffled = []

    bfs_experiment_cglvm = []
    bfs_control_cglvm = []
    bfs_shuffled_cglvm = []
    for ii in range(NUM_REPEATS):

        # ------- generate data ---------

        cplvm_for_data = CPLVM(
            k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
        )

        concrete_cplvm_model = functools.partial(
            cplvm_for_data.model,
            data_dim=data_dim,
            num_datapoints_x=num_datapoints_x,
            num_datapoints_y=num_datapoints_y,
            counts_per_cell_X=1,
            counts_per_cell_Y=1,
            is_H0=False,
        )

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_cplvm_model)

        deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
        # mu_x, mu_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()

        X, Y = X_sampled.numpy(), Y_sampled.numpy()

        ########## "Treatment" data CGLVM ##########

        # Run H0 and H1 models on data
        cglvm = CGLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

        approx_model = CGLVMMFGaussianApprox(
            X, Y, latent_dim_shared, latent_dim_foreground
        )
        H1_results = cglvm.fit_model_vi(
            X, Y, approx_model, compute_size_factors=False, is_H0=False
        )
        H0_results = cglvm.fit_model_vi(X, Y, compute_size_factors=False, is_H0=True)

        H1_elbo = (
            -1
            * H1_results["loss_trace"][-1].numpy()
            / (num_datapoints_x + num_datapoints_y)
        )

        H0_elbo = (
            -1
            * H0_results["loss_trace"][-1].numpy()
            / (num_datapoints_x + num_datapoints_y)
        )

        curr_bf = H1_elbo - H0_elbo
        print("BF treatment: {}".format(curr_bf))
        bfs_experiment_cglvm.append(curr_bf)

        # ########## Shuffled data CGLVM ##########
        # Shuffle background and foreground labels

        all_data = np.concatenate([X, Y], axis=1)
        shuffled_idx = np.random.permutation(
            np.arange(num_datapoints_x + num_datapoints_y)
        )
        x_idx = shuffled_idx[:num_datapoints_x]
        y_idx = shuffled_idx[num_datapoints_x:]
        X = all_data[:, x_idx]
        Y = all_data[:, y_idx]

        # Run H0 and H1 models on data
        cglvm = CGLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

        H1_results = cglvm.fit_model_vi(X, Y, compute_size_factors=False, is_H0=False)
        H0_results = cglvm.fit_model_vi(X, Y, compute_size_factors=False, is_H0=True)

        H1_elbo = (
            -1
            * H1_results["loss_trace"][-1].numpy()
            / (num_datapoints_x + num_datapoints_y)
        )
        H0_elbo = (
            -1
            * H0_results["loss_trace"][-1].numpy()
            / (num_datapoints_x + num_datapoints_y)
        )

        curr_bf = H1_elbo - H0_elbo
        print("BF shuffled: {}".format(curr_bf))
        bfs_shuffled_cglvm.append(curr_bf)

        ########## Negative control data ##########

        # Simulate data from null model

        cplvm_for_data = CPLVM(
            k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
        )

        concrete_cplvm_model = functools.partial(
            cplvm_for_data.model,
            data_dim=data_dim,
            num_datapoints_x=num_datapoints_x,
            num_datapoints_y=num_datapoints_y,
            counts_per_cell_X=1,
            counts_per_cell_Y=1,
            is_H0=True,
        )

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_cplvm_model)

        deltax, sf_x, sf_y, s, zx, zy, X_sampled, Y_sampled = model.sample()

        X, Y = X_sampled.numpy(), Y_sampled.numpy()

        # Run H0 and H1 models on data
        cglvm = CGLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

        H1_results = cglvm.fit_model_vi(X, Y, compute_size_factors=False, is_H0=False)
        H0_results = cglvm.fit_model_vi(X, Y, compute_size_factors=False, is_H0=True)

        H1_elbo = (
            -1
            * H1_results["loss_trace"][-1].numpy()
            / (num_datapoints_x + num_datapoints_y)
        )
        H0_elbo = (
            -1
            * H0_results["loss_trace"][-1].numpy()
            / (num_datapoints_x + num_datapoints_y)
        )

        curr_bf = H1_elbo - H0_elbo
        print("BF negative control: {}".format(curr_bf))
        bfs_control_cglvm.append(curr_bf)

        import ipdb

        ipdb.set_trace()

        # plt.figure(figsize=(9, 8))
        # sns.boxplot(np.arange(3), [bfs_control_cglvm, bfs_shuffled_cglvm, bfs_experiment_cglvm])
        # plt.title("Global EBFs, CGLVM")
        # plt.xticks(np.arange(3), labels=[
        #            "Unperturbed\nnull", "Shuffled\nnull", "Perturbed"])
        # plt.ylabel("log(EBF)")
        # plt.tight_layout()
        # plt.savefig("./out/evidences_simulated_data_cglvm.png")
        # plt.close()
