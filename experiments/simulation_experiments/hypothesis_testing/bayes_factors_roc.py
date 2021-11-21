import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import poisson
from scipy.special import logsumexp
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox

import functools
import warnings
import tensorflow.compat.v2 as tf


from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import matplotlib

font = {"size": 18}
matplotlib.rc("font", **font)

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    p_list = [10, 100, 1000]

    NUM_REPEATS = 50

    for ii in range(NUM_REPEATS):

        num_datapoints_x = 100
        num_datapoints_y = 100
        latent_dim_shared = 3
        latent_dim_foreground = 3

        bfs_experiment = []
        # bfs_control = []
        bfs_shuffled = []

        for data_dim in p_list:

            cplvm_for_data = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )

            concrete_clvm_model = functools.partial(
                cplvm_for_data.model,
                data_dim=data_dim,
                num_datapoints_x=num_datapoints_x,
                num_datapoints_y=num_datapoints_y,
                counts_per_cell_X=1,
                counts_per_cell_Y=1,
                is_H0=False,
            )

            model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

            deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()

            X, Y = X_sampled.numpy(), Y_sampled.numpy()

            ## Run H0 and H1 models on data
            cplvm = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )
            approx_model = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,  # offset_term=False
            )

            H1_results = cplvm._fit_model_vi(
                X,
                Y,
                approx_model,
                compute_size_factors=True,
                is_H0=False,  # offset_term=False
            )

            approx_model = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                is_H0=True,  # offset_term=False
            )
            H0_results = cplvm._fit_model_vi(
                X, Y, approx_model, compute_size_factors=True, is_H0=True
            )  # offset_term=False)

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
            bfs_experiment.append(curr_bf)

            ### Shuffle background and foreground labels

            all_data = np.concatenate([X, Y], axis=1)
            shuffled_idx = np.random.permutation(
                np.arange(num_datapoints_x + num_datapoints_y)
            )
            x_idx = shuffled_idx[:num_datapoints_x]
            y_idx = shuffled_idx[num_datapoints_x:]
            X = all_data[:, x_idx]
            Y = all_data[:, y_idx]

            ## Run H0 and H1 models on data
            cplvm = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )

            approx_model = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,  # , offset_term=False
            )
            H1_results = cplvm._fit_model_vi(
                X,
                Y,
                approx_model,
                compute_size_factors=True,
                is_H0=False,  # offset_term=False
            )

            approx_model = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                is_H0=True,  # , offset_term=False
            )
            H0_results = cplvm._fit_model_vi(
                X, Y, approx_model, compute_size_factors=True, is_H0=True
            )  # , offset_term=False)

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
            bfs_shuffled.append(curr_bf)

            # bfs_control = np.array(bfs_control)[~np.isnan(bfs_control)]
            bfs_experiment = list(np.array(bfs_experiment)[~np.isnan(bfs_experiment)])
            bfs_shuffled = list(np.array(bfs_shuffled)[~np.isnan(bfs_shuffled)])
            # tpr_true, fpr_true, thresholds_true = roc_curve(y_true=np.concatenate([np.zeros(len(bfs_control)), np.ones(len(bfs_experiment))]), y_score=np.concatenate([bfs_control, bfs_experiment]))

            print(
                np.concatenate(
                    [np.zeros(len(bfs_shuffled)), np.ones(len(bfs_experiment))]
                )
            )
            print(np.concatenate([bfs_shuffled, bfs_experiment]))
            tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
                y_true=np.concatenate(
                    [np.zeros(len(bfs_shuffled)), np.ones(len(bfs_experiment))]
                ),
                y_score=np.concatenate([bfs_shuffled, bfs_experiment]),
            )

            np.save(
                "../out/cai/bfs_experiment_p{}.npy".format(data_dim), bfs_experiment
            )
            np.save("../out/cai/bfs_shuffled_p{}.npy".format(data_dim), bfs_shuffled)

            auc = roc_auc_score(
                y_true=np.concatenate(
                    [np.zeros(len(bfs_shuffled)), np.ones(len(bfs_experiment))]
                ),
                y_score=np.concatenate([bfs_shuffled, bfs_experiment]),
            )

            plt.figure(figsize=(7, 5))
            # plt.plot(tpr_true, fpr_true, label="Experiment vs. null")
            plt.plot(tpr_shuffled, fpr_shuffled, label="Experiment vs. shuffled")
            plt.plot([0, 1], [0, 1], "--", color="black")
            plt.title("AUC={}".format(round(auc, 2)))
            plt.legend()
            plt.xlabel("Power")
            plt.ylabel("Size")
            plt.tight_layout()
            plt.savefig("../out/bf_roc_curve_p{}.png".format(data_dim))
            plt.close()
