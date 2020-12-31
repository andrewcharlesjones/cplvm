import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import poisson
from scipy.special import logsumexp
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import sys
sys.path.append("../models")

from clvm_tfp_poisson import fit_model as fit_clvm
from clvm_tfp_poisson import clvm
from pca_poisson import fit_model as fit_pca

import matplotlib
font = {'size'   : 18}
matplotlib.rc('font', **font)


# from clvm_tfp_poisson_link import fit_model as fit_clvm
# from clvm_tfp_poisson_link import clvm


tf.enable_v2_behavior()

plt.style.use("ggplot")
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    p_list = [10, 100, 1000]

    for data_dim in p_list:

        num_datapoints_x = 100
        num_datapoints_y = 100
        # data_dim = 200
        latent_dim_shared = 3
        latent_dim_target = 3

        actual_a, actual_b = 3, 3

        NUM_REPEATS = 50
        bfs_experiment = []
        # bfs_control = []
        bfs_shuffled = []
        for ii in range(NUM_REPEATS):

            # counts_per_cell_X = np.random.randint(low=20, high=150, size=num_datapoints_x)
            # counts_per_cell_X = np.expand_dims(counts_per_cell_X, 0)
            # counts_per_cell_Y = np.random.randint(low=20, high=150, size=num_datapoints_y)
            # counts_per_cell_Y = np.expand_dims(counts_per_cell_Y, 0)

            # ------- Specify model ---------

            concrete_clvm_model = functools.partial(clvm,
                                                    data_dim=data_dim,
                                                    latent_dim_shared=latent_dim_shared,
                                                    latent_dim_target=latent_dim_target,
                                                    num_datapoints_x=num_datapoints_x,
                                                    num_datapoints_y=num_datapoints_y,
                                                    counts_per_cell_X=1,
                                                    counts_per_cell_Y=1,
                                                    is_H0=False)

            model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

            deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
            
            X, Y = X_sampled.numpy(), Y_sampled.numpy()

            ## Run H0 and H1 models on data

            H1_results = fit_clvm(
                X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False)
            
            H0_results = fit_clvm(
                X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=True)

            H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)

            H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)

            curr_bf = H1_elbo - H0_elbo
            print("BF treatment: {}".format(curr_bf))
            bfs_experiment.append(curr_bf)


            ### Shuffle background and foreground labels

            # all_data = np.concatenate([X, Y], axis=1)
            # shuffled_idx = np.random.permutation(np.arange(num_datapoints_x + num_datapoints_y))
            # x_idx = shuffled_idx[:num_datapoints_x]
            # y_idx = shuffled_idx[num_datapoints_x:]
            # X = all_data[:, x_idx]
            # Y = all_data[:, y_idx]

            # ## Run H0 and H1 models on data

            # H1_results = fit_clvm(
            #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False)
            # H0_results = fit_clvm(
            #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=True)


            # H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)
            # H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)

            # curr_bf = H1_elbo - H0_elbo
            # print("BF shuffled: {}".format(curr_bf))
            # bfs_shuffled.append(curr_bf)



            ## Simulate data from null model

            concrete_clvm_model = functools.partial(clvm,
                                                    data_dim=data_dim,
                                                    latent_dim_shared=latent_dim_shared,
                                                    latent_dim_target=latent_dim_target,
                                                    num_datapoints_x=num_datapoints_x,
                                                    num_datapoints_y=num_datapoints_y,
                                                    counts_per_cell_X=1,
                                                    counts_per_cell_Y=1,
                                                    is_H0=True)

            model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

            deltax, sf_x, sf_y, s, zx, zy, X_sampled, Y_sampled = model.sample()
            
            X, Y = X_sampled.numpy(), Y_sampled.numpy()

            ## Run H0 and H1 models on data

            H1_results = fit_clvm(
                X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False)
            H0_results = fit_clvm(
                X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=True)


            H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)
            H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)

            curr_bf = H1_elbo - H0_elbo
            print("BF control: {}".format(curr_bf))
            bfs_shuffled.append(curr_bf)

                

            # bfs_control = np.array(bfs_control)[~np.isnan(bfs_control)]
            bfs_experiment = list(np.array(bfs_experiment)[~np.isnan(bfs_experiment)])
            bfs_shuffled = list(np.array(bfs_shuffled)[~np.isnan(bfs_shuffled)])
            # tpr_true, fpr_true, thresholds_true = roc_curve(y_true=np.concatenate([np.zeros(len(bfs_control)), np.ones(len(bfs_experiment))]), y_score=np.concatenate([bfs_control, bfs_experiment]))
            tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(y_true=np.concatenate([np.zeros(len(bfs_shuffled)), np.ones(len(bfs_experiment))]), y_score=np.concatenate([bfs_shuffled, bfs_experiment]))

            np.save("./out/cai/bfs_experiment_p{}.npy".format(data_dim), bfs_experiment)
            np.save("./out/cai/bfs_shuffled_p{}.npy".format(data_dim), bfs_shuffled)

            auc = roc_auc_score(y_true=np.concatenate([np.zeros(len(bfs_shuffled)), np.ones(len(bfs_experiment))]), y_score=np.concatenate([bfs_shuffled, bfs_experiment]))

            plt.figure(figsize=(7, 5))
            # plt.plot(tpr_true, fpr_true, label="Experiment vs. null")
            plt.plot(tpr_shuffled, fpr_shuffled, label="Experiment vs. shuffled")
            plt.plot([0, 1], [0, 1], '--', color='black')
            plt.title("AUC={}".format(round(auc, 2)))
            plt.legend()
            plt.xlabel("Power")
            plt.ylabel("Size")
            plt.tight_layout()
            plt.savefig("./out/bf_roc_curve_p{}.png".format(data_dim))
            plt.close()
        #     plt.show()
        # import ipdb; ipdb.set_trace()

