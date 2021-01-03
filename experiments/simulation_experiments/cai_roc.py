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

from cai2013_test import cai_test

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
        decisions_experiment = []
        decisions_shuffled = []
        test_stats_experiment = []
        test_stats_shuffled = []
        for ii in range(NUM_REPEATS):

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

            # Run test
            test_stat, curr_reject = cai_test(X, Y, verbose=True)
            decisions_experiment.append(curr_reject)
            test_stats_experiment.append(test_stat)


            ### Shuffle background and foreground labels

            # all_data = np.concatenate([X, Y], axis=1)
            # shuffled_idx = np.random.permutation(np.arange(num_datapoints_x + num_datapoints_y))
            # x_idx = shuffled_idx[:num_datapoints_x]
            # y_idx = shuffled_idx[num_datapoints_x:]
            # X = all_data[:, x_idx]
            # Y = all_data[:, y_idx]

            # # Run test
            # test_stat, curr_reject = cai_test(X, Y)
            # decisions_shuffled.append(curr_reject)
            # test_stats_shuffled.append(test_stat)

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
            # Run test
            test_stat, curr_reject = cai_test(X, Y, verbose=True)
            decisions_shuffled.append(curr_reject)
            test_stats_shuffled.append(test_stat)




        power = np.sum(decisions_experiment) / len(decisions_experiment)
        size = np.sum(decisions_shuffled) / len(decisions_experiment)
        print("Power: {}".format(power))
        print("Size: {}".format(size))
                

        # bfs_control = np.array(bfs_control)[~np.isnan(bfs_control)]
        test_stats_experiment = list(np.array(test_stats_experiment)[~np.isnan(test_stats_experiment)])
        test_stats_shuffled = list(np.array(test_stats_shuffled)[~np.isnan(test_stats_shuffled)])
        tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(y_true=np.concatenate([np.zeros(len(test_stats_shuffled)), np.ones(len(test_stats_experiment))]), y_score=np.concatenate([test_stats_shuffled, test_stats_experiment]))

        auc = roc_auc_score(y_true=np.concatenate([np.zeros(len(test_stats_shuffled)), np.ones(len(test_stats_experiment))]), y_score=np.concatenate([test_stats_shuffled, test_stats_experiment]))

        np.save("./out/cai/test_stats_experiment_p{}.npy".format(data_dim), test_stats_experiment)
        np.save("./out/cai/test_stats_shuffled_p{}.npy".format(data_dim), test_stats_shuffled)

        plt.figure(figsize=(7, 5))
        plt.plot(tpr_shuffled, fpr_shuffled, label="Experiment vs. shuffled")
        plt.plot([0, 1], [0, 1], '--', color='black')
        plt.title("AUC={}".format(round(auc, 2)))
        plt.legend()
        plt.xlabel("Power")
        plt.ylabel("Size")
        plt.tight_layout()
        plt.savefig("./out/cai_roc_curve_p{}.png".format(data_dim))
        # plt.close()
        # plt.show()
    import ipdb; ipdb.set_trace()

