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

import sys
sys.path.append("../models")

from clvm_tfp_poisson import fit_model as fit_clvm
from clvm_tfp_poisson import clvm

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

tf.enable_v2_behavior()

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    num_datapoints_x = 1000
    num_datapoints_y = 1000
    data_dim = 500
    latent_dim_shared = 5
    latent_dim_target = 5

    gene_set_size = 25
    gene_set_indices = [np.arange(ii, ii+gene_set_size) for ii in np.arange(0, data_dim, gene_set_size)]
    stimulated_gene_set_idx = gene_set_indices[0]
    unstimulated_gene_set_idx = np.setdiff1d(np.arange(data_dim), stimulated_gene_set_idx)


    NUM_REPEATS = 5
    all_elbos = []

    for _ in range(NUM_REPEATS):


        # Generate data
        # a, b = 1, 1

        # actual_s = np.random.gamma(a, 1/b, size=(data_dim, latent_dim_shared))
        
        # actual_zx = np.random.gamma(a, 1/b, size=(latent_dim_shared, num_datapoints_x))
        # actual_zy = np.random.gamma(a, 1/b, size=(latent_dim_shared, num_datapoints_y))
        # actual_ty = np.random.gamma(a, 1/b, size=(latent_dim_target, num_datapoints_y))

        # # Stimulate one gene set
        # actual_w = np.zeros((data_dim, latent_dim_target))
        # # actual_w = np.random.gamma(a, 1/b, size=(data_dim, latent_dim_target))
        # actual_w[stimulated_gene_set_idx, :] = np.random.gamma(1, 5, size=(stimulated_gene_set_idx.shape[0], latent_dim_target))

        # X = np.random.poisson(actual_s @ actual_zx)
        # Y = np.random.poisson(actual_s @ actual_zy + actual_w @ actual_ty)

        # import ipdb; ipdb.set_trace()

        # Simulate data from null model
        concrete_clvm_model = functools.partial(clvm,
                                                data_dim=data_dim,
                                                latent_dim_shared=latent_dim_shared,
                                                latent_dim_target=latent_dim_target,
                                                num_datapoints_x=num_datapoints_x,
                                                num_datapoints_y=num_datapoints_y,
                                                counts_per_cell_X=1,
                                                counts_per_cell_Y=1,
                                                is_H0=False,
                                                num_test_genes=unstimulated_gene_set_idx.shape[0])

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

        deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
        
        X, Y = X_sampled.numpy(), Y_sampled.numpy()


        gene_set_bfs = []
        for ii, in_set_idx in enumerate(gene_set_indices):

            num_test_genes = in_set_idx.shape[0]
            out_set_idx = np.setdiff1d(np.arange(data_dim), in_set_idx)


            

            curr_X = np.concatenate([X[out_set_idx, :], X[in_set_idx, :]])
            curr_Y = np.concatenate([Y[out_set_idx, :], Y[in_set_idx, :]])
            

            H1_results = fit_clvm(curr_X, curr_Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False, num_test_genes=0)
            H0_results = fit_clvm(curr_X, curr_Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False, num_test_genes=in_set_idx.shape[0])

            H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (X.shape[1] + Y.shape[1])
            H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (X.shape[1] + Y.shape[1])

            

            curr_treatment_bf = H1_elbo - H0_elbo
            gene_set_bfs.append(curr_treatment_bf)

            print("BF for gene set {}: {}".format(ii, curr_treatment_bf))

            n_gene_sets = len(gene_set_bfs)
            gene_set_names = ["Set {}".format(x + 1) for x in range(n_gene_sets)]

            import matplotlib
            font = {'size'   : 18}
            matplotlib.rc('font', **font)

            plt.figure(figsize=(14, 6))
            plt.bar(np.arange(n_gene_sets), gene_set_bfs)
            plt.xticks(np.arange(n_gene_sets), gene_set_names, rotation=90)
            plt.ylabel("log(BF)")
            plt.tight_layout()
            plt.savefig("./out/bfs_targeted_gene_sets.png")
            plt.close()

            plt.figure(figsize=(7, 6))
            plt.axvline(gene_set_bfs[0], linestyle="--", color="black")
            plt.hist(gene_set_bfs)
            plt.xlabel("log(BF)")
            plt.ylabel("Count")
            plt.savefig("./out/bfs_targeted_gene_sets_hist.png")
            plt.close()

        all_elbos.append(gene_set_bfs)
        # import ipdb; ipdb.set_trace()
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=pd.melt(pd.DataFrame(all_elbos, columns=gene_set_names)), x="variable", y="value")
        # plt.xticks(np.arange(n_gene_sets), gene_set_names, rotation=90)
        plt.ylabel("log(BF)")
        plt.xlabel("")
        plt.xticks(rotation=90)
        plt.title("Targeted Bayes factors")
        plt.tight_layout()
        plt.savefig("./out/bfs_targeted_gene_sets_boxplot.png")
        plt.close()

            # import ipdb; ipdb.set_trace()

            # np.exp(H1_results['qw_mean'].numpy() + H1_results['qw_stddv'].numpy()**2 / 2)

            # ------- Specify model ---------

            # concrete_clvm_model = functools.partial(clvm,
            #                                         data_dim=data_dim,
            #                                         latent_dim_shared=latent_dim_shared,
            #                                         latent_dim_target=latent_dim_target,
            #                                         num_datapoints_x=num_datapoints_x,
            #                                         num_datapoints_y=num_datapoints_y,
            #                                         counts_per_cell_X=1,
            #                                         counts_per_cell_Y=1,
            #                                         is_H0=False,
            #                                         num_test_genes=0)

            # model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

            # deltax, deltay, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
            
            # X, Y = X_sampled.numpy(), Y_sampled.numpy()

            # # Move test genes to the end of the matrix
            # nontest_idx = np.setdiff1d(np.arange(data_dim), in_set_idx)
            # X = np.concatenate([X[nontest_idx, :], X[in_set_idx, :]])
            # Y = np.concatenate([Y[nontest_idx, :], Y[in_set_idx, :]])


            

            # ## Run H0 and H1 models on data

            # H1_results = fit_clvm(
            #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=False, is_H0=False, num_test_genes=0)
            
            # H0_results = fit_clvm(
            #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=False, is_H0=False, num_test_genes=num_test_genes)

            

            # H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)

            # H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)

            # curr_bf = H1_elbo - H0_elbo
            # print("BF treatment: {}".format(curr_bf))
            # bfs_experiment.append(curr_bf)

            # ## Simulate data from null model

            # # Y = np.random.poisson(actual_s @ actual_zy)
            # concrete_clvm_model = functools.partial(clvm,
            #                                         data_dim=data_dim,
            #                                         latent_dim_shared=latent_dim_shared,
            #                                         latent_dim_target=latent_dim_target,
            #                                         num_datapoints_x=num_datapoints_x,
            #                                         num_datapoints_y=num_datapoints_y,
            #                                         counts_per_cell_X=1,
            #                                         counts_per_cell_Y=1,
            #                                         is_H0=True,
            #                                         num_test_genes=num_test_genes)

            # model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

            # deltax, deltay, s, zx, zy, X_sampled, Y_sampled = model.sample()
            
            # X, Y = X_sampled.numpy(), Y_sampled.numpy()
            # X = np.concatenate([X[nontest_idx, :], X[w_test_idx, :]])
            # Y = np.concatenate([Y[nontest_idx, :], Y[w_test_idx, :]])

            # # a, b = 1, 1

            # # actual_s = np.random.gamma(a, 1/b, size=(data_dim, latent_dim_shared))
            # # actual_w = np.random.gamma(a, 1/b, size=(data_dim, latent_dim_target))
            # # # actual_w[w_zero_idx, 0] = np.random.gamma(40, 1/5, size=len(w_zero_idx))

            # # actual_zx = np.random.gamma(a, 1/b, size=(latent_dim_shared, num_datapoints_x))
            # # actual_zy = np.random.gamma(a, 1/b, size=(latent_dim_shared, num_datapoints_y))
            # # actual_ty = np.random.gamma(a, 1/b, size=(latent_dim_target, num_datapoints_y))

            # # X = np.random.poisson(actual_s @ actual_zx)

            # # Y = np.random.poisson(actual_s @ actual_zy + actual_w @ actual_ty)

            # ## Run H0 and H1 models on data

            # H1_results = fit_clvm(
            #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=False, is_H0=False, num_test_genes=0)
            # H0_results = fit_clvm(
            #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=False, is_H0=False, num_test_genes=num_test_genes)

            # # plt.plot(H0_results['loss_trace'])
            # # plt.show()
            # # plt.scatter(np.exp(H0_results['qzx_mean'].numpy()[0, :]), zx.numpy()[0, :])
            # # plt.show()

            # # plt.scatter(np.exp(H0_results['qs_mean'].numpy()[:, 0]), s.numpy()[:, 0])
            # # plt.show()

            # # import ipdb; ipdb.set_trace()

            # H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)
            # H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y)

            # curr_bf = H1_elbo - H0_elbo
            # print("BF control: {}".format(curr_bf))
            # bfs_control.append(curr_bf)
            # # import ipdb; ipdb.set_trace()

            # plt.figure(figsize=(7, 5))
            # sns.boxplot(np.arange(2), [bfs_control, bfs_experiment])
            # plt.title("Bayes factors, simulated data")
            # plt.xticks(np.arange(2), labels=[
            #            "Data simulated from\nnull model", "Data simulated from\nalternative model"])
            # plt.ylabel("log(BF)")
            # plt.savefig("./out/evidences_simulated_data_targeted.png")
            # plt.close()

