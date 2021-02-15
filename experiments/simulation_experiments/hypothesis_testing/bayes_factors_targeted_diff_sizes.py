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

from cplvm import CPLVM

from clvm_tfp_poisson import fit_model as fit_clvm
from clvm_tfp_poisson import clvm

import matplotlib
font = {'size': 18}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

tf.enable_v2_behavior()

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    num_datapoints_x = 1000
    num_datapoints_y = 1000
    data_dim = 250
    latent_dim_shared = 5
    latent_dim_target = 5

    gene_set_sizes = [25, 20, 15, 10, 5, 1]


    NUM_REPEATS = 20
    all_bfs = []

    for ii, gene_set_size in enumerate(gene_set_sizes):

        gene_set_indices = [np.arange(ii, ii+gene_set_size) for ii in np.arange(0, data_dim//2, gene_set_size)]
        stimulated_gene_set_idx = gene_set_indices[0]
        unstimulated_gene_set_idx = np.setdiff1d(np.arange(data_dim), stimulated_gene_set_idx)

        curr_bfs = []

        for _ in range(NUM_REPEATS):

            # Generate data
            cplvm_for_data = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

            concrete_clvm_model = functools.partial(cplvm_for_data.model,
                                                    data_dim=data_dim,
                                                    num_datapoints_x=num_datapoints_x,
                                                    num_datapoints_y=num_datapoints_y,
                                                    counts_per_cell_X=1,
                                                    counts_per_cell_Y=1,
                                                    is_H0=False,
                                                    num_test_genes=data_dim-stimulated_gene_set_idx.shape[0])

            model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

            deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
            
            X, Y = X_sampled.numpy(), Y_sampled.numpy()


            in_set_idx = stimulated_gene_set_idx

            num_test_genes = in_set_idx.shape[0]
            out_set_idx = np.setdiff1d(np.arange(data_dim), in_set_idx)

            curr_X = np.concatenate([X[out_set_idx, :], X[in_set_idx, :]])
            curr_Y = np.concatenate([Y[out_set_idx, :], Y[in_set_idx, :]])

            cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

            H1_results = cplvm.fit_model_vi(curr_X, curr_Y, compute_size_factors=True, is_H0=False, num_test_genes=0)
            H0_results = cplvm.fit_model_vi(curr_X, curr_Y, compute_size_factors=True, is_H0=False, num_test_genes=in_set_idx.shape[0])

            H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (X.shape[1] + Y.shape[1])
            H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (X.shape[1] + Y.shape[1])

            

            curr_treatment_bf = H1_elbo - H0_elbo
            curr_bfs.append(curr_treatment_bf)

            print("BF: {}".format(curr_treatment_bf))

        

        all_bfs.append(curr_bfs)
        
        bf_df = pd.DataFrame(np.array(all_bfs).T, columns=gene_set_sizes[:ii+1])
        bf_df_melted = pd.melt(pd.DataFrame(bf_df))
        

        plt.figure(figsize=(14, 6))
        sns.boxplot(data=bf_df_melted, x="variable", y="value")
        plt.ylabel("log(EBF)")
        plt.xlabel("Perturbed gene set size")
        plt.title("Targeted Bayes factors")
        plt.tight_layout()
        plt.savefig("./out/bfs_gene_sets_vary_size.png")
        plt.show()
        plt.close()

        np.save("./out/bfs_targeted_vary_size.npy", np.array(np.array(all_bfs)))

