import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import poisson
from scipy.special import logsumexp

from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

# import sys
# import socket

# if socket.gethostname() == "andyjones":
#     sys.path.append("../../models")
# else:
#     sys.path.append("../models")

# from clvm_tfp_poisson import fit_model as fit_clvm
# from clvm_tfp_poisson import clvm

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    num_datapoints_x = 1000
    num_datapoints_y = 1000
    data_dim = 500
    latent_dim_shared = 5
    latent_dim_foreground = 5

    gene_set_size = 25
    gene_set_indices = [
        np.arange(ii, ii + gene_set_size)
        for ii in np.arange(0, data_dim // 2, gene_set_size)
    ]
    stimulated_gene_set_idx = gene_set_indices[0]
    unstimulated_gene_set_idx = np.setdiff1d(
        np.arange(data_dim), stimulated_gene_set_idx
    )
    # import ipdb; ipdb.set_trace()

    NUM_REPEATS = 5
    NUM_SHUFFLE_REPEATS = 2
    all_elbos = []

    for _ in range(NUM_REPEATS):

        # Generate data
        cplvm_for_data = CPLVM(
            k_shared=latent_dim_shared,
            k_foreground=latent_dim_foreground,
            compute_size_factors=True,
        )

        concrete_clvm_model = functools.partial(
            cplvm_for_data.model,
            data_dim=data_dim,
            num_datapoints_x=num_datapoints_x,
            num_datapoints_y=num_datapoints_y,
            counts_per_cell_X=1.0,
            counts_per_cell_Y=1.0,
            is_H0=False,
            num_test_genes=data_dim - stimulated_gene_set_idx.shape[0],
            offset_term=False,
        )

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

        # deltax, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
        sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()

        X, Y = X_sampled.numpy(), Y_sampled.numpy()

        gene_set_bfs = []
        for ii, in_set_idx in enumerate(gene_set_indices):

            num_test_genes = in_set_idx.shape[0]
            out_set_idx = np.setdiff1d(np.arange(data_dim), in_set_idx)

            curr_X = np.concatenate([X[out_set_idx, :], X[in_set_idx, :]])
            curr_Y = np.concatenate([Y[out_set_idx, :], Y[in_set_idx, :]])

            cplvm = CPLVM(
                k_shared=latent_dim_shared,
                k_foreground=latent_dim_foreground,
                compute_size_factors=True,
                is_H0=False,
                offset_term=True,
            )
            approx_model = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                is_H0=False,
                num_test_genes=0,
                offset_term=True,
                compute_size_factors=True,
            )

            H1_results = cplvm.fit_model_vi(
                X, Y, approx_model, num_test_genes=0
            )
            # import ipdb

            # ipdb.set_trace()

            cplvm = CPLVM(
                k_shared=latent_dim_shared,
                k_foreground=latent_dim_foreground,
                compute_size_factors=True,
                is_H0=False,
                offset_term=True,
            )
            approx_model = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                is_H0=False,
                num_test_genes=in_set_idx.shape[0],
                offset_term=True,
                compute_size_factors=True,
            )

            H0_results = cplvm.fit_model_vi(
                X,
                Y,
                approx_model,
                num_test_genes=in_set_idx.shape[0],
            )
            # import ipdb

            # ipdb.set_trace()

            H1_elbo = (
                -1 * H1_results["loss_trace"][-1].numpy() / (X.shape[1] + Y.shape[1])
            )
            H0_elbo = (
                -1 * H0_results["loss_trace"][-1].numpy() / (X.shape[1] + Y.shape[1])
            )

            curr_treatment_bf = H1_elbo - H0_elbo
            gene_set_bfs.append(curr_treatment_bf)

            print("BF for gene set {}: {}".format(ii, curr_treatment_bf))

        n_gene_sets = len(gene_set_bfs)
        gene_set_bfs.extend(shuffled_bfs)
        all_elbos.append(gene_set_bfs)

        gene_set_names = ["Set {}".format(x + 1) for x in range(n_gene_sets)]
        gene_set_names.extend(["Shuffled null" for _ in range(NUM_SHUFFLE_REPEATS)])

        import matplotlib

        font = {"size": 18}
        matplotlib.rc("font", **font)

        plt.figure(figsize=(14, 6))
        sns.boxplot(
            data=pd.melt(pd.DataFrame(all_elbos, columns=gene_set_names)),
            x="variable",
            y="value",
        )
        plt.ylabel("log(EBF)")
        plt.xlabel("")
        plt.xticks(rotation=90)
        plt.title("Targeted Bayes factors")
        plt.tight_layout()
        plt.savefig("../out/bfs_targeted_gene_sets_boxplot.png")
        plt.close()

        np.save("../out/bfs_targeted.npy", np.array(all_elbos))
