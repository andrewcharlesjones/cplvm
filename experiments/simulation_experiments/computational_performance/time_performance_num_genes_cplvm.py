from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox
from pcpca import CPCA, PCPCA

# import sys
# sys.path.append("../../../cplvm/models/")
from clvm_gaussian import fit_model as fit_clvm_gaussian

import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd

import matplotlib
import time

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    n_genes_list = [10, 50, 100, 500, 1000]
    # n_genes_list = [10, 50]
    num_datapoints_x, num_datapoints_y = 200, 200
    NUM_REPEATS = 10
    latent_dim_shared, latent_dim_foreground = 3, 3

    times_cplvm = np.empty((NUM_REPEATS, len(n_genes_list)))
    times_clvm_gaussian = np.empty((NUM_REPEATS, len(n_genes_list)))
    times_pcpca = np.empty((NUM_REPEATS, len(n_genes_list)))
    times_cpca = np.empty((NUM_REPEATS, len(n_genes_list)))

    for ii, n_genes in enumerate(n_genes_list):

        for jj in range(NUM_REPEATS):

            # ------- generate data ---------

            cplvm_for_data = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )

            concrete_cplvm_model = functools.partial(
                cplvm_for_data.model,
                data_dim=n_genes,
                num_datapoints_x=num_datapoints_x,
                num_datapoints_y=num_datapoints_y,
                counts_per_cell_X=1,
                counts_per_cell_Y=1,
                is_H0=False,
            )

            model = tfd.JointDistributionCoroutineAutoBatched(concrete_cplvm_model)
            deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
            X, Y = X_sampled.numpy(), Y_sampled.numpy()

            ##### CPLVM #####

            t0 = time.time()

            cplvm = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )
            approx_model = CPLVMLogNormalApprox(
                X, Y, latent_dim_shared, latent_dim_foreground
            )
            model_fit = cplvm._fit_model_vi(X, Y, approx_model, is_H0=False)

            t1 = time.time()

            curr_time = t1 - t0

            times_cplvm[jj, ii] = curr_time

            ##### CLVM (gaussian model) #####
            t0 = time.time()
            fit_clvm_gaussian(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                compute_size_factors=False,
                is_H0=False,
            )
            t1 = time.time()
            curr_time = t1 - t0
            times_clvm_gaussian[jj, ii] = curr_time

            ##### PCPCA #####
            t0 = time.time()
            pcpca = PCPCA(gamma=0.7, n_components=latent_dim_foreground)
            pcpca.fit(X, Y)
            pcpca.transform(X, Y)
            t1 = time.time()
            curr_time = t1 - t0
            times_pcpca[jj, ii] = curr_time

            ##### CPCA #####
            t0 = time.time()
            cpca = CPCA(gamma=0.7, n_components=latent_dim_foreground)
            cpca.fit(X, Y)
            cpca.transform(X, Y)
            t1 = time.time()
            curr_time = t1 - t0
            times_cpca[jj, ii] = curr_time

    times_cplvm_df = pd.DataFrame(times_cplvm, columns=n_genes_list)
    times_cplvm_df_melted = pd.melt(times_cplvm_df)
    times_cplvm_df_melted["model"] = [
        "cplvm" for _ in range(NUM_REPEATS * len(n_genes_list))
    ]
    # times_cplvm_df_melted.to_csv("../out/time_performance_num_genes_cplvm.csv")

    times_clvm_df = pd.DataFrame(times_clvm_gaussian, columns=n_genes_list)
    times_clvm_df_melted = pd.melt(times_clvm_df)
    times_clvm_df_melted["model"] = [
        "clvm" for _ in range(NUM_REPEATS * len(n_genes_list))
    ]
    # times_clvm_df_melted.to_csv("../out/time_performance_num_genes_clvm.csv")

    times_pcpca_df = pd.DataFrame(times_pcpca, columns=n_genes_list)
    times_pcpca_df_melted = pd.melt(times_pcpca_df)
    times_pcpca_df_melted["model"] = [
        "pcpca" for _ in range(NUM_REPEATS * len(n_genes_list))
    ]
    # times_pcpca_df_melted.to_csv("../out/time_performance_num_genes_pcpca.csv")

    times_cpca_df = pd.DataFrame(times_cpca, columns=n_genes_list)
    times_cpca_df_melted = pd.melt(times_cpca_df)
    times_cpca_df_melted["model"] = [
        "cpca" for _ in range(NUM_REPEATS * len(n_genes_list))
    ]
    # times_cpca_df_melted.to_csv("../out/time_performance_num_genes_cpca.csv")

    times_df_melted = pd.concat(
        [
            times_cplvm_df_melted,
            times_clvm_df_melted,
            times_pcpca_df_melted,
            times_cpca_df_melted,
        ],
        axis=0,
    )
    times_df_melted.to_csv("../out/time_performance_num_genes.csv")

    # plt.figure(figsize=(7, 7))
    # sns.lineplot(data=times_df_melted, x="variable", y="value", ci=95, err_style="bars", color="black")
    # plt.xlabel("Number of genes")
    # plt.ylabel("Time (s)")
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.savefig("../out/time_performance_num_genes_cplvm.png")
    # plt.show()
    # import ipdb; ipdb.set_trace()
