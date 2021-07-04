import matplotlib
from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox

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
import time

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    n_samples_per_condition_list = [10, 100, 1000]
    n_genes = 200
    NUM_REPEATS = 2
    latent_dim_shared, latent_dim_foreground = 3, 3

    times = np.empty((NUM_REPEATS, len(n_samples_per_condition_list)))

    for ii, n_samples in enumerate(n_samples_per_condition_list):

        for jj in range(NUM_REPEATS):

            num_datapoints_x, num_datapoints_y = n_samples, n_samples

            

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

            # ------- fit model ---------

            t0 = time.time()

            cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)
            approx_model = CPLVMLogNormalApprox(
                X, Y, latent_dim_shared, latent_dim_foreground
            )
            model_fit = cplvm._fit_model_vi(
                X, Y, approx_model, compute_size_factors=True, is_H0=False
            )

            t1 = time.time()

            curr_time = t1 - t0

            times[jj, ii] = curr_time


    times_df = pd.DataFrame(times, columns=n_samples_per_condition_list)
    times_df_melted = pd.melt(times_df)

    plt.figure(figsize=(7, 7))
    sns.lineplot(data=times_df_melted, x="variable", y="value", ci=95, err_style="bars")
    plt.xlabel("Number of samples\nin each condition")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig("../out/time_performance_num_samples_cplvm.png")
    plt.show()
    import ipdb; ipdb.set_trace()




        