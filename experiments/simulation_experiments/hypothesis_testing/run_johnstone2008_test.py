from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox
from pcpca import CPCA, PCPCA

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
import subprocess
import os

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    p_list = [10, 100, 1000]
    num_datapoints_x, num_datapoints_y = 100, 100
    NUM_REPEATS = 20
    latent_dim_shared, latent_dim_foreground = 3, 3

    decisions_experiment = []
    decisions_shuffled = []
    test_stats_experiment = []
    test_stats_shuffled = []

    alpha = 0.05

    for ii, n_genes in enumerate(p_list):

        for jj in range(NUM_REPEATS):

            #################################
            ######### H1 is true ############
            #################################

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
            X = np.log(X + 1)
            Y = np.log(Y + 1)
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
            Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)

            pd.DataFrame(X.T).to_csv("./tmp/X.csv")
            pd.DataFrame(Y.T).to_csv("./tmp/Y.csv")

            ##### Run test procedure #####

            os.system("Rscript johnston2008_test.R")
            curr_output = pd.read_csv("./tmp/curr_johnston_output.csv", index_col=0)
            test_stats_experiment.append(curr_output.values[0, 0])
            decisions_experiment.append(curr_output.values[0, 1] < alpha)

            #################################
            ######### H0 is true ############
            #################################

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
                is_H0=True,
            )

            model = tfd.JointDistributionCoroutineAutoBatched(concrete_cplvm_model)
            deltax, sf_x, sf_y, s, zx, zy, X_sampled, Y_sampled = model.sample()
            X, Y = X_sampled.numpy(), Y_sampled.numpy()
            X = np.log(X + 1)
            Y = np.log(Y + 1)
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
            Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)

            pd.DataFrame(X.T).to_csv("./tmp/X.csv")
            pd.DataFrame(Y.T).to_csv("./tmp/Y.csv")

            ##### Run test procedure #####

            os.system("Rscript johnston2008_test.R")
            curr_output = pd.read_csv("./tmp/curr_johnston_output.csv", index_col=0)
            test_stats_shuffled.append(curr_output.values[0, 0])
            decisions_shuffled.append(curr_output.values[0, 1] < alpha)

            for ff in os.listdir("./tmp"):
                os.remove(os.path.join("tmp", ff))

        np.save(
            "../out/johnstone/test_stats_experiment_p{}.npy".format(n_genes),
            test_stats_experiment,
        )

        np.save(
            "../out/johnstone/test_stats_shuffled_p{}.npy".format(n_genes),
            test_stats_shuffled,
        )

    # import ipdb; ipdb.set_trace()
