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

    NUM_REPEATS = 20
    NUM_SHUFFLE_REPEATS = 2
    all_elbos = []

    decisions_experiment = []
    decisions_shuffled = []
    test_stats_experiment = []
    test_stats_shuffled = []

    alpha = 0.05


    for jj in range(NUM_REPEATS):


        #################################
        ######### H1 is true ############
        #################################
        
        # ------- generate data ---------

        # Generate data
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
            num_test_genes=data_dim - stimulated_gene_set_idx.shape[0],
            offset_term=True,
        )

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

        deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
        # sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()

        X, Y = X_sampled.numpy(), Y_sampled.numpy()

        # X = np.log(X + 1)
        # Y = np.log(Y + 1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)

        X_stimulated_gene_set = X.T[:, stimulated_gene_set_idx]
        Y_stimulated_gene_set = Y.T[:, stimulated_gene_set_idx]

        pd.DataFrame(X_stimulated_gene_set).to_csv("./tmp/X.csv")
        pd.DataFrame(Y_stimulated_gene_set).to_csv("./tmp/Y.csv")

        ##### Run test procedure #####

        os.system("Rscript li2012_test.R")
        curr_output = pd.read_csv("./tmp/curr_li2012_output.csv", index_col=0)
        # import ipdb; ipdb.set_trace()
        test_stats_experiment.append(curr_output.values[0, 0])
        decisions_experiment.append(curr_output.values[0, 1] < alpha)


        #################################
        ######### H0 is true ############
        #################################

        for ii, in_set_idx in enumerate(gene_set_indices[1:]):

            X_unstimulated_gene_set = X.T[:, in_set_idx]
            Y_unstimulated_gene_set = Y.T[:, in_set_idx]

            pd.DataFrame(X_unstimulated_gene_set).to_csv("./tmp/X.csv")
            pd.DataFrame(Y_unstimulated_gene_set).to_csv("./tmp/Y.csv")

            ##### Run test procedure #####

            os.system("Rscript li2012_test.R")
            curr_output = pd.read_csv("./tmp/curr_li2012_output.csv", index_col=0)

            test_stats_shuffled.append(curr_output.values[0, 0])
            decisions_shuffled.append(curr_output.values[0, 1] < alpha)

            for ff in os.listdir("./tmp"):
                os.remove(os.path.join("tmp", ff))

    plt.hist(test_stats_experiment, 30, label="Experiment", alpha=0.3)
    plt.hist(test_stats_shuffled, 30, label="Null", alpha=0.3)
    plt.legend()
    plt.show()
    np.save(
        "../out/li2012/test_stats_experiment.npy",
        test_stats_experiment,
    )

    np.save(
        "../out/li2012/test_stats_shuffled.npy",
        test_stats_shuffled,
    )
        
            


