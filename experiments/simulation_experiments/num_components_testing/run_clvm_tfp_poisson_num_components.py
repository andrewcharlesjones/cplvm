import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import socket
from scipy import stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import sys

sys.path.append("../models")

from clvm_tfp_poisson import fit_model as fit_clvm
from clvm_tfp_poisson import clvm

if socket.gethostname() == "andyjones":
    DATA_DIR = "../../perturb_seq/data/targeted_genes"
else:
    DATA_DIR = "../data/targeted_genes/"

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


STDDEV_MULTIPLIER = 1e-4
NUM_VI_ITERS = 400
LEARNING_RATE_VI = 0.05


def mean_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    m, se = np.mean(data, axis=0), stats.sem(data, axis=0)
    width = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return width


if __name__ == "__main__":

    latent_dim_shared_true = 5
    latent_dim_target_true = 5
    num_datapoints_x = 5000
    num_datapoints_y = 5000
    data_dim = 200
    n_repeats = 10
    latent_dim_range = np.arange(1, 13)

    concrete_clvm_model = functools.partial(
        clvm,
        data_dim=data_dim,
        latent_dim_shared=latent_dim_shared_true,
        latent_dim_target=latent_dim_target_true,
        num_datapoints_x=num_datapoints_x,
        num_datapoints_y=num_datapoints_y,
        counts_per_cell_X=1,
        counts_per_cell_Y=1,
        is_H0=False,
        num_test_genes=0,
    )

    model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

    deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()

    X, Y = X_sampled.numpy(), Y_sampled.numpy()

    control_bfs = []
    treatment_bfs = []
    gene_names_so_far = []

    latent_dim_shared = latent_dim_shared_true

    elbos = np.empty((n_repeats, len(latent_dim_range)))

    for repeat_ii in range(n_repeats):
        for dim_ii, curr_latent_dim in enumerate(latent_dim_range):
            latent_dim_target = curr_latent_dim
            # latent_dim_shared = curr_latent_dim

            model_dict = fit_clvm(
                X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True
            )
            curr_elbo = -model_dict["loss_trace"][-1].numpy() / (
                num_datapoints_x + num_datapoints_y
            )

            print("ELBO: {}".format(round(curr_elbo, 2)))
            elbos[repeat_ii, dim_ii] = curr_elbo
            # import ipdb; ils pdb.set_trace()

            # model_dict = fit_clvm(
            #     X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=False, is_H0=True)

            # elbos_H0.append(-model_dict['loss_trace'][-1].numpy() / (num_datapoints_x + num_datapoints_y))

        plt.figure(figsize=(10, 6))
        plt.xlabel("Latent dimension")
        plt.ylabel("ELBO")
        plt.errorbar(
            latent_dim_range,
            np.mean(elbos[:repeat_ii, :], axis=0),
            yerr=mean_confidence_interval(elbos[:repeat_ii, :]),
        )
        plt.axvline(
            latent_dim_shared_true, linestyle="--", c="black", label="True dimension"
        )
        plt.legend(prop={"size": 20})
        plt.tight_layout()
        plt.savefig("./out/elbo_by_numlvs.png")

        # plt.show()
        plt.close()
