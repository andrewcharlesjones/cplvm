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

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


def test_loss_drop():

    num_datapoints_x = 1000
    num_datapoints_y = 1000
    data_dim = 200
    latent_dim_shared = 3
    latent_dim_foreground = 3

    actual_a, actual_b = 3, 3

    NUM_REPEATS = 1
    bfs_experiment = []
    bfs_control = []
    bfs_shuffled = []

    bfs_experiment_cglvm = []
    bfs_control_cglvm = []
    bfs_shuffled_cglvm = []
    for ii in range(NUM_REPEATS):

        # ------- generate data ---------

        cplvm_for_data = CPLVM(
            k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
        )

        concrete_cplvm_model = functools.partial(
            cplvm_for_data.model,
            data_dim=data_dim,
            num_datapoints_x=num_datapoints_x,
            num_datapoints_y=num_datapoints_y,
            counts_per_cell_X=1,
            counts_per_cell_Y=1,
            is_H0=False,
        )

        model = tfd.JointDistributionCoroutineAutoBatched(concrete_cplvm_model)

        deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()

        X, Y = X_sampled.numpy(), Y_sampled.numpy()

        ########## "Treatment" data ##########

        # Run H0 and H1 models on data
        cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

        approx_model = CPLVMLogNormalApprox(
            X, Y, latent_dim_shared, latent_dim_foreground
        )
        results = cplvm._fit_model_vi(
            X, Y, approx_model, compute_size_factors=True, is_H0=False
        )
        assert results['loss_trace'][0] > results['loss_trace'][-1]

