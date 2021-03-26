import matplotlib
from cplvm import CPLVM
from cplvm import CGLVM
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


def test_initialization():

    num_datapoints_x = 1000
    num_datapoints_y = 1000
    data_dim = 200
    latent_dim_shared = 3
    latent_dim_foreground = 3

    actual_a, actual_b = 3, 3

    NUM_REPEATS = 5
    bfs_experiment = []
    bfs_control = []
    bfs_shuffled = []

    bfs_experiment_cglvm = []
    bfs_control_cglvm = []
    bfs_shuffled_cglvm = []
    for ii in range(NUM_REPEATS):

        # ------- generate data ---------

        cplvm = CPLVM(
            k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
        )

        cglvm = CGLVM(
            k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
        )
