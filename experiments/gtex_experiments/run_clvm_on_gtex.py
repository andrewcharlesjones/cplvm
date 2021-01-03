import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import socket

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import sys
sys.path.append("../models")

from clvm_tfp_poisson import fit_model as fit_clvm

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True



if __name__ == "__main__":

    latent_dim_shared = 5
    latent_dim_target = 5

    # X_fname = "/Users/andrewjones/Documents/beehive/differential_covariance/clvm/gtex_experiments/data/gtex_lung_expression_noventilator.csv"
    # Y_fname = "/Users/andrewjones/Documents/beehive/differential_covariance/clvm/gtex_experiments/data/gtex_lung_expression_ventilator.csv"

    X_fname = "/Users/andrewjones/Documents/beehive/differential_covariance/clvm/gtex_experiments/data/gtex_lung_expression_noheartdisease.csv"
    Y_fname = "/Users/andrewjones/Documents/beehive/differential_covariance/clvm/gtex_experiments/data/gtex_lung_expression_heartdisease.csv"

    # Read in data
    X = pd.read_csv(X_fname, index_col=0)
    Y = pd.read_csv(Y_fname, index_col=0)

    model_dict = fit_clvm(
        X.values.T, Y.values.T, latent_dim_shared, latent_dim_target, compute_size_factors=True)

    # Mean of log-normal (take mean of posterior)
    S_estimated = np.exp(model_dict['qs_mean'].numpy() + model_dict['qs_stddv'].numpy()**2 / 2)
    S_estimated = pd.DataFrame(S_estimated, index=X.columns.values)

    # Mean of log-normal (take mean of posterior)
    W_estimated = np.exp(model_dict['qw_mean'].numpy() + model_dict['qw_stddv'].numpy()**2 / 2)
    W_estimated = pd.DataFrame(W_estimated, index=X.columns.values)

    zx_estimated = np.exp(model_dict['qzx_mean'].numpy() + model_dict['qzx_stddv'].numpy()**2 / 2)
    zy_estimated = np.exp(model_dict['qzy_mean'].numpy() + model_dict['qzy_stddv'].numpy()**2 / 2)
    ty_estimated = np.exp(model_dict['qty_mean'].numpy() + model_dict['qty_stddv'].numpy()**2 / 2)

    zy_df = pd.DataFrame(zy_estimated.T)
    ty_df = pd.DataFrame(ty_estimated.T)

    ## Save S and W matrices
    S_estimated.to_csv("./out/gtex_heart_S.csv", header=None)
    W_estimated.to_csv("./out/gtex_heart_W.csv", header=None)


    # import ipdb; ipdb.set_trace()


