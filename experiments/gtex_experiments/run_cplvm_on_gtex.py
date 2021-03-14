import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from os.path import join as pjoin

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from cplvm import CPLVM

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


if __name__ == "__main__":

    latent_dim_shared = 5
    latent_dim_target = 5

    data_dir = "../../data/gtex/data"
    X_fname = pjoin(data_dir, "gtex_expression_artery_heartdisease.csv")
    Y_fname = pjoin(data_dir, "gtex_expression_artery_noheartdisease.csv")

    # Read in data
    X = pd.read_csv(X_fname, index_col=0)
    Y = pd.read_csv(Y_fname, index_col=0)

    cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

    model_dict = cplvm.fit_model_vi(X, Y, compute_size_factors=True, is_H0=False)

    # Mean of log-normal (take mean of posterior)
    S_estimated = np.exp(
        model_dict["qs_mean"].numpy() + model_dict["qs_stddv"].numpy() ** 2 / 2
    )
    S_estimated = pd.DataFrame(S_estimated, index=X.columns.values)

    # Mean of log-normal (take mean of posterior)
    W_estimated = np.exp(
        model_dict["qw_mean"].numpy() + model_dict["qw_stddv"].numpy() ** 2 / 2
    )
    W_estimated = pd.DataFrame(W_estimated, index=X.columns.values)

    zx_estimated = np.exp(
        model_dict["qzx_mean"].numpy() + model_dict["qzx_stddv"].numpy() ** 2 / 2
    )
    zy_estimated = np.exp(
        model_dict["qzy_mean"].numpy() + model_dict["qzy_stddv"].numpy() ** 2 / 2
    )
    ty_estimated = np.exp(
        model_dict["qty_mean"].numpy() + model_dict["qty_stddv"].numpy() ** 2 / 2
    )

    zy_df = pd.DataFrame(zy_estimated.T)
    ty_df = pd.DataFrame(ty_estimated.T)

    ## Save S and W matrices
    S_estimated.to_csv("./out/gtex_heart_S.csv", header=None)
    W_estimated.to_csv("./out/gtex_heart_W.csv", header=None)
