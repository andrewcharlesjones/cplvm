import matplotlib
from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox
from os.path import join as pjoin

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

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["lines.markersize"] = 10

tf.enable_v2_behavior()

warnings.filterwarnings("ignore")


if __name__ == "__main__":

	latent_dim_shared = 2
	latent_dim_foreground = 2

	## Load data simulated from Splatter
	data_dir = "/Users/andrewjones/Documents/beehive/cplvm/data/splatter/two_clusters"
	bg_path = pjoin(data_dir, "bg.csv")
	fg_path = pjoin(data_dir, "fg.csv")
	fg_labels_path = pjoin(data_dir, "fg_labels.csv")
	bg_data = pd.read_csv(bg_path, index_col=0)
	fg_data = pd.read_csv(fg_path, index_col=0)
	fg_labels = pd.read_csv(fg_labels_path, index_col=0).iloc[:, 0].values
	X = bg_data.values.T
	Y = fg_data.values.T	

	## Fit CPLVM to data
	cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

	approx_model = CPLVMLogNormalApprox(
		X, Y, latent_dim_shared, latent_dim_foreground, offset_term=False
	)
	model_fit = cplvm._fit_model_vi(
		X, Y, approx_model, compute_size_factors=True, is_H0=False, offset_term=False
	)

	## Extract latent variables
	zx_mean = model_fit['approximate_model'].qzx_mean.numpy().T
	zx_stddv = model_fit['approximate_model'].qzx_stddv.numpy().T
	zy_mean = model_fit['approximate_model'].qzy_mean.numpy().T
	zy_stddv = model_fit['approximate_model'].qzy_stddv.numpy().T
	ty_mean = model_fit['approximate_model'].qty_mean.numpy().T
	ty_stddv = model_fit['approximate_model'].qty_stddv.numpy().T

	zx = np.exp(zx_mean + zx_stddv**2 / 2)
	zy = np.exp(zy_mean + zy_stddv**2 / 2)
	ty = np.exp(ty_mean + ty_stddv**2 / 2)
	

	## Compute silhouetter score

	## Plot latent variables colored by true cluster membership
	plt.figure(figsize=(14, 7))

	plt.subplot(121)
	zy_df = pd.DataFrame(zy, columns=["z1", "z2"])
	fg_labels_strings = np.array(["FG (unresponsive)" if fg_labels[i] == 0 else "FG (responsive)" for i in range(len(fg_labels))])
	zy_df["label"] = fg_labels_strings

	zx_df = pd.DataFrame(zx, columns=["z1", "z2"])
	bg_labels_strings = np.array(["BG" for _ in range(zx_df.shape[0])])
	zx_df["label"] = bg_labels_strings

	z_df = pd.concat([zx_df, zy_df], axis=0)

	sns.scatterplot(data=z_df, x="z1", y="z2", style="label", markers=['*', 'o', 'v'], color="black")
	plt.xlabel(r"$z_1$")
	plt.ylabel(r"$z_2$")
	ax = plt.gca()
	ax.get_legend().set_title("")
	plt.title("Shared latent space")
	plt.tight_layout()

	plt.subplot(122)
	ty_df = pd.DataFrame(ty, columns=["ty1", "ty2"])
	fg_labels_strings = np.array(["FG (unresponsive)" if fg_labels[i] == 0 else "FG (responsive)" for i in range(len(fg_labels))])
	ty_df["label"] = fg_labels_strings
	sns.scatterplot(data=ty_df, x="ty1", y="ty2", style="label", markers=['o', 'v'], color="black")
	plt.xlabel(r"$t_1$")
	plt.ylabel(r"$t_2$")
	ax = plt.gca()
	ax.get_legend().set_title("")
	plt.title("Foreground-specific latent space")
	plt.tight_layout()
	plt.savefig("../out/splatter/subset_response_splatter.png")
	plt.show()
	import ipdb; ipdb.set_trace()

