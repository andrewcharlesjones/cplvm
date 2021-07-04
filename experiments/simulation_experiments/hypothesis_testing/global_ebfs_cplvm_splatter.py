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
	n_bg = bg_data.shape[0]
	n_fg = fg_data.shape[0]

	num_repeats = 5
	bfs_experiment = []
	bfs_shuffled = []
	bfs_control = []

	for ii in range(num_repeats):


		######################################
		######### Original datasets ##########
		######################################
		X = bg_data.values.T
		Y = fg_data.values.T

		num_datapoints_x = X.shape[1]
		num_datapoints_y = Y.shape[1]

		cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)
		approx_model_H0 = CPLVMLogNormalApprox(
			X, Y, latent_dim_shared, latent_dim_foreground, offset_term=False, is_H0=True
		)
		approx_model_H1 = CPLVMLogNormalApprox(
			X, Y, latent_dim_shared, latent_dim_foreground, offset_term=False, is_H0=False
		)
		H0_results = cplvm._fit_model_vi(X, Y, approx_model_H0, compute_size_factors=True, offset_term=False, is_H0=True)
		H1_results = cplvm._fit_model_vi(X, Y, approx_model_H1, compute_size_factors=True, offset_term=False, is_H0=False)

		H1_elbo = (
			-1
			* H1_results["loss_trace"][-1].numpy()
			/ (num_datapoints_x + num_datapoints_y)
		)

		H0_elbo = (
			-1
			* H0_results["loss_trace"][-1].numpy()
			/ (num_datapoints_x + num_datapoints_y)
		)

		curr_bf = H1_elbo - H0_elbo
		print("BF treatment: {}".format(curr_bf))
		bfs_experiment.append(curr_bf)


		######################################
		######### Shuffled datasets ##########
		######################################
		shuffled_idx = np.random.permutation(np.arange(n_bg + n_fg))
		shuffled_idx_X = shuffled_idx[:n_bg]
		shuffled_idx_Y = shuffled_idx[n_bg:]
		all_data = np.concatenate([bg_data.values, fg_data.values], axis=0).T
		X = all_data[:, shuffled_idx_X]
		Y = all_data[:, shuffled_idx_Y]

		num_datapoints_x = X.shape[1]
		num_datapoints_y = Y.shape[1]

		cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)
		approx_model_H0 = CPLVMLogNormalApprox(
			X, Y, latent_dim_shared, latent_dim_foreground, offset_term=False, is_H0=True
		)
		approx_model_H1 = CPLVMLogNormalApprox(
			X, Y, latent_dim_shared, latent_dim_foreground, offset_term=False, is_H0=False
		)
		H0_results = cplvm._fit_model_vi(X, Y, approx_model_H0, compute_size_factors=True, offset_term=False, is_H0=True)
		H1_results = cplvm._fit_model_vi(X, Y, approx_model_H1, compute_size_factors=True, offset_term=False, is_H0=False)

		H1_elbo = (
			-1
			* H1_results["loss_trace"][-1].numpy()
			/ (num_datapoints_x + num_datapoints_y)
		)

		H0_elbo = (
			-1
			* H0_results["loss_trace"][-1].numpy()
			/ (num_datapoints_x + num_datapoints_y)
		)

		curr_bf = H1_elbo - H0_elbo
		print("BF shuffled: {}".format(curr_bf))
		bfs_shuffled.append(curr_bf)


		######################################
		######### Null datasets ##########
		######################################
		fg_unresponsive_data = fg_data.iloc[fg_labels == 0, :]
		all_data = np.concatenate([bg_data.values, fg_unresponsive_data.values], axis=0).T
		shuffled_idx = np.random.permutation(np.arange(all_data.shape[1]))
		shuffled_idx_X = shuffled_idx[:bg_data.shape[0]]
		shuffled_idx_Y = shuffled_idx[bg_data.shape[0]:]
		X = all_data[:, shuffled_idx_X]
		Y = all_data[:, shuffled_idx_Y]

		num_datapoints_x = X.shape[1]
		num_datapoints_y = Y.shape[1]

		cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)
		approx_model_H0 = CPLVMLogNormalApprox(
			X, Y, latent_dim_shared, latent_dim_foreground, offset_term=False, is_H0=True
		)
		approx_model_H1 = CPLVMLogNormalApprox(
			X, Y, latent_dim_shared, latent_dim_foreground, offset_term=False, is_H0=False
		)
		H0_results = cplvm._fit_model_vi(X, Y, approx_model_H0, compute_size_factors=True, offset_term=False, is_H0=True)
		H1_results = cplvm._fit_model_vi(X, Y, approx_model_H1, compute_size_factors=True, offset_term=False, is_H0=False)

		H1_elbo = (
			-1
			* H1_results["loss_trace"][-1].numpy()
			/ (num_datapoints_x + num_datapoints_y)
		)

		H0_elbo = (
			-1
			* H0_results["loss_trace"][-1].numpy()
			/ (num_datapoints_x + num_datapoints_y)
		)

		curr_bf = H1_elbo - H0_elbo
		print("BF control: {}".format(curr_bf))
		bfs_control.append(curr_bf)

	plt.figure(figsize=(9, 8))
	results_df = pd.DataFrame({"control": bfs_control, "shuffled": bfs_shuffled, "experiment": bfs_experiment})
	results_df_melted = pd.melt(results_df)
	results_df_melted.to_csv("../out/splatter/global_ebfs.csv")
	ax = sns.boxplot(data=results_df_melted, x="variable", y="value", color="black")
	for patch in ax.artists:
		r, g, b, a = patch.get_facecolor()
		patch.set_facecolor((r, g, b, .3))
	plt.title("Global ELBO Bayes factors")
	plt.xticks(np.arange(3), labels=[
			   "Unperturbed\nnull", "Shuffled\nnull", "Perturbed"])
	plt.ylabel("log(EBF)")
	plt.tight_layout()
	plt.savefig("../out/splatter/global_ebfs.png")
	plt.close()
	

