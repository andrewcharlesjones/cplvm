import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

import sys
sys.path.append("../../models")

from clvm_tfp_poisson import fit_model as fit_clvm

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


if __name__ == "__main__":

    latent_dim_shared = 2
    latent_dim_target = 2

    X_fname = "/Users/andrewjones/Documents/beehive/differential_covariance/mix_seq/data/nutlin/dmso_expt1.csv"
    Y_fname = "/Users/andrewjones/Documents/beehive/differential_covariance/mix_seq/data/nutlin/nutlin_expt1.csv"

    p53_mutations = pd.read_csv("/Users/andrewjones/Documents/beehive/differential_covariance/mix_seq/data/nutlin/p53_mutations.csv", index_col=0)
    p53_mutations.tp53_mutation[p53_mutations.tp53_mutation == "Hotspot"] = "Mutated"
    p53_mutations.tp53_mutation[p53_mutations.tp53_mutation == "Other"] = "Wild-type"


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

    plt.figure(figsize=(14, 7))
    idx_to_plot = np.where(p53_mutations.tp53_mutation.values != "NotAvailable")[0]
    plt.subplot(121)
    sns.scatterplot(data=zy_df.iloc[idx_to_plot, :], x=0, y=1, hue=p53_mutations.tp53_mutation.values[idx_to_plot], alpha=0.5)
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title("Shared latent variables")
    plt.subplot(122)
    sns.scatterplot(data=ty_df.iloc[idx_to_plot, :], x=0, y=1, hue=p53_mutations.tp53_mutation.values[idx_to_plot], alpha=0.5)
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title("Foreground-specific\nlatent variables")
    plt.tight_layout()
    plt.savefig("./out/nutlin_latent_scatter.png")
    plt.show()


    # plt.figure(figsize=(14, 6))
    # plt.subplot(121)
    # sns.kdeplot(ty_estimated[0, p53_mutations.tp53_mutation.values == "Hotspot"], label="Hotspot")
    # sns.kdeplot(ty_estimated[0, p53_mutations.tp53_mutation.values == "Other"], label="Other")
    # plt.title("ty, latent dim 1")
    # plt.legend(prop={'size': 20})

    # plt.subplot(122)
    # sns.kdeplot(ty_estimated[1, p53_mutations.tp53_mutation.values == "Hotspot"], label="Hotspot")
    # sns.kdeplot(ty_estimated[1, p53_mutations.tp53_mutation.values == "Other"], label="Other")
    # plt.title("ty, latent dim 2")
    # plt.legend(prop={'size': 20})
    # plt.savefig("./out/nutlin_ty_hist.png")
    # plt.show()

    # plt.figure(figsize=(14, 6))
    # plt.subplot(121)
    # sns.kdeplot(zy_estimated[0, p53_mutations.tp53_mutation.values == "Hotspot"], label="Hotspot")
    # sns.kdeplot(zy_estimated[0, p53_mutations.tp53_mutation.values == "Other"], label="Other")
    # plt.title("zy, latent dim 1")
    # plt.legend(prop={'size': 20})

    # plt.subplot(122)
    # sns.kdeplot(zy_estimated[1, p53_mutations.tp53_mutation.values == "Hotspot"], label="Hotspot")
    # sns.kdeplot(zy_estimated[1, p53_mutations.tp53_mutation.values == "Other"], label="Other")
    # plt.title("zy, latent dim 2")
    # plt.legend(prop={'size': 20})
    # plt.savefig("./out/nutlin_zy_hist.png")
    # plt.show()


    # import ipdb; ipdb.set_trace()


