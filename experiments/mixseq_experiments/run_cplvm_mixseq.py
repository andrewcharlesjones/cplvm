import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from os.path import join as pjoin

from cplvm import CPLVM

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../../data/mix_seq/data/nutlin/"


if __name__ == "__main__":

    latent_dim_shared = 2
    latent_dim_foreground = 2

    X_fname = pjoin(DATA_DIR, "dmso_expt1.csv")
    Y_fname = pjoin(DATA_DIR, "nutlin_expt1.csv")

    X_mutation_fname = pjoin(DATA_DIR, "p53_mutations_dmso.csv")
    Y_mutation_fname = pjoin(DATA_DIR, "p53_mutations_nutlin.csv")

    p53_mutations_X = pd.read_csv(X_mutation_fname, index_col=0)
    p53_mutations_X.tp53_mutation[
        p53_mutations_X.tp53_mutation == "Hotspot"
    ] = "Mutated"
    p53_mutations_X.tp53_mutation[
        p53_mutations_X.tp53_mutation == "Other"
    ] = "Wild-type"

    p53_mutations_Y = pd.read_csv(Y_mutation_fname, index_col=0)
    p53_mutations_Y.tp53_mutation[
        p53_mutations_Y.tp53_mutation == "Hotspot"
    ] = "Mutated"
    p53_mutations_Y.tp53_mutation[
        p53_mutations_Y.tp53_mutation == "Other"
    ] = "Wild-type"

    # Read in data
    X = pd.read_csv(X_fname, index_col=0)
    Y = pd.read_csv(Y_fname, index_col=0)

    plt.figure(figsize=(21, 7))
    mutations_X = p53_mutations_X
    mutations_Y = p53_mutations_Y

    idx_to_plot_Y = np.where(mutations_Y.tp53_mutation.values != "NotAvailable")[0]

    ###### PCA ######

    # Run PCA
    pca_reduced = PCA(n_components=2).fit_transform(np.concatenate([X, Y], axis=0))
    pca_reduced_df = pd.DataFrame(pca_reduced)
    pca_reduced_Y_df = pca_reduced_df.iloc[X.shape[0] :, :]

    plt.subplot(131)
    sns.scatterplot(
        data=pca_reduced_Y_df.iloc[idx_to_plot_Y, :],
        x=0,
        y=1,
        hue=mutations_Y.tp53_mutation.values[idx_to_plot_Y],
        alpha=0.5,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA")

    # Compute SS
    cluster_score_pca = silhouette_score(
        X=pca_reduced_Y_df.iloc[idx_to_plot_Y, :],
        labels=mutations_Y.tp53_mutation.values[idx_to_plot_Y],
    )
    print("SS PCA: {}".format(cluster_score_pca))

    ###### CPLVM ######
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

    plt.subplot(132)
    sns.scatterplot(
        data=ty_df.iloc[idx_to_plot_Y, :],
        x=0,
        y=1,
        hue=mutations_Y.tp53_mutation.values[idx_to_plot_Y],
        alpha=0.5,
    )
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title("CPLVM")

    # Compute SS
    cluster_score_cplvm = silhouette_score(
        X=ty_df.iloc[idx_to_plot_Y, :],
        labels=mutations_Y.tp53_mutation.values[idx_to_plot_Y],
    )

    # Plot silhouette scores
    plt.subplot(133)
    plt.bar(np.arange(2), [cluster_score_pca, cluster_score_cplvm])
    plt.xticks(np.arange(2), labels=["PCA", "CPLVM"])
    plt.ylabel("Silhouette score")

    plt.tight_layout()
    plt.savefig("./out/nutlin_latent_scatter.png")
    plt.show()
