import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import poisson


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from cplvm import CPLVM

import socket
from os.path import join as pjoin


if socket.gethostname() == "andyjones":
    DATA_DIR = "../../data/mix_seq/data/nutlin/"
else:
    DATA_DIR = "../data/mix_seq/"


import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

NUM_SETS_TO_PLOT = 8


if __name__ == "__main__":

    # Load gene sets
    gene_sets = pd.read_csv("../perturbseq_experiments/hallmark_genesets.csv", index_col=0)

    gene_sets_unique = gene_sets.gene_set.values
    gene_sets_for_plot = np.array([' '.join(x.split("_")[1:]) for x in gene_sets_unique])

    control_bfs = []
    
    latent_dim_shared = 2
    latent_dim_target = 2

    X_fname = pjoin(DATA_DIR, "data/nutlin/dmso_expt1.csv")
    Y_fname = pjoin(DATA_DIR, "data/nutlin/nutlin_expt1.csv")
    gene_fname = pjoin(DATA_DIR, "data/nutlin/gene_symbols.csv")


    # Read in data
    X = pd.read_csv(X_fname, index_col=0)
    Y = pd.read_csv(Y_fname, index_col=0)
    gene_names = pd.read_csv(gene_fname, index_col=0).iloc[:, 0].values
    gene_names[pd.isna(gene_names)] = ''
    gene_names = [x.lower() for x in gene_names]
    X.columns = gene_names
    Y.columns = gene_names
    data_gene_names = np.array(gene_names)
        
    X = X.values.T
    Y = Y.values.T

    # Loop over gene sets
    treatment_bfs = []
    for curr_gene_set in gene_sets_unique:

        gene_string = gene_sets.genes[gene_sets.gene_set == curr_gene_set].values[0].lower()
        genes_in_set = np.array(gene_string.split(","))

        in_set_idx = np.where(np.isin(data_gene_names, genes_in_set))[0]
        out_set_idx = np.setdiff1d(np.arange(data_gene_names.shape[0]), in_set_idx)

        X = np.concatenate([X[out_set_idx, :], X[in_set_idx, :]])
        Y = np.concatenate([Y[out_set_idx, :], Y[in_set_idx, :]])
        
        cplvm = CPLVM(k_shared=latent_dim_shared, k_foreground=latent_dim_foreground)

        H1_results = cplvm.fit_model_vi(X, Y, compute_size_factors=True, is_H0=False, num_test_genes=0)
        H0_results = cplvm.fit_model_vi(X, Y, compute_size_factors=True, is_H0=False, num_test_genes=in_set_idx.shape[0])

        H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (X.shape[1] + Y.shape[1])
        H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (X.shape[1] + Y.shape[1])

        curr_treatment_bf = H1_elbo - H0_elbo
        treatment_bfs.append(curr_treatment_bf)

        print("{} treatment BF for gene set {}: {}".format("Nutlin", curr_gene_set, curr_treatment_bf))

        plt.figure(figsize=(10, 7))

        

        sorted_idx = np.argsort(-np.array(treatment_bfs))[:NUM_SETS_TO_PLOT]
        treatment_bfs_to_plot = np.array(treatment_bfs)[sorted_idx]
        plt.bar(np.arange(len(treatment_bfs_to_plot)), treatment_bfs_to_plot)
        plt.title("Nutlin")
        plt.ylabel("log(BF)")


        plt.xticks(np.arange(len(treatment_bfs_to_plot)), labels=gene_sets_for_plot[:len(treatment_bfs)][sorted_idx])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("./out/nutlin_gene_sets.png")
        plt.close()

        # Save BFs and gene set names
        np.save("./out/geneset_bfs_nutlin.npy", treatment_bfs)
        np.save("./out/geneset_names_nutlin.npy", gene_sets_for_plot[:len(treatment_bfs)])

