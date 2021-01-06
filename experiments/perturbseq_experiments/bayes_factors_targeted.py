import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import poisson

import sys
import socket


if socket.gethostname() == "andyjones":
    DATA_DIR = "../../perturb_seq/data/targeted_genes"
    sys.path.append("../../models")
else:
    DATA_DIR = "../data/targeted_genes/"
    sys.path.append("../models")

from clvm_tfp_poisson import fit_model

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

NUM_SETS_TO_PLOT = 8


if __name__ == "__main__":

    # Load gene sets
    gene_sets = pd.read_csv("./hallmark_genesets.csv", index_col=0)

    gene_sets_unique = gene_sets.gene_set.values
    gene_sets_for_plot = np.array([' '.join(x.split("_")[1:]) for x in gene_sets_unique])

    gene_names = [x for x in os.listdir(DATA_DIR) if x[0] != "."]
    gene_names = ["Hif1a"]

    latent_dim_shared = 3
    latent_dim_target = 1

    control_bfs = []
    
    gene_names_so_far = []
    for gene_name in gene_names:

        print("-" * 80)
        print("Fitting {}".format(gene_name))
        print("-" * 80)
        curr_dir = os.path.join(DATA_DIR, gene_name)
        curr_files = os.listdir(curr_dir)
        X_fname = [x for x in curr_files if "0hr" in x][0]
        Y_fname = [x for x in curr_files if "3hr" in x][0]
        X_fname = os.path.join(curr_dir, X_fname)
        Y_fname = os.path.join(curr_dir, Y_fname)

        #### Treatment and control data #####

        # Read in data
        X = pd.read_csv(X_fname, index_col=0)
        Y = pd.read_csv(Y_fname, index_col=0)

        data_gene_names = X.columns.values
        data_gene_names = np.array([x.split("_")[1].lower() for x in data_gene_names])

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
            

            H1_results = fit_model(X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False, num_test_genes=0)
            H0_results = fit_model(X, Y, latent_dim_shared, latent_dim_target, compute_size_factors=True, is_H0=False, num_test_genes=in_set_idx.shape[0])

            H1_elbo = -1 * H1_results['loss_trace'][-1].numpy() / (X.shape[1] + Y.shape[1])
            H0_elbo = -1 * H0_results['loss_trace'][-1].numpy() / (X.shape[1] + Y.shape[1])

            curr_treatment_bf = H1_elbo - H0_elbo
            treatment_bfs.append(curr_treatment_bf)

            print("{} treatment BF for gene set {}: {}".format(gene_name, curr_gene_set, curr_treatment_bf))

            plt.figure(figsize=(10, 7))
            
            sorted_idx = np.argsort(-np.array(treatment_bfs))[:NUM_SETS_TO_PLOT]
            treatment_bfs_to_plot = np.array(treatment_bfs)[sorted_idx]
            plt.bar(np.arange(len(treatment_bfs_to_plot)), treatment_bfs_to_plot)
            plt.title(gene_name.upper())
            plt.ylabel("log(BF)")


            plt.xticks(np.arange(len(treatment_bfs_to_plot)), labels=gene_sets_for_plot[:len(treatment_bfs)][sorted_idx])
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("./out/{}_gene_set_bfs.png".format(gene_name))
            plt.close()

            # Save BFs and gene set names
            np.save("./out/geneset_bfs_{}.npy".format(gene_name), treatment_bfs)
            np.save("./out/geneset_names_{}.npy".format(gene_name), gene_sets_for_plot[:len(treatment_bfs)])


