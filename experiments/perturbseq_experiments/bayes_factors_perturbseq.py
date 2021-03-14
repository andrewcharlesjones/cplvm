import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

import socket

if socket.gethostname() == "andyjones":
    DATA_DIR = "../../data/perturb_seq/data/targeted_genes"
else:
    DATA_DIR = "../data/targeted_genes/"

from cplvm import CPLVM

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


if __name__ == "__main__":

    gene_names = [x for x in os.listdir(DATA_DIR) if x[0] != "."]

    latent_dim_shared = 3
    latent_dim_target = 3

    num_null_repeats = 5

    control_bfs = []
    control_bf_stds = []
    treatment_bfs = []
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

        #### Negative control data #####

        # Read in data
        dat1 = pd.read_csv(X_fname, index_col=0)
        dat2 = pd.read_csv(Y_fname, index_col=0)
        dat = pd.concat([dat1, dat2], axis=0)
        # dat = pd.read_csv(X_fname, index_col=0)

        # Split 0hr data into two pieces to do Bayes Factor calibration
        n_total = dat.shape[0]
        n1 = n_total // 2
        x1_idx = np.random.choice(np.arange(n_total), size=n1, replace=False)
        x2_idx = np.setdiff1d(np.arange(n_total), x1_idx)

        # Arbitrarily call one of the datasets X and the other Y
        X = dat.iloc[x1_idx, :].values.T
        Y = dat.iloc[x2_idx, :].values.T

        # import ipdb; ipdb.set_trace()

        curr_control_bfs = []
        for _ in range(num_null_repeats):
            cplvm = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )

            H1_results = cplvm.fit_model_vi(
                X, Y, compute_size_factors=True, is_H0=False
            )
            H0_results = cplvm.fit_model_vi(X, Y, compute_size_factors=True, is_H0=True)

            H1_elbo = (
                -1 * H1_results["loss_trace"][-1].numpy() / (X.shape[1] + Y.shape[1])
            )
            H0_elbo = (
                -1 * H0_results["loss_trace"][-1].numpy() / (X.shape[1] + Y.shape[1])
            )

            curr_bf = H1_elbo - H0_elbo
            curr_control_bfs.append(curr_bf)
            print("{} control BF: {}".format(gene_name, curr_bf))
        control_bfs.append(np.mean(curr_control_bfs))
        control_bf_stds.append(np.std(curr_control_bfs))

        print("{} control BF: {}".format(gene_name, np.mean(curr_control_bfs)))

        #### Treatment and control data #####

        # Read in data
        X = pd.read_csv(X_fname, index_col=0).values.T
        Y = pd.read_csv(Y_fname, index_col=0).values.T

        H1_results = fit_clvm(
            X,
            Y,
            latent_dim_shared,
            latent_dim_target,
            compute_size_factors=True,
            is_H0=False,
        )
        H0_results = fit_clvm(
            X,
            Y,
            latent_dim_shared,
            latent_dim_target,
            compute_size_factors=True,
            is_H0=True,
        )

        H1_elbo = -1 * H1_results["loss_trace"][-1].numpy() / (X.shape[1] + Y.shape[1])
        H0_elbo = -1 * H0_results["loss_trace"][-1].numpy() / (X.shape[1] + Y.shape[1])

        curr_treatment_bf = H1_elbo - H0_elbo

        treatment_bfs.append(np.mean(curr_treatment_bf))

        print("{} treatment BF: {}".format(gene_name, curr_treatment_bf))

        gene_names_so_far.append(gene_name.upper())

        # Make paired barplot for each gene
        bf_df = pd.DataFrame(
            {
                "gene": gene_names_so_far,
                "control_bf": control_bfs,
                "treatment_bf": treatment_bfs,
            }
        )
        bf_df_melted = pd.melt(bf_df, id_vars="gene")

        N = bf_df.shape[0]

        fig, ax = plt.subplots(figsize=(21, 5))

        ind = np.arange(N)
        width = 0.35
        p1 = ax.bar(ind, control_bfs, width, yerr=control_bf_stds)

        p2 = ax.bar(ind + width, treatment_bfs, width)

        np.save("./out/control_bfs.npy", np.array(control_bfs))
        np.save("./out/control_bf_stds.npy", np.array(control_bf_stds))
        np.save("./out/treatment_bfs.npy", np.array(treatment_bfs))
        np.save("./out/ps_gene_names.npy", np.array(gene_names_so_far))

        ax.set_title("Global Bayes factors, Perturb-seq")
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(gene_names_so_far)

        ax.legend((p1[0], p2[0]), ("Control", "Treatment"), prop={"size": 20})
        plt.ylabel("log(BF)")
        plt.tight_layout()
        plt.savefig("./out/perturbseq_global_BFs.png")
