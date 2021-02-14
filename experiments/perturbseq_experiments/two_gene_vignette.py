import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import os
from scipy.stats import pearsonr
import seaborn as sns

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

DATA_DIR = "/Users/andrewjones/Documents/beehive/differential_covariance/perturb_seq/data/targeted_genes/"



gene = "Hif1a"
mim_path = "/Users/andrewjones/Documents/beehive/differential_covariance/clvm/perturbseq_experiments/out/mimosca_coefficients/{}/beta.csv".format(gene)
mimosca_coeffs = pd.read_csv(mim_path, index_col=0)
mimosca_coeffs.columns = ['mimosca_beta']


curr_dir = os.path.join(DATA_DIR, gene)
curr_files = os.listdir(curr_dir)
X_fname = [x for x in curr_files if "0hr" in x][0]
Y_fname = [x for x in curr_files if "3hr" in x][0]
X_fname = os.path.join(curr_dir, X_fname)
Y_fname = os.path.join(curr_dir, Y_fname)

X = pd.read_csv(X_fname, index_col=0)
Y = pd.read_csv(Y_fname, index_col=0)

### Plot expression of two genes against each other ###

# gene_name1 = "ENSMUSG00000069516_Lyz2"
gene_name1 = "ENSMUSG00000069516_Lyz2"
gene_name2 = "ENSMUSG00000018930_Ccl4"

X1, X2 = np.log(X[gene_name1] + 1), np.log(X[gene_name2] + 1)
Y1, Y2 = np.log(Y[gene_name1] + 1), np.log(Y[gene_name2] + 1)
# X1, X2 = X[gene_name1], X[gene_name2]
# Y1, Y2 = Y[gene_name1], Y[gene_name2]

print("Correlation in control:", round(pearsonr(X1, X2)[0], 3))
print("Correlation in treatment:", round(pearsonr(Y1, Y2)[0], 3))

plt.figure(figsize=(22, 7))
plt.subplot(131)
sns.regplot(X1, X2, label="Background")
sns.regplot(Y1, Y2, label="Foreground")
plt.legend(prop={'size': 20})
plt.xlabel(gene_name1.split("_")[1].upper() + " log-expression")
plt.ylabel(gene_name2.split("_")[1].upper() + " log-expression")
plt.title(gene.upper() + " experiment")
# plt.tight_layout()
# plt.show()



## Plot linear model coefficients

plot_mat = mimosca_coeffs.sort_values("mimosca_beta", ascending=False).iloc[:, 0].values
gene1_idx = np.where(mimosca_coeffs.sort_values("mimosca_beta", ascending=False).index.values == gene_name1)[0]
gene2_idx = np.where(mimosca_coeffs.sort_values("mimosca_beta", ascending=False).index.values == gene_name2)[0]
# plt.figure(figsize=(10, 7))
plt.subplot(132)
plt.scatter(np.arange(len(plot_mat)), plot_mat)
plt.scatter(gene1_idx, plot_mat[gene1_idx], color="red")
plt.scatter(gene2_idx, plot_mat[gene2_idx], color="red")
# plt.axvline(gene1_idx, c="r", linestyle="--")
# plt.axvline(gene2_idx, c="r", linestyle="--")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Gene index")
plt.ylabel("Linear model coefficient")
plt.title("Linear model")
ax = plt.gca()
# plt.text(gene1_idx[0] - 20, ax.get_ylim()[1] + 0.15, gene_name1.split("_")[1].upper())
# plt.text(gene2_idx[0] - 20, ax.get_ylim()[1] + 0.15, gene_name2.split("_")[1].upper())
plt.text(gene1_idx[0] + 20, plot_mat[gene1_idx] + 0.25, gene_name1.split("_")[1].upper())
plt.text(gene2_idx[0] + 20, plot_mat[gene2_idx] - 0.25, gene_name2.split("_")[1].upper())



## Plot CPLVM loadings values
W_cplvm = pd.read_csv("/Users/andrewjones/Documents/beehive/differential_covariance/clvm/perturbseq_experiments/out/loading_matrices/Hif1a/W_mat.csv", index_col=0)

plot_mat = W_cplvm.iloc[:, 2].sort_values(ascending=False).values
gene1_idx = np.where(W_cplvm.iloc[:, 2].sort_values(ascending=False).index.values == gene_name1)[0]
gene2_idx = np.where(W_cplvm.iloc[:, 2].sort_values(ascending=False).index.values == gene_name2)[0]

plt.subplot(133)
plt.scatter(np.arange(len(plot_mat)), plot_mat)
plt.scatter(gene1_idx, plot_mat[gene1_idx], color="red")
plt.scatter(gene2_idx, plot_mat[gene2_idx], color="red")
# plt.axvline(gene1_idx, c="r", linestyle="--")
# plt.axvline(gene2_idx, c="r", linestyle="--")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Gene index")
plt.ylabel("CPLVM loading")
plt.title("CPLVM")
ax = plt.gca()
# plt.text(gene1_idx[0] - 20, ax.get_ylim()[1] + 0.15, gene_name1.split("_")[1].upper())
# plt.text(gene2_idx[0] - 20, ax.get_ylim()[1] + 0.15, gene_name2.split("_")[1].upper())
plt.text(gene1_idx[0] + 20, plot_mat[gene1_idx] - .5, gene_name1.split("_")[1].upper())
plt.text(gene2_idx[0] + 20, plot_mat[gene2_idx] - .5, gene_name2.split("_")[1].upper())





plt.tight_layout()
plt.savefig("./out/lyz2_vs_ccl4.png")
plt.show()

import ipdb; ipdb.set_trace()

