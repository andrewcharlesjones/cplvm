import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.patches as mpatches
from scipy import stats


import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


gene_set_sizes = [25, 20, 15, 10, 5, 1]


plt.figure(figsize=(28, 7))

### Plot BFs for misspecified gene sets
all_elbos = np.load("./out/bfs_targeted_misdefined.npy")
n_gene_sets = all_elbos.shape[1]
gene_set_names = ["Set {}".format(x + 1) for x in range(n_gene_sets)]


box_colors = ["gray" for _ in range(n_gene_sets)]
box_colors[0] = "red"

# Plot boxplot
plt.subplot(121)
ax = sns.boxplot(
    data=pd.melt(pd.DataFrame(all_elbos, columns=gene_set_names)),
    x="variable",
    y="value",
    color="gray",
)


mybox = ax.artists[0]
mybox.set_facecolor("red")

red_patch = mpatches.Patch(color="red", label="Perturbed")
gray_patch = mpatches.Patch(color="gray", label="Unperturbed")
plt.legend(handles=[red_patch, gray_patch], fontsize=20, loc="upper center")


### Plot confidence interval from shuffled null


def mean_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    m, se = np.mean(data), stats.sem(data)
    width = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return width


# Load BFs
all_elbos = np.load("./out/bfs_targeted.npy")
n_gene_sets = 10

num_shuffled_null = len(all_elbos[0]) - n_gene_sets
shuffled_null_bfs = all_elbos[:, -num_shuffled_null:].flatten()

# Plot confidence interval bands
sn_mean = np.mean(shuffled_null_bfs)
sn_ci = mean_confidence_interval(shuffled_null_bfs)
plt.axhline(sn_mean, color="black")
plt.axhline(sn_mean + sn_ci, linestyle="--", color="black")
plt.axhline(sn_mean - sn_ci, linestyle="--", color="black")


plt.ylabel("log(EBF)")
plt.xlabel("")
plt.xticks(rotation=90)
# plt.title("Gene set ELBO Bayes factors")
ax1 = plt.gca()


### Plot BFs for varying gene set sizes

all_bfs = np.load("./out/bfs_targeted_vary_size.npy")

bf_df = pd.DataFrame(np.array(all_bfs).T, columns=gene_set_sizes)
bf_df_melted = pd.melt(pd.DataFrame(bf_df))


plt.subplot(122)
sns.boxplot(data=bf_df_melted, x="variable", y="value")
plt.ylabel("log(EBF)")
plt.xlabel("Perturbed gene set size")
ax2 = plt.gca()

ylims = [
    min(ax1.get_ylim()[0], ax2.get_ylim()[0]),
    max(ax1.get_ylim()[1], ax2.get_ylim()[1]),
]
ax1.set_ylim(ylims)
ax2.set_ylim(ylims)


plt.tight_layout()
plt.savefig("./out/bfs_gene_sets_robustness.png")
plt.show()
plt.close()

import ipdb

ipdb.set_trace()
