import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from scipy import stats


import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


def mean_confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    m, se = np.mean(data), stats.sem(data)
    width = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return width


# Load BFs
all_elbos = np.load("../out/bfs_targeted.npy")
n_gene_sets = 10  # all_elbos.shape[1]
gene_set_names = ["Set {}".format(x + 1) for x in range(n_gene_sets)]


box_colors = ["gray" for _ in range(n_gene_sets)]
box_colors[0] = "red"

# Shuffled null
box_colors.append("black")
num_shuffled_null = len(all_elbos[0]) - n_gene_sets
shuffled_null_bfs = all_elbos[:, -num_shuffled_null:].flatten()
gene_set_names.extend(["Shuffled null" for _ in range(num_shuffled_null)])

# import ipdb; ipdb.set_trace()
# Plot boxplot
plt.figure(figsize=(14, 7))
ax = sns.boxplot(
    data=pd.melt(pd.DataFrame(all_elbos, columns=gene_set_names)),
    x="variable",
    y="value",
    color="gray",
)


mybox = ax.artists[0]
mybox.set_facecolor("red")

mybox = ax.artists[-1]
mybox.set_facecolor("black")

red_patch = mpatches.Patch(color="red", label="Perturbed")
gray_patch = mpatches.Patch(color="gray", label="Unperturbed")
black_patch = mpatches.Patch(color="black", label="Shuffled")
plt.legend(
    handles=[red_patch, gray_patch, black_patch], fontsize=20, loc="upper center"
)

# Plot confidence interval bands
sn_mean = np.mean(shuffled_null_bfs)
sn_ci = mean_confidence_interval(shuffled_null_bfs)
# import ipdb; ipdb.set_trace()
plt.axhline(sn_mean)
plt.axhline(sn_mean + sn_ci, linestyle="--")
plt.axhline(sn_mean - sn_ci, linestyle="--")


plt.ylabel("log(EBF)")
plt.xlabel("")
plt.xticks(rotation=90)
plt.title("Gene set ELBO Bayes factors")
plt.tight_layout()
# plt.savefig("./out/bfs_targeted_gene_sets_boxplot.png")
plt.show()
