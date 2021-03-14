import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


# Load BFs
all_elbos = np.load("./out/bfs_targeted_misdefined.npy")
n_gene_sets = all_elbos.shape[1]
gene_set_names = ["Set {}".format(x + 1) for x in range(n_gene_sets)]


box_colors = ["gray" for _ in range(n_gene_sets)]
box_colors[0] = "red"

# Shuffled null
# box_colors.append("black")
# gene_set_names.extend(["Shuffled null" for _ in range(len(all_elbos[0]) - n_gene_sets)])


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

# mybox = ax.artists[-1]
# mybox.set_facecolor('black')

red_patch = mpatches.Patch(color="red", label="Perturbed")
gray_patch = mpatches.Patch(color="gray", label="Unperturbed")
# black_patch = mpatches.Patch(color='black', label='Shuffled')
plt.legend(handles=[red_patch, gray_patch], fontsize=20, loc="upper center")


plt.ylabel("log(EBF)")
plt.xlabel("")
plt.xticks(rotation=90)
plt.title("Gene set ELBO Bayes factors")
plt.tight_layout()
plt.savefig("./out/bfs_gene_sets_misspecified.png")
plt.show()
