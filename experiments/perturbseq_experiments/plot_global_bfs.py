import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

control_bfs = np.load("./out/control_bfs.npy")
control_bf_stds = np.load("./out/control_bf_stds.npy")
treatment_bfs = np.load("./out/treatment_bfs.npy")
gene_names_so_far = np.load("./out/ps_gene_names.npy")
gene_names_so_far = np.array([str(x.decode("UTF-8")) for x in gene_names_so_far])
gene_names_so_far = gene_names_so_far[gene_names_so_far != ".DS_Store"]
# import ipdb; ipdb.set_trace()

N = treatment_bfs.shape[0]

fig, ax = plt.subplots(figsize=(21, 5))

ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars
p1 = ax.bar(ind, control_bfs, width, yerr=control_bf_stds)

p2 = ax.bar(ind + width, treatment_bfs, width)


# ax.set_title('Scores by group and gender')
ax.set_title("Global ELBO Bayes factors, Perturb-seq")
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(gene_names_so_far[:N], size=20)

ax.legend((p1[0], p2[0]), ("Control", "Treatment"), prop={"size": 20})
plt.ylabel("log(EBF)")
plt.tight_layout()
plt.savefig("./out/perturbseq_global_BFs.png")
plt.show()
