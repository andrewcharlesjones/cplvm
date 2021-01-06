import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

NUM_SETS_TO_PLOT = 8
gene_name = "HIF1A"

# Load BFs and gene set names
treatment_bfs = np.load("./out/geneset_bfs_{}.npy".format(gene_name))
gene_sets_for_plot = np.load("./out/geneset_names_{}.npy".format(gene_name))

plt.figure(figsize=(10, 7))

sorted_idx = np.argsort(-np.array(treatment_bfs))[:NUM_SETS_TO_PLOT]
treatment_bfs_to_plot = np.array(treatment_bfs)[sorted_idx]
plt.bar(np.arange(len(treatment_bfs_to_plot)), treatment_bfs_to_plot)
plt.title(gene_name.upper() + " gene set BFs")
plt.ylabel("log(BF)")


plt.xticks(np.arange(len(treatment_bfs_to_plot)), labels=gene_sets_for_plot[:len(treatment_bfs)][sorted_idx])
plt.xticks(rotation=-45, size=20, ha="left")
plt.tight_layout()
plt.savefig("./out/{}_gene_set_bfs.png".format(gene_name))
# plt.close()
# plt.show()
# import ipdb; ipdb.set_trace()

