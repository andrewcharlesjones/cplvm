import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

NUM_SETS_TO_PLOT = 8
gene_name = "HIF1A"

# Load BFs and gene set names
treatment_bfs = np.load("./out/geneset_bfs_{}.npy".format(gene_name))
gene_sets_for_plot = np.load("./out/geneset_names_{}.npy".format(gene_name))
gene_sets = pd.read_csv("./hallmark_genesets.csv", index_col=0)


plt.figure(figsize=(10, 7))

sorted_idx = np.argsort(-np.array(treatment_bfs))[:NUM_SETS_TO_PLOT]
treatment_bfs_to_plot = np.array(treatment_bfs)[sorted_idx]
plt.bar(np.arange(len(treatment_bfs_to_plot)), treatment_bfs_to_plot, color="black")
plt.title(gene_name.upper() + " gene set EBFs")
plt.ylabel("log(EBF)")


plt.xticks(np.arange(len(treatment_bfs_to_plot)), labels=gene_sets_for_plot[:len(treatment_bfs)][sorted_idx])
plt.xticks(rotation=-45, size=20, ha="left")
plt.tight_layout()
plt.savefig("./out/{}_gene_set_bfs.png".format(gene_name))
plt.show()
plt.close()
import ipdb

ipdb.set_trace()

### Plot size of each gene set by their enrichments
gene_sets["set_size"] = [len(x) for x in gene_sets.genes]
gene_sets.gene_set = gene_sets.gene_set.str.split("_").str[1:].str.join(" ")
bf_df = pd.DataFrame({"ebf": treatment_bfs, "gene_set": gene_sets_for_plot})

plot_df = pd.merge(gene_sets, bf_df, on="gene_set")

plt.scatter(plot_df.set_size.values, plot_df.ebf.values)
# sns.scatterplot(data=plot_df, x="ebf", y="set_size")
plt.show()
import ipdb

ipdb.set_trace()
