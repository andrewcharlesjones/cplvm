import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

S_gtex = pd.read_csv("./out/gtex_heart_S.csv", header=None, index_col=0)
W_gtex = pd.read_csv("./out/gtex_heart_W.csv", header=None, index_col=0)

RIGHT_OFFSET_TEXT = 6
FONTSIZE_TEXT = 20

plt.figure(figsize=(14, 6))

## Plot S
plt.subplot(121)

S_comp_idx = 0
S_sorted = S_gtex.iloc[:, S_comp_idx].sort_values(ascending=False)
plt.scatter(np.arange(S_sorted.shape[0]), S_sorted, color="black")

gene_idx = 0
plt.text(
    gene_idx + RIGHT_OFFSET_TEXT,
    S_sorted.iloc[gene_idx] - 2,
    "MYH7",
    fontsize=FONTSIZE_TEXT,
)

gene_idx = 1
plt.text(
    gene_idx + RIGHT_OFFSET_TEXT,
    S_sorted.iloc[gene_idx] - 12,
    "ACTC1",
    fontsize=FONTSIZE_TEXT,
)

gene_idx = 2
plt.text(
    gene_idx + RIGHT_OFFSET_TEXT,
    S_sorted.iloc[gene_idx] - 6,
    "ATP1A3",
    fontsize=FONTSIZE_TEXT,
)

gene_idx = 3
plt.text(
    gene_idx + RIGHT_OFFSET_TEXT,
    S_sorted.iloc[gene_idx] - 6,
    "TCAP",
    fontsize=FONTSIZE_TEXT,
)

plt.title("S, example component")
plt.xlabel("Gene index")
plt.ylabel("Loading")


## Plot W
plt.subplot(122)
W_comp_idx = 0
W_sorted = W_gtex.iloc[:, W_comp_idx].sort_values(ascending=False)
plt.scatter(np.arange(W_sorted.shape[0]), W_sorted, color="black")

gene_idx = 0
plt.text(
    gene_idx + RIGHT_OFFSET_TEXT,
    W_sorted.iloc[gene_idx] - 15,
    "SFTPB",
    fontsize=FONTSIZE_TEXT,
)

gene_idx = 1
plt.text(
    gene_idx + RIGHT_OFFSET_TEXT,
    W_sorted.iloc[gene_idx] - 15,
    "SFTPA2",
    fontsize=FONTSIZE_TEXT,
)

gene_idx = 2
plt.text(
    gene_idx + RIGHT_OFFSET_TEXT,
    W_sorted.iloc[gene_idx] - 15,
    "SFTPC",
    fontsize=FONTSIZE_TEXT,
)

gene_idx = 3
plt.text(
    gene_idx + RIGHT_OFFSET_TEXT,
    W_sorted.iloc[gene_idx] - 15,
    "SFTPA1",
    fontsize=FONTSIZE_TEXT,
)

plt.title("W, example component")
plt.xlabel("Gene index")
plt.ylabel("Loading")

plt.tight_layout()
plt.savefig("./out/gtex_heart_scatter.png")
plt.show()
# import ipdb; ipdb.set_trace()
