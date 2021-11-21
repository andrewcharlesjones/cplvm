import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd
import os
from os.path import join as pjoin
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

DATA_DIR = "../out/cai_good"

files = os.listdir(DATA_DIR)

p_list = [10, 100, 1000]


plt.figure(figsize=(7, 7))


######## CPLVM ##########

# Load CPLVM results
# cplvm_stats = np.load(
#     pjoin(DATA_DIR, "bfs_targeted.npy")
# )
cplvm_stats = np.load(
    pjoin("/Users/andrewjones/Desktop", "bfs_targeted.npy")
)

bfs_stimulated_set = cplvm_stats[:, 0]
bfs_unstimulated_set = np.ndarray.flatten(cplvm_stats[:, 1:])
# import ipdb; ipdb.set_trace()

tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
    y_true=np.concatenate(
        [np.zeros(len(bfs_unstimulated_set)), np.ones(len(bfs_stimulated_set))]
    ),
    y_score=np.concatenate([bfs_unstimulated_set, bfs_stimulated_set]),
)
plt.plot(tpr_shuffled, fpr_shuffled, label="CPLVM", linestyle=":", color="black")
# plt.show()

cplvm_df = pd.DataFrame({"TPR": tpr_shuffled, "FPR": fpr_shuffled})
cplvm_df['method'] = "CPLVM"

######## Li 2012 ##########
li2012_stats_experiment = np.load(
    "../out/li2012/test_stats_experiment.npy"
)
li2012_stats_null = np.load("../out/li2012/test_stats_shuffled.npy")

tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
    y_true=np.concatenate(
        [np.zeros(len(li2012_stats_null)), np.ones(len(li2012_stats_experiment))]
    ),
    y_score=np.concatenate([li2012_stats_null, li2012_stats_experiment]),
)
plt.plot(tpr_shuffled, fpr_shuffled, label=r"$\emph{Li}$", linestyle="--", color="black")
li2012_df = pd.DataFrame({"TPR": tpr_shuffled, "FPR": fpr_shuffled})
# li2012_df['method'] = "Li 2012"
li2012_df['method'] = r"$\emph{Li}$"




# plt.legend(prop={"size": 20})
# plt.xlabel("TPR")
# plt.ylabel("FPR")
# plt.plot([0, 1], [0, 1], "--", color="black")

# plot_df = pd.concat([cplvm_df, li2012_df], axis=0)
# g = sns.lineplot(data=plot_df, x="TPR", y="FPR", style="method", color="black", ci=95, err_style="band")
# g.legend_.set_title(None)
plt.legend(prop={"size": 20})
plt.xlabel("TPR")
plt.ylabel("FPR")


# import ipdb; ipdb.set_trace()

plt.tight_layout()
plt.savefig("../out/roc_comparison_genesets.png")
plt.show()
import ipdb; ipdb.set_trace()

