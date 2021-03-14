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

DATA_DIR = "./out/cai"

files = os.listdir(DATA_DIR)

p_list = [10, 100, 1000]


plt.figure(figsize=(21, 6))
for ii, p in enumerate(p_list):
    plt.subplot(1, 3, ii + 1)
    plt.title("n=m=100, p={}".format(p))

    # Load Cai results
    cai_stats_experiment = np.load(
        pjoin(DATA_DIR, "test_stats_experiment_p{}.npy".format(p))
    )
    cai_stats_null = np.load(pjoin(DATA_DIR, "test_stats_shuffled_p{}.npy".format(p)))

    # Load CPLVM results
    cplvm_stats_experiment = np.load(
        pjoin(DATA_DIR, "bfs_experiment_p{}.npy".format(p))
    )
    cplvm_stats_null = np.load(pjoin(DATA_DIR, "bfs_shuffled_p{}.npy".format(p)))

    # Compute and plot ROC curves
    tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
        y_true=np.concatenate(
            [np.zeros(len(cai_stats_null)), np.ones(len(cai_stats_experiment))]
        ),
        y_score=np.concatenate([cai_stats_null, cai_stats_experiment]),
    )
    plt.plot(tpr_shuffled, fpr_shuffled, label="Cai et al., 2013")

    tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
        y_true=np.concatenate(
            [np.zeros(len(cplvm_stats_null)), np.ones(len(cplvm_stats_experiment))]
        ),
        y_score=np.concatenate([cplvm_stats_null, cplvm_stats_experiment]),
    )
    plt.plot(tpr_shuffled, fpr_shuffled, label="CPLVM")

    plt.legend(prop={"size": 20})
    plt.xlabel("TPR")
    plt.ylabel("FPR")
    plt.plot([0, 1], [0, 1], "--", color="black")

    plt.tight_layout()
plt.savefig("./out/roc_comparison.png")
plt.show()
