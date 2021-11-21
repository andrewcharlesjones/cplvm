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


plt.figure(figsize=(21, 7))
for ii, p in enumerate(p_list):

    plt.subplot(1, 3, ii + 1)
    plt.title(r"$p={}$".format(p))

    ######## Cai ###########

    cai_stats_experiment = np.load(
        pjoin(DATA_DIR, "test_stats_experiment_p{}.npy".format(p))
    )
    cai_stats_null = np.load(pjoin(DATA_DIR, "test_stats_shuffled_p{}.npy".format(p)))

    # Compute and plot ROC curves
    tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
        y_true=np.concatenate(
            [np.zeros(len(cai_stats_null)), np.ones(len(cai_stats_experiment))]
        ),
        y_score=np.concatenate([cai_stats_null, cai_stats_experiment]),
    )

    cai_df = pd.DataFrame({"TPR": tpr_shuffled, "FPR": fpr_shuffled})
    # cai_df['method'] = "Cai et al., 2013"
    # cai_df['method'] = "Cai"
    cai_df["method"] = r"$\emph{Cai}$"

    # plt.plot(tpr_shuffled, fpr_shuffled, label="Cai et al., 2013")

    ######## CPLVM ##########

    # Load CPLVM results
    cplvm_stats_experiment = np.load(
        pjoin(DATA_DIR, "bfs_experiment_p{}.npy".format(p))
    )
    cplvm_stats_null = np.load(pjoin(DATA_DIR, "bfs_shuffled_p{}.npy".format(p)))

    tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
        y_true=np.concatenate(
            [np.zeros(len(cplvm_stats_null)), np.ones(len(cplvm_stats_experiment))]
        ),
        y_score=np.concatenate([cplvm_stats_null, cplvm_stats_experiment]),
    )
    # plt.plot(tpr_shuffled, fpr_shuffled, label="CPLVM", linestyle="--")
    cplvm_df = pd.DataFrame({"TPR": tpr_shuffled, "FPR": fpr_shuffled})
    cplvm_df["method"] = "CPLVM"

    ######## Johnstone ##########
    johnstone_stats_experiment = np.load(
        "../out/johnstone/test_stats_experiment_p{}.npy".format(p)
    )
    johnstone_stats_null = np.load(
        "../out/johnstone/test_stats_shuffled_p{}.npy".format(p)
    )

    tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
        y_true=np.concatenate(
            [
                np.zeros(len(johnstone_stats_null)),
                np.ones(len(johnstone_stats_experiment)),
            ]
        ),
        y_score=np.concatenate([johnstone_stats_null, johnstone_stats_experiment]),
    )
    # plt.plot(tpr_shuffled, fpr_shuffled, label="Johnstone")
    johnstone_df = pd.DataFrame({"TPR": tpr_shuffled, "FPR": fpr_shuffled})
    # johnstone_df['method'] = "Johnstone, 2008"
    # johnstone_df['method'] = "Johnstone"
    johnstone_df["method"] = r"$\emph{Johnstone}$"

    ######## sLED ##########
    sled_stats_experiment = np.load(
        "../out/sled/test_stats_experiment_p{}.npy".format(p)
    )
    sled_stats_null = np.load("../out/sled/test_stats_shuffled_p{}.npy".format(p))

    tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
        y_true=np.concatenate(
            [np.zeros(len(sled_stats_null)), np.ones(len(sled_stats_experiment))]
        ),
        y_score=np.concatenate([sled_stats_null, sled_stats_experiment]),
    )
    # plt.plot(tpr_shuffled, fpr_shuffled, label="sLED")
    sled_df = pd.DataFrame({"TPR": tpr_shuffled, "FPR": fpr_shuffled})
    # sled_df['method'] = "Zhu et al., 2017"
    # sled_df['method'] = "Zhu"
    sled_df["method"] = r"$\emph{Zhu}$"

    # plt.legend(prop={"size": 20})
    # plt.xlabel("TPR")
    # plt.ylabel("FPR")
    # plt.plot([0, 1], [0, 1], "-", color="black")

    plot_df = pd.concat([cai_df, cplvm_df, johnstone_df, sled_df], axis=0)
    # g = sns.lineplot(data=plot_df, x="TPR", y="FPR", style="method", color="black", ci=95, err_style="band")
    g = sns.lineplot(
        data=plot_df, x="TPR", y="FPR", hue="method", ci=95, err_style="band"
    )
    g.legend_.set_title(None)
    plt.legend(prop={"size": 20})

    # import ipdb; ipdb.set_trace()

    plt.tight_layout()
plt.savefig("../out/roc_comparison.png")
plt.show()
import ipdb

ipdb.set_trace()
