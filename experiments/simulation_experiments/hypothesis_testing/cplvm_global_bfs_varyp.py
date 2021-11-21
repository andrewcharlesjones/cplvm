import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import poisson
from scipy.special import logsumexp
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox

import functools
import warnings
import tensorflow.compat.v2 as tf

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import matplotlib

font = {"size": 18}
matplotlib.rc("font", **font)

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    p_list = [10, 100, 1000]
    NUM_REPEATS = 50
    results_alternative = np.empty((NUM_REPEATS, len(p_list)))
    results_null = np.empty((NUM_REPEATS, len(p_list)))

    for ii in range(NUM_REPEATS):

        num_datapoints_x = 100
        num_datapoints_y = 100
        latent_dim_shared = 3
        latent_dim_foreground = 3

        bfs_experiment = []
        # bfs_control = []
        bfs_shuffled = []

        for jj, data_dim in enumerate(p_list):

            cplvm_for_data = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )

            concrete_clvm_model = functools.partial(
                cplvm_for_data.model,
                data_dim=data_dim,
                num_datapoints_x=num_datapoints_x,
                num_datapoints_y=num_datapoints_y,
                counts_per_cell_X=1,
                counts_per_cell_Y=1,
                is_H0=False,
                offset_term=True,
            )

            model = tfd.JointDistributionCoroutineAutoBatched(concrete_clvm_model)

            deltax, sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()
            # sf_x, sf_y, s, zx, zy, w, ty, X_sampled, Y_sampled = model.sample()

            X, Y = X_sampled.numpy(), Y_sampled.numpy()

            ## Run H0 and H1 models on data
            cplvm = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )
            approx_model_H0 = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                offset_term=True,
                is_H0=True,
            )
            approx_model_H1 = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                offset_term=True,
                is_H0=False,
            )
            H1_results = cplvm._fit_model_vi(
                X, Y, approx_model_H1, offset_term=True, is_H0=False
            )
            H0_results = cplvm._fit_model_vi(
                X, Y, approx_model_H0, offset_term=True, is_H0=True
            )

            H1_elbo = (
                -1
                * H1_results["loss_trace"][-1].numpy()
                / (num_datapoints_x + num_datapoints_y)
            )

            H0_elbo = (
                -1
                * H0_results["loss_trace"][-1].numpy()
                / (num_datapoints_x + num_datapoints_y)
            )

            curr_bf = H1_elbo - H0_elbo
            print("p: {0: <10} BF treatment: {1: .2f}".format(data_dim, curr_bf))
            bfs_experiment.append(curr_bf)
            results_alternative[ii, jj] = curr_bf

            ### Shuffle background and foreground labels

            all_data = np.concatenate([X, Y], axis=1)
            shuffled_idx = np.random.permutation(
                np.arange(num_datapoints_x + num_datapoints_y)
            )
            x_idx = shuffled_idx[:num_datapoints_x]
            y_idx = shuffled_idx[num_datapoints_x:]
            X = all_data[:, x_idx]
            Y = all_data[:, y_idx]

            ## Run H0 and H1 models on data
            ## Run H0 and H1 models on data
            cplvm = CPLVM(
                k_shared=latent_dim_shared, k_foreground=latent_dim_foreground
            )
            approx_model_H0 = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                offset_term=True,
                is_H0=True,
            )
            approx_model_H1 = CPLVMLogNormalApprox(
                X,
                Y,
                latent_dim_shared,
                latent_dim_foreground,
                offset_term=True,
                is_H0=False,
            )
            H1_results = cplvm._fit_model_vi(
                X, Y, approx_model_H1, offset_term=True, is_H0=False
            )
            H0_results = cplvm._fit_model_vi(
                X, Y, approx_model_H0, offset_term=True, is_H0=True
            )

            H1_elbo = (
                -1
                * H1_results["loss_trace"][-1].numpy()
                / (num_datapoints_x + num_datapoints_y)
            )

            H0_elbo = (
                -1
                * H0_results["loss_trace"][-1].numpy()
                / (num_datapoints_x + num_datapoints_y)
            )

            curr_bf = H1_elbo - H0_elbo
            print("p: {0: <10} BF shuffled: {1: .2f}".format(data_dim, curr_bf))
            bfs_shuffled.append(curr_bf)
            results_null[ii, jj] = curr_bf

            # bfs_control = np.array(bfs_control)[~np.isnan(bfs_control)]
            bfs_experiment = list(np.array(bfs_experiment)[~np.isnan(bfs_experiment)])
            bfs_shuffled = list(np.array(bfs_shuffled)[~np.isnan(bfs_shuffled)])
            # tpr_true, fpr_true, thresholds_true = roc_curve(y_true=np.concatenate([np.zeros(len(bfs_control)), np.ones(len(bfs_experiment))]), y_score=np.concatenate([bfs_control, bfs_experiment]))

            # print(np.concatenate([np.zeros(len(bfs_shuffled)), np.ones(len(bfs_experiment))]))
            # print(np.concatenate([bfs_shuffled, bfs_experiment]))
            # tpr_shuffled, fpr_shuffled, thresholds_shuffled = roc_curve(
            #     y_true=np.concatenate(
            #         [np.zeros(len(bfs_shuffled)), np.ones(len(bfs_experiment))]
            #     ),
            #     y_score=np.concatenate([bfs_shuffled, bfs_experiment]),
            # )

            # np.save("../out/cai/bfs_experiment_p{}.npy".format(data_dim), bfs_experiment)
            # np.save("../out/cai/bfs_shuffled_p{}.npy".format(data_dim), bfs_shuffled)

            # auc = roc_auc_score(
            #     y_true=np.concatenate(
            #         [np.zeros(len(bfs_shuffled)), np.ones(len(bfs_experiment))]
            #     ),
            #     y_score=np.concatenate([bfs_shuffled, bfs_experiment]),
            # )

        results_alternative_df = pd.melt(
            pd.DataFrame(results_alternative[: ii + 1, :], columns=p_list)
        )
        results_alternative_df["context"] = "Perturbed"
        results_null_df = pd.melt(
            pd.DataFrame(results_null[: ii + 1, :], columns=p_list)
        )
        results_null_df["context"] = "Shuffled null"

        results_df = pd.concat([results_alternative_df, results_null_df], axis=0)

        results_df.to_csv("../out/data_dimension_vs_ebfs.csv")

        plt.figure(figsize=(7, 7))
        g = sns.lineplot(
            data=results_df, x="variable", y="value", hue="context", err_style="bars"
        )
        g.legend_.set_title(None)
        plt.xlabel("Data dimension")
        plt.ylabel("EBF")
        plt.tight_layout()
        plt.savefig("../out/data_dimension_vs_ebfs.png")
        # plt.show()
        plt.close()
        # import ipdb; ipdb.set_trace()
