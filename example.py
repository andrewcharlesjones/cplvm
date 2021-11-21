from cplvm import CPLVM, CPLVMLogNormalApprox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

############ Load data ############

X = pd.read_csv("./data/toy/toy_background.csv", header=None).values
Y = pd.read_csv("./data/toy/toy_foreground.csv", header=None).values

n, m = X.shape[0], Y.shape[0]
assert X.shape[1] == Y.shape[1]
p = X.shape[1]

# Plot the data
plt.figure(figsize=(12, 7))
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
plt.scatter(
    Y[: m // 2, 0], Y[: m // 2, 1], label="Foreground group 1", color="green", alpha=0.4
)
plt.scatter(
    Y[m // 2 :, 0],
    Y[m // 2 :, 1],
    label="Foreground group 2",
    color="orange",
    alpha=0.4,
)
plt.legend(bbox_to_anchor=(1.2, 1.05), fontsize=20)
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.title("Toy data")
plt.tight_layout()
plt.savefig("./experiments/simulation_experiments/toy_example/out/toy_data.png")
# plt.show()
plt.close()


############ Fit CPLVM ############

# Set up CPLVM
cplvm = CPLVM(
    k_shared=1,
    k_foreground=2,
    compute_size_factors=True,
    offset_term=False)

# Set up approximate model
approx_model = CPLVMLogNormalApprox(
    X.T, 
    Y.T, 
    k_shared=1, 
    k_foreground=2, 
    compute_size_factors=True, 
    offset_term=False
)

# Fit model
model_output = cplvm.fit_model_vi(
    X.T,
    Y.T,
    approximate_model=approx_model,
)

## Extract parameters

# Foreground-specific loadings
W_mean = model_output["approximate_model"].qw_mean.numpy()
W_stddev = model_output["approximate_model"].qw_stddv.numpy()
W = np.exp(W_mean + W_stddev ** 2)

# Shared loadings
S_mean = model_output["approximate_model"].qs_mean.numpy()
S_stddev = model_output["approximate_model"].qs_stddv.numpy()
S = np.exp(S_mean + S_stddev ** 2)


############ Visualize result ############

## Plot results
plt.figure(figsize=(12, 7))

# Plot data
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
plt.scatter(
    Y[: m // 2, 0], Y[: m // 2, 1], label="Foreground group 1", color="green", alpha=0.4
)
plt.scatter(
    Y[m // 2 :, 0],
    Y[m // 2 :, 1],
    label="Foreground group 2",
    color="orange",
    alpha=0.4,
)

axes = plt.gca()
xlims = np.array(axes.get_xlim())
ylims = np.array(axes.get_ylim())

# Plot S
X_mean = np.mean(X, axis=0)
S_slope = S[1, 0] / S[0, 0]
S_intercept = X_mean[1] - X_mean[0] * S_slope
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = S_slope * x_vals + S_intercept
plt.plot(x_vals, y_vals, "--", label="S", color="black", linewidth=3)


# Plot W1
Y_mean = np.mean(Y, axis=0)

W_slope = W[1, 0] / W[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, "--", label="W1", color="red", linewidth=3)

# Plot W2
W_slope = W[1, 1] / W[0, 1]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, "--", label="W2", color="blue", linewidth=3)

plt.xlim(xlims)
plt.ylim(ylims)


plt.legend(bbox_to_anchor=(1.2, 1.05), fontsize=20)

plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.title("CPLVM")
plt.tight_layout()
plt.savefig("./experiments/simulation_experiments/toy_example/out/cplvm_toy.png")
plt.show()


import ipdb

ipdb.set_trace()
