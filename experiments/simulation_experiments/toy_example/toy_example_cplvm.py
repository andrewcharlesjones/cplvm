from cplvm import CPLVM
from cplvm import CPLVMLogNormalApprox
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from pcpca import PCPCA, CPCA

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

n, m = 1000, 1000
p = 2


# DATA FOR NONNEGATIVE CLVM

xs = np.random.normal(20, 5, size=n).astype(int)
ys = np.random.poisson(4, size=n)
X = np.vstack([xs, ys]).T

xs = np.random.normal(20, 5, size=n // 2).astype(int)
ys = np.random.poisson(4, size=n // 2)
Y1 = np.vstack([xs, ys]).T

ys = np.random.normal(20, 5, size=n // 2).astype(int)
xs = np.random.poisson(4, size=n // 2)
Y2 = np.vstack([xs, ys]).T
Y = np.concatenate([Y1, Y2], axis=0)


plt.figure(figsize=(28, 7))

############ PCA ############

# pcpca = PCPCA(gamma=0, n_components=1)
# pcpca.fit((Y - Y.mean(0)).T, (X - X.mean(0)).T)

# plt.subplot(141)
# # Plot
# plt.xlim([-3, 38])
# plt.ylim([-3, 38])
# plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
# plt.scatter(
#     Y[: m // 2, 0], Y[: m // 2, 1], label="Foreground group 1", color="green", alpha=0.4
# )
# plt.scatter(
#     Y[m // 2 :, 0],
#     Y[m // 2 :, 1],
#     label="Foreground group 2",
#     color="orange",
#     alpha=0.4,
# )


# Y_mean = np.mean(Y, axis=0)
# W_slope = pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0]
# W_intercept = Y_mean[1] - Y_mean[0] * W_slope

# axes = plt.gca()
# xlims = np.array(axes.get_xlim())
# x_vals = np.linspace(xlims[0], xlims[1], 100)
# y_vals = W_slope * x_vals + W_intercept
# plt.plot(x_vals, y_vals, "--", label="W", color="red", linewidth=3)
# plt.xlabel("Gene 1")
# plt.ylabel("Gene 2")
# plt.legend(prop={"size": 20})

# plt.title("PCA")


# ############ CPCA ############

# cpca = CPCA(gamma=200, n_components=1)
# cpca.fit((Y - Y.mean(0)).T, (X - X.mean(0)).T)

# plt.subplot(142)
# # Plot
# plt.xlim([-3, 38])
# plt.ylim([-3, 38])
# plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
# plt.scatter(
#     Y[: m // 2, 0], Y[: m // 2, 1], label="Foreground group 1", color="green", alpha=0.4
# )
# plt.scatter(
#     Y[m // 2 :, 0],
#     Y[m // 2 :, 1],
#     label="Foreground group 2",
#     color="orange",
#     alpha=0.4,
# )


# Y_mean = np.mean(Y, axis=0)
# W_slope = cpca.W[1, 0] / cpca.W[0, 0]
# W_intercept = Y_mean[1] - Y_mean[0] * W_slope

# axes = plt.gca()
# xlims = np.array(axes.get_xlim())
# x_vals = np.linspace(xlims[0], xlims[1], 100)
# y_vals = W_slope * x_vals + W_intercept
# plt.plot(x_vals, y_vals, "--", label="W", color="red", linewidth=3)
# plt.xlabel("Gene 1")
# plt.ylabel("Gene 2")
# plt.legend(prop={"size": 20})

# plt.title("CPCA")


# ############ PCPCA ############

# pcpca = PCPCA(gamma=200, n_components=1)
# pcpca.fit((Y - Y.mean(0)).T, (X - X.mean(0)).T)

# plt.subplot(143)
# # Plot
# plt.xlim([-3, 38])
# plt.ylim([-3, 38])
# plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
# plt.scatter(
#     Y[: m // 2, 0], Y[: m // 2, 1], label="Foreground group 1", color="green", alpha=0.4
# )
# plt.scatter(
#     Y[m // 2 :, 0],
#     Y[m // 2 :, 1],
#     label="Foreground group 2",
#     color="orange",
#     alpha=0.4,
# )


# Y_mean = np.mean(Y, axis=0)
# W_slope = pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0]
# W_intercept = Y_mean[1] - Y_mean[0] * W_slope

# axes = plt.gca()
# xlims = np.array(axes.get_xlim())
# x_vals = np.linspace(xlims[0], xlims[1], 100)
# y_vals = W_slope * x_vals + W_intercept
# plt.plot(x_vals, y_vals, "--", label="W", color="red", linewidth=3)
# plt.xlabel("Gene 1")
# plt.ylabel("Gene 2")
# plt.legend(prop={"size": 20})

# plt.title("PCPCA")


############ CPLVM ############

plt.subplot(144)


# Fit model
cplvm = CPLVM(k_shared=2, k_foreground=1, compute_size_factors=True, offset_term=False)
approx_model = CPLVMLogNormalApprox(
    X.T, Y.T, k_shared=2, k_foreground=1, compute_size_factors=True, offset_term=False
)

model_dict = cplvm.fit_model_vi(
    X.T,
    Y.T,
    approximate_model=approx_model,
)


W_mean = model_dict["approximate_model"].qw_mean.numpy()
W_stddev = model_dict["approximate_model"].qw_stddv.numpy()

S_mean = model_dict["approximate_model"].qs_mean.numpy()
S_stddev = model_dict["approximate_model"].qs_stddv.numpy()

W = np.exp(W_mean + W_stddev ** 2)
S = np.exp(S_mean + S_stddev ** 2)


# Plot
plt.xlim([-3, 38])
plt.ylim([-3, 38])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)

S_slope = S[1, 0] / S[0, 0]
S_intercept = 0
axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = S_slope * x_vals
plt.plot(x_vals, y_vals, "--", label="S", color="black", linewidth=3)

S_slope = S[1, 1] / S[0, 1]
S_intercept = 0
axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = S_slope * x_vals
plt.plot(x_vals, y_vals, "--", color="black", linewidth=3)




# Y_mean = np.mean(Y, axis=0)
W_slope = W[1, 0] / W[0, 0]
W_intercept = 0
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

W_slope = W[1, 0] / W[0, 0]
axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = W_slope * x_vals
plt.plot(x_vals, y_vals, "--", label="W", color="red", linewidth=3)
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")


plt.legend(prop={"size": 20})
plt.title("CPLVM")
plt.tight_layout()
plt.savefig("./out/toy_example_cplvm.png")


plt.show()


import ipdb

ipdb.set_trace()
