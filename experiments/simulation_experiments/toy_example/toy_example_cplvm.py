import ipdb
import sys
sys.path.append("../../cplvm")
from cplvm import CPLVM
# from clvm_tfp_poisson_link import fit_model as fit_clvm_link
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from pcpca import PCPCA, CPCA

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

n, m = 1000, 1000
p = 2


# DATA FOR NONNEGATIVE CLVM

xs = np.random.normal(20, 5, size=n).astype(int)
ys = np.random.poisson(4, size=n)
X = np.vstack([xs, ys]).T

xs = np.random.normal(20, 5, size=n//2).astype(int)
ys = np.random.poisson(4, size=n//2)
Y1 = np.vstack([xs, ys]).T

ys = np.random.normal(20, 5, size=n//2).astype(int)
xs = np.random.poisson(4, size=n//2)
Y2 = np.vstack([xs, ys]).T
Y = np.concatenate([Y1, Y2], axis=0)




plt.figure(figsize=(28, 7))

############ PCA ############

pcpca = PCPCA(gamma=0, n_components=1)
pcpca.fit((Y - Y.mean(0)).T, (X - X.mean(0)).T)

plt.subplot(141)
# Plot
plt.xlim([-3, 38])
plt.ylim([-3, 38])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.4)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.4)



Y_mean = np.mean(Y, axis=0)
W_slope = pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope

axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W", color="red", linewidth=3)
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.legend(prop={'size': 20})

plt.title("PCA")


############ CPCA ############

cpca = CPCA(gamma=200, n_components=1)
cpca.fit((Y - Y.mean(0)).T, (X - X.mean(0)).T)

plt.subplot(142)
# Plot
plt.xlim([-3, 38])
plt.ylim([-3, 38])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.4)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.4)


Y_mean = np.mean(Y, axis=0)
W_slope = cpca.W[1, 0] / cpca.W[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope

axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W", color="red", linewidth=3)
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.legend(prop={'size': 20})

plt.title("CPCA")


############ PCPCA ############

pcpca = PCPCA(gamma=200, n_components=1)
pcpca.fit((Y - Y.mean(0)).T, (X - X.mean(0)).T)

plt.subplot(143)
# Plot
plt.xlim([-3, 38])
plt.ylim([-3, 38])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.4)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.4)


Y_mean = np.mean(Y, axis=0)
W_slope = pcpca.W_mle[1, 0] / pcpca.W_mle[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope

axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W", color="red", linewidth=3)
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.legend(prop={'size': 20})

plt.title("PCPCA")
# plt.show()


############ CPLVM ############

plt.subplot(144)


# Fit model
model_dict = fit_clvm_nonnegative(
    X.T, Y.T, 1, 1, compute_size_factors=True, is_H0=False)

W = np.exp(model_dict['qw_mean'].numpy() + model_dict['qw_stddv'].numpy()**2)
S = np.exp(model_dict['qs_mean'].numpy() + model_dict['qs_stddv'].numpy()**2)


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
plt.plot(x_vals, y_vals, '--', label="S", color="black", linewidth=3)


# Y_mean = np.mean(Y, axis=0)
W_slope = W[1, 0] / W[0, 0]
W_intercept = 0
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.4)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.4)

W_slope = W[1, 0] / W[0, 0]
axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = W_slope * x_vals
plt.plot(x_vals, y_vals, '--', label="W", color="red", linewidth=3)
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")


plt.legend(prop={'size': 20})
plt.title("CPLVM")
plt.tight_layout()
plt.savefig("./out/toy_example_cplvm.png")


plt.show()
# ipdb.set_trace()
