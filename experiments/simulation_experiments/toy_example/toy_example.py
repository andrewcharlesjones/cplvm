import ipdb
from cplvm import CPLVM, CGLVM
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from pcpca import PCPCA, CPCA
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import adjusted_rand_score, silhouette_score
import pandas as pd

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


METHODS = ['PCA', 'NMF', 'CPCA', 'PCPCA', 'CGLVM', 'CPLVM']

############ Generate data ############

# Covariance of RVs
cov_mat = np.array([
    [2.7, 2.6],
    [2.6, 2.7]])

n, m = 1000, 1000
p = 2


############ Generate data ############

Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m//2)
Z_tilde = norm.cdf(Z)
Y1 = poisson.ppf(q=Z_tilde, mu=10)
Y1[:, 0] += 8

Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m//2)
Z_tilde = norm.cdf(Z)
Y2 = poisson.ppf(q=Z_tilde, mu=10)
Y2[:, 1] += 8
Y = np.concatenate([Y1, Y2], axis=0)

# # Generate latent variables
Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=n)

# Pass through standard normal CDF
Z_tilde = norm.cdf(Z)

# Inverse of observed distribution function
X = poisson.ppf(q=Z_tilde, mu=10)
X += 4

# Pre-standardized data
X_standardized = (X - X.mean(0)) / X.std(0)
Y_standardized = (Y - Y.mean(0)) / Y.std(0)

plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)
import ipdb; ipdb.set_trace()
plt.show()

# Labels of the foreground clusters
true_labels = np.zeros(m)
true_labels[m//2:] = 1


plt.figure(figsize=((len(METHODS)) / 2 * 7, 14))

############ PCA ############

pca = PCA(n_components=1)

pca.fit(np.concatenate([Y - Y.mean(0), X - X.mean(0)], axis=0))
W_pca = pca.components_.T

plt.subplot(2, (len(METHODS) + 1) / 2, 1)
# Plot
plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)

Y_mean = np.mean(Y, axis=0)
W_slope = W_pca[1, 0] / W_pca[0, 0]
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


############ NMF ############

nmf = NMF(n_components=2)
nmf.fit(np.concatenate([Y, X], axis=0))
W_nmf = nmf.components_.T

plt.subplot(2, (len(METHODS) + 1) / 2, 2)
# Plot
plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)

Y_mean = np.mean(Y, axis=0)

W_slope = W_nmf[1, 0] / (W_nmf[0, 0] + 1e-6)
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W1", color="red", linewidth=3)

W_slope = W_nmf[1, 1] / (W_nmf[0, 1] + 1e-6)
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W2", color="blue", linewidth=3)
# ipdb.set_trace()

plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.legend(prop={'size': 20})

plt.title("NMF")


############ CPCA ############

cpca = CPCA(gamma=0.9, n_components=1)
cpca.fit(Y_standardized.T, X_standardized.T)

plt.subplot(2, (len(METHODS) + 1) / 2, 3)

# Plot
plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)


Y_mean = np.mean(Y, axis=0)
W_slope = cpca.W_mle[1, 0] / cpca.W_mle[0, 0]
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

pcpca = PCPCA(gamma=0.9, n_components=1)
pcpca.fit(Y_standardized.T, X_standardized.T)

plt.subplot(2, (len(METHODS) + 1) / 2, 4)

# Plot
plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)


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


############ CGLVM ############

# Fit model
cglvm = CGLVM(k_shared=1, k_foreground=1)

model_dict = cglvm.git_model_vi(X.T, Y.T, compute_size_factors=True, is_H0=False)

W = model_dict['qw_mean'].numpy()
S = model_dict['qs_mean'].numpy()

zy = model_dict['qzy_mean'].numpy()
ty = model_dict['qty_mean'].numpy()

mu_y = model_dict['qmu_y_mean'].numpy()
sf_y = model_dict['qsize_factor_y_mean'].numpy()


plt.subplot(2, (len(METHODS) + 1) / 2, 5)

# Plot data
plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)

# Plot S
X_mean = np.mean(X, axis=0)
S_slope = S[1, 0] / S[0, 0]
S_intercept = X_mean[1] - X_mean[0] * S_slope
axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = S_slope * x_vals + S_intercept
plt.plot(x_vals, y_vals, '--', label="S", color="black", linewidth=3)


Y_mean = np.mean(Y, axis=0)

W_slope = W[1, 0] / W[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W", color="red", linewidth=3)


plt.title("CGLVM")
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.legend(prop={'size': 20})



############ CPLVM ############

# Fit model
cplvm = CGPLVM(k_shared=1, k_foreground=1)

model_dict = cplvm.git_model_vi(X.T, Y.T, compute_size_factors=True, is_H0=False)
model_dict = fit_clvm_nonnegative(
    X.T, Y.T, 1, 2, compute_size_factors=True, is_H0=False, offset_term=False)

W = np.exp(model_dict['qw_mean'].numpy() + model_dict['qw_stddv'].numpy()**2)
S = np.exp(model_dict['qs_mean'].numpy() + model_dict['qs_stddv'].numpy()**2)

zx = np.exp(model_dict['qzx_mean'].numpy() + model_dict['qzx_stddv'].numpy()**2)
zy = np.exp(model_dict['qzy_mean'].numpy() + model_dict['qzy_stddv'].numpy()**2)
ty = np.exp(model_dict['qty_mean'].numpy() + model_dict['qty_stddv'].numpy()**2)

sf_y = np.exp(model_dict['qsize_factors_y_mean'].numpy() + model_dict['qsize_factor_y_stddv'].numpy()**2)

plt.subplot(2, (len(METHODS) + 1) / 2, 6)

# Plot data
plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)

X_mean = np.mean(X, axis=0)
S_slope = S[1, 0] / S[0, 0]
S_intercept = X_mean[1] - X_mean[0] * S_slope
axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = S_slope * x_vals + S_intercept
plt.plot(x_vals, y_vals, '--', label="S", color="black", linewidth=3)


Y_mean = np.mean(Y, axis=0)

W_slope = W[1, 0] / W[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W1", color="red", linewidth=3)

W_slope = W[1, 1] / W[0, 1]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W2", color="blue", linewidth=3)

plt.title("CPLVM")
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.legend(prop={'size': 20})


plt.tight_layout()
plt.show()

