import ipdb
import sys
sys.path.append("../../models")
from clvm_tfp_poisson_link import fit_model as fit_clvm_link
from clvm_tfp_poisson import fit_model as fit_clvm_nonnegative
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from pcpca import PCPCA, CPCA
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import adjusted_rand_score, silhouette_score

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True


METHODS = ['PCA', 'PPCA', 'NMF', 'CPCA', 'PCPCA', 'CGLVM', 'CPLVM']

############ Generate data ############

# Covariance of RVs
# cov_mat = np.array([
#     [2.7, 2.6],
#     [2.6, 2.7]])

n, m = 1000, 1000
p = 2

# # # Generate latent variables
# Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=n)

# # Pass through standard normal CDF
# Z_tilde = norm.cdf(Z)

# # Inverse of observed distribution function
# X = poisson.ppf(q=Z_tilde, mu=10)
# X += 4

# Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m//2)
# Z_tilde = norm.cdf(Z)
# Y1 = poisson.ppf(q=Z_tilde, mu=10)
# Y1[:, 0] += 8

# Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m//2)
# Z_tilde = norm.cdf(Z)
# Y2 = poisson.ppf(q=Z_tilde, mu=10)
# Y2[:, 1] += 8
# Y = np.concatenate([Y1, Y2], axis=0)

xs = np.random.normal(20, 5, size=n).astype(int)
ys = np.random.poisson(4, size=n)
X = np.vstack([xs, ys]).T

xs = np.random.normal(20, 5, size=n//2).astype(int)
ys = np.random.poisson(4, size=n//2)
Y1 = np.vstack([xs, ys]).T

ys = np.random.normal(20, 5, size=m//2).astype(int)
xs = np.random.poisson(4, size=m//2)
Y2 = np.vstack([xs, ys]).T
Y = np.concatenate([Y1, Y2], axis=0)

# Pre-standardized data
X_standardized = (X - X.mean(0)) / X.std(0)
Y_standardized = (Y - Y.mean(0)) / Y.std(0)

# Labels of the foreground clusters
true_labels = np.zeros(m)
true_labels[m//2:] = 1


plt.figure(figsize=((len(METHODS) + 1) / 2 * 7, 14))

############ PCA ############

pca = PCA(n_components=1)
pca.fit(np.concatenate([Y - Y.mean(0), X - X.mean(0)], axis=0))
W_pca = pca.components_.T
# import ipdb; ipdb.set_trace()

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

# Silhouette score
ss_pca = silhouette_score(X=pca.fit_transform(Y - Y.mean(0)), labels=true_labels)

# Reconstruction error
recons = pca.transform(Y - Y.mean(0)) @ pca.components_
recon_error_pca = np.mean(((Y - Y.mean(0)) - recons)**2)


############ PPCA ############

pcpca = PCPCA(gamma=0, n_components=1)
pcpca.fit(Y_standardized.T, X_standardized.T)

plt.subplot(2, (len(METHODS) + 1) / 2, 2)
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

plt.title("PPCA")

# Silhouette score
ppca_fg_projections = pcpca.transform((Y - Y.mean(0)).T, (X - X.mean(0)).T)[0].T
ss_ppca = silhouette_score(X=ppca_fg_projections, labels=true_labels)

# Reconstruction error
recons = (pcpca.W_mle @ pcpca.transform(Y_standardized.T, X_standardized.T)[0]).T
recons = recons * Y.std(0)
recon_error_ppca = np.mean(((Y - Y.mean(0)) - recons)**2)

############ NMF ############

nmf = NMF(n_components=2)
nmf.fit(np.concatenate([Y, X], axis=0))
W_nmf = nmf.components_.T
# import ipdb; ipdb.set_trace()

plt.subplot(2, (len(METHODS) + 1) / 2, 3)
# Plot
plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)

Y_mean = np.mean(Y, axis=0)

W_slope = W_nmf[1, 0] / W_nmf[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W1", color="red", linewidth=3)

W_slope = W_nmf[1, 1] / W_nmf[0, 1]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W2", color="blue", linewidth=3)

plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.legend(prop={'size': 20})

plt.title("NMF")

# Silhouette score
nmf_fg_projections = nmf.transform(Y)
ss_nmf = silhouette_score(X=nmf_fg_projections, labels=true_labels)

# Reconstruction error
recons = nmf.transform(Y) @ nmf.components_
recon_error_nmf = np.mean((Y - recons)**2)


############ CPCA ############

cpca = CPCA(gamma=0.9, n_components=1)
cpca.fit(Y_standardized.T, X_standardized.T)

plt.subplot(2, (len(METHODS) + 1) / 2, 4)
# Plot
plt.xlim([-3, 50])
plt.ylim([-3, 50])
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.8)


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

# Silhouette score
cpca_fg_projections = cpca.transform((Y - Y.mean(0)).T, (X - X.mean(0)).T)[0].T
ss_cpca = silhouette_score(X=cpca_fg_projections, labels=true_labels)

# Reconstruction error
recons = (pcpca.W_mle @ pcpca.transform(Y_standardized.T, X_standardized.T)[0]).T
recons = recons * Y.std(0)
recon_error_cpca = np.mean(((Y - Y.mean(0)) - recons)**2)


############ PCPCA ############

pcpca = PCPCA(gamma=0.9, n_components=1)
pcpca.fit(Y_standardized.T, X_standardized.T)

plt.subplot(2, (len(METHODS) + 1) / 2, 5)

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

# Silhouette score
pcpca_fg_projections = pcpca.transform((Y - Y.mean(0)).T, (X - X.mean(0)).T)[0].T
ss_pcpca = silhouette_score(X=pcpca_fg_projections, labels=true_labels)

# Reconstruction error
recons = (pcpca.W_mle @ pcpca.transform(Y_standardized.T, X_standardized.T)[0]).T
recons = recons * Y.std(0)
recon_error_pcpca = np.mean(((Y - Y.mean(0)) - recons)**2)


############ CGLVM ############

# Fit model
model_dict = fit_clvm_link(
    X.T, Y.T, 1, 1, compute_size_factors=True, is_H0=False)

W = model_dict['qw_mean'].numpy()
S = model_dict['qs_mean'].numpy()

zy = model_dict['qzy_mean'].numpy()
ty = model_dict['qty_mean'].numpy()

mu_y = model_dict['qmu_y_mean'].numpy()
sf_y = model_dict['qsize_factor_y_mean'].numpy()


plt.subplot(2, (len(METHODS) + 1) / 2, 6)

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


# Silhouette score
cglvm_fg_projections = ty.T
ss_cglvm = silhouette_score(X=cglvm_fg_projections, labels=true_labels)

# Reconstruction error
recons = np.exp(S @ zy + W @ ty + np.log(sf_y) + mu_y)
recon_error_cglvm = np.mean((Y - recons.T)**2)



############ CPLVM ############

# Fit model
model_dict = fit_clvm_nonnegative(
    X.T, Y.T, 1, 1, compute_size_factors=True, is_H0=False, offset_term=False)

W = np.exp(model_dict['qw_mean'].numpy() + model_dict['qw_stddv'].numpy()**2)
S = np.exp(model_dict['qs_mean'].numpy() + model_dict['qs_stddv'].numpy()**2)

zx = np.exp(model_dict['qzx_mean'].numpy() + model_dict['qzx_stddv'].numpy()**2)
zy = np.exp(model_dict['qzy_mean'].numpy() + model_dict['qzy_stddv'].numpy()**2)
ty = np.exp(model_dict['qty_mean'].numpy() + model_dict['qty_stddv'].numpy()**2)

sf_y = np.exp(model_dict['qsize_factors_y_mean'].numpy() + model_dict['qsize_factor_y_stddv'].numpy()**2)



# deltax = np.exp(model_dict['qdeltax_mean'].numpy() + model_dict['qdeltax_stddv'].numpy()**2)





plt.subplot(2, (len(METHODS) + 1) / 2, 7)

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
plt.plot(x_vals, y_vals, '--', label="W", color="red", linewidth=3)

# W_slope = W[1, 1] / W[0, 1]
# W_intercept = Y_mean[1] - Y_mean[0] * W_slope
# axes = plt.gca()
# ylims = np.array(axes.get_ylim())
# y_vals = np.linspace(ylims[0], ylims[1], 100)
# y_vals = W_slope * x_vals + W_intercept
# plt.plot(x_vals, y_vals, '--', label="W2", color="blue", linewidth=3)

plt.title("CPLVM")
plt.xlabel("Gene 1")
plt.ylabel("Gene 2")
plt.legend(prop={'size': 20})

# Silhouette score
cplvm_fg_projections = ty.T
ss_cplvm = silhouette_score(X=cplvm_fg_projections, labels=true_labels)

# Reconstruction error
recons = np.multiply(S @ zy + W @ ty, sf_y)
recon_error_cplvm = np.mean((Y - recons.T)**2)
# ipdb.set_trace()


METHODS = ['PCA', 'PPCA', 'NMF', 'CPCA', 'PCPCA', 'CGLVM', 'CPLVM']


plt.subplot(2, (len(METHODS) + 1) / 2, 8)

plt.bar(np.arange(len(METHODS)), [ss_pca, ss_ppca, ss_nmf, ss_cpca, ss_pcpca, ss_cglvm, ss_cplvm])
# plt.bar(np.arange(len(METHODS)), [recon_error_pca, recon_error_ppca, recon_error_nmf, recon_error_cpca, recon_error_pcpca, recon_error_cglvm, recon_error_cplvm])
plt.xticks(np.arange(len(METHODS)), labels=METHODS)
plt.xticks(rotation=90)
plt.ylabel("Silhouette score")
# ipdb.set_trace()


# plt.subplot(132)
# plt.title("Latent variables")
# plt.scatter(ty[0, :m//2], ty[1, :m//2], color="green", label="Foreground group 1")
# plt.scatter(ty[0, m//2:], ty[1, m//2:], color="orange", label="Foreground group 2")
# plt.xlabel("Latent dim 1")
# plt.ylabel("Latent dim 2")
# plt.legend(prop={'size': 10})

# plt.subplot(133)
# plt.title("Reconstruction")
# # X_reconstructed = np.multiply(S @ zx, deltax)
# X_reconstructed = S @ zx
# Y_reconstructed = S @ zy + W @ ty
# plt.scatter(Y_reconstructed[0, :m//2], Y_reconstructed[1, :m//2], color="green")
# plt.scatter(Y_reconstructed[0, m//2:], Y_reconstructed[1, m//2:], color="orange")
# plt.scatter(X_reconstructed[0, :], X_reconstructed[1, :], color="gray")
# plt.xlabel("Data dim 1")
# plt.ylabel("Data dim 2")



plt.tight_layout()
plt.savefig("./out/toy_example_cplvm_neg_corr.png")
plt.show()

# ipdb.set_trace()

