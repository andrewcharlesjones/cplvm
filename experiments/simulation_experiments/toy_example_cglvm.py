import ipdb
import sys
sys.path.append("../models")
from clvm_tfp_poisson_link import fit_model as fit_clvm_link
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import poisson

import matplotlib
font = {'size': 30}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

# Covariance of RVs
cov_mat = np.array([
    [2.7, 2.5],
    [2.5, 2.7]])

n, m = 1000, 1000
p = 2

# Generate latent variables
Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=n)

# Pass through standard normal CDF
Z_tilde = norm.cdf(Z)

# Inverse of observed distribution function
X = poisson.ppf(q=Z_tilde, mu=10)
X += 8

Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=n//2)
Z_tilde = norm.cdf(Z)
Y1 = poisson.ppf(q=Z_tilde, mu=10)
Y1[:, 0] += 15

Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=n//2)
Z_tilde = norm.cdf(Z)
Y2 = poisson.ppf(q=Z_tilde, mu=10)
Y2[:, 1] += 15
Y = np.concatenate([Y1, Y2], axis=0)


# Fit model
model_dict = fit_clvm_link(
    X.T, Y.T, 1, 1, compute_size_factors=True, is_H0=False)

W = model_dict['qw_mean'].numpy()
S = model_dict['qs_mean'].numpy()


# Plot
plt.figure(figsize=(7, 5))
plt.xlim([-3, 50])
plt.ylim([-3, 50])

plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray")
axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = S[1, 0] / S[0, 0] * x_vals
plt.plot(x_vals, y_vals, '--', label="S", color="black", linewidth=3)


Y_mean = np.mean(Y, axis=0)
W_slope = W[1, 0] / W[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
plt.scatter(Y[:, 0], Y[:, 1], label="Foreground", color="green")
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W", color="red", linewidth=3)



plt.legend(prop={'size': 20})
plt.title("CGLVM")
plt.savefig("./out/toy_example_cglvm.png")

plt.show()
ipdb.set_trace()

