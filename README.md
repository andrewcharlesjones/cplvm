# [Contrastive Poisson latent variable models](https://arxiv.org/abs/2102.06731)
![Build Status](https://github.com/andrewcharlesjones/cplvm/actions/workflows/cplvm.yml/badge.svg)

This repo contains models and algorithms for contrastive Poisson latent variable models (CPLVM). Given a count-based foreground dataset and a count-based backround dataset, the CPLVM is designed to find structure and variation that is enriched in the foreground relative to the background.

The accompanying paper can be found here: https://arxiv.org/abs/2102.06731.

## Installation

See requirements.txt for a list installed packages and their versions. The package can be installed with pip.
```bash
pip install cplvm
```

You should then be able to import the model in Python as follows:
```python
from cplvm import CPLVM
```
## Example

Here we show a simple example of fitting the CPLVM. First, let's load some data that has two subgroups in the foreground. To be able to specify the covariance of Poisson-distributed data, these data were generated using a Gaussian copula with Poisson marginal likelihoods.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

X = pd.read_csv("./data/toy/toy_background.csv", header=None).values
Y = pd.read_csv("./data/toy/toy_foreground.csv", header=None).values

n, m = X.shape[0], Y.shape[0]
assert X.shape[1] == Y.shape[1]
p = X.shape[1]

# Plot the data
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.4)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.4)
plt.show()
```

<p align="center">
	<img src="./experiments/simulation_experiments/toy_example/out/toy_data.png" width="500">
</p>

Now, we fit the CPLVM.

```python
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
```

Let's inspect the fitted loadings matrices. To do this, let's take the mean of the variational distribution for each component. The variational families are log-normal.

```python
# Foreground-specific loadings
W_mean = model_output["approximate_model"].qw_mean.numpy()
W_stddev = model_output["approximate_model"].qw_stddv.numpy()
W = np.exp(W_mean + W_stddev ** 2)

# Shared loadings
S_mean = model_output["approximate_model"].qs_mean.numpy()
S_stddev = model_output["approximate_model"].qs_stddv.numpy()
S = np.exp(S_mean + S_stddev ** 2)
```

Now we can visualize each component as a 1D line.

```python
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
plt.show()
```

<p align="center">
	<img src="./experiments/simulation_experiments/toy_example/out/cplvm_toy.png" />
</p>

For context, we can visualize the analogous loadings for a suite of other related methods.

<p align="center">
	<img src="./experiments/simulation_experiments/toy_example/out/toy_example.png" />
</p>










