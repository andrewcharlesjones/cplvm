# Contrastive Poisson latent variable models

This repo contains models and algorithms for contrastive Poisson latent variable models (CPLVM). Given a count-based foreground dataset and a count-based backround dataset, the CPLVM is designed to find structure and variation that is enriched in the foreground relative to the background.

The accompanying paper can be found here: XXX.

## Installation

Run the following commands in a terminal to install the package.
```
git clone git@github.com:andrewcharlesjones/cplvm.git
cd cplvm
python setup.py install
```

You should then be able to import the model in Python as follows:
```python
from cplvm import CPLVM
```
## Example

Here we show a simple example of fitting the CPLVM. First, let's generate some data that has two subgroups in the foreground. To be able to specify the covariance of Poisson-distributed data, we'll use a Gaussian copula with Poisson marginal likelihoods.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

n, m = 1000, 1000
p = 2

# Generate latent variables
cov_mat = np.array([
    [2.7, 2.6],
    [2.6, 2.7]])
Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=n)

# Pass through standard normal CDF
Z_tilde = norm.cdf(Z)

# Inverse of observed distribution function
X = poisson.ppf(q=Z_tilde, mu=10)
X += 4

Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m//2)
Z_tilde = norm.cdf(Z)
Y1 = poisson.ppf(q=Z_tilde, mu=10)
Y1[:, 0] += 8

Z = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_mat, size=m//2)
Z_tilde = norm.cdf(Z)
Y2 = poisson.ppf(q=Z_tilde, mu=10)
Y2[:, 1] += 8
Y = np.concatenate([Y1, Y2], axis=0)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], label="Background", color="gray", alpha=0.4)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], label="Foreground group 1", color="green", alpha=0.4)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], label="Foreground group 2", color="orange", alpha=0.4)
plt.show()
```

![toyexample](./experiments/simulation_experiments/toy_example/out/toy_example.png)
