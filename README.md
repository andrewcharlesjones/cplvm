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

![toyexample](./experiments/simulation_experiments/toy_example/out/toy_data.png)


![toyexample](./experiments/simulation_experiments/toy_example/out/toy_example.png)
