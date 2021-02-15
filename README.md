# Contrastive Poisson latent variable models

This repo contains models and algorithms for contrastive Poisson latent variable models (CPLVM). Given a count-based foreground dataset and a count-based backround dataset, the CPLVM is designed to find structure and variation that is enriched in the foreground relative to the background.

The accompanying paper can be found here: XXX.

<!-- <img src="https://latex.codecogs.com/svg.latex?O_t=\text { Onset event at time bin } t " />  -->
![equation](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign%7D%20%5Cmathbf%7By%7D_i%20%7C%20%5Cmathbf%7Bz%7D_i%20%26%20%5Csim%20%5Ctext%7BPoisson%7D%5Cleft%28%5Calpha_i%5E%7B%5Ctext%7Bb%7D%7D%20%5Cboldsymbol%7B%5Cdelta%7D%20%5Codot%20%5Cleft%28%5Cmathbf%7BS%7D%5E%5Ctop%20%5Cmathbf%7Bz%7D_i%5E%7B%5Ctext%7Bb%7D%7D%5Cright%29%5Cright%29%20%5C%5C%20%5Cmathbf%7Bx%7D_j%20%7C%20%5Cmathbf%7Bz%7D_j%2C%20%5Cmathbf%7Bt%7D_j%20%26%20%5Csim%20%5Ctext%7BPoisson%7D%5Cleft%28%5Calpha_j%5E%7B%5Ctext%7Bf%7D%7D%20%5Cleft%28%20%5Cmathbf%7BS%7D%5E%5Ctop%20%5Cmathbf%7Bz%7D_j%5E%7B%5Ctext%7Bf%7D%7D%20&plus;%20%5Cmathbf%7BW%7D%5E%5Ctop%20%5Cmathbf%7Bt%7D_j%20%5Cright%29%20%5Cright%29%20%5C%5C%20z_%7Bil%7D%5E%7B%5Ctext%7Bb%7D%7D%20%5Csim%20%5Ctext%7BGamma%7D%28%5Cgamma_1%2C%20%5Cbeta_1%29%26%2C%20%5C%3B%5C%3B%5C%3B%20z_%7Bjl%7D%5E%7B%5Ctext%7Bf%7D%7D%20%5Csim%20%5Ctext%7BGamma%7D%28%5Cgamma_2%2C%20%5Cbeta_2%29%2C%20%5C%3B%5C%3B%5C%3B%20t_%7Bjd%7D%20%5Csim%20%5Ctext%7BGamma%7D%28%5Cgamma_3%2C%20%5Cbeta_3%29%2C%20%5C%5C%20W_%7Bkd%7D%20%5Csim%20%5Ctext%7BGamma%7D%28%5Cgamma_4%2C%20%5Cbeta_4%29%26%2C%20%5C%3B%5C%3B%5C%3B%20S_%7Bjl%7D%20%5Csim%20%5Ctext%7BGamma%7D%28%5Cgamma_5%2C%20%5Cbeta_5%29%2C%20%5C%3B%5C%3B%5C%3B%20%5Cboldsymbol%7B%5Cdelta%7D%20%5Csim%20%5Ctext%7BLogNormal%7D%280%2C%20%5Ctextbf%7BI%7D_p%29%2C%20%5Cend%7Balign%7D)



## Installation

To install the package, run the following commands in a terminal from the directory in which you want to install it.
```bash
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

<p align="center">
	<img src="./experiments/simulation_experiments/toy_example/out/toy_data.png" />
</p>

Now, we fit the CPLVM.

```python
# Initialize the model
cplvm = CPLVM(k_shared=1, k_foreground=2)

# Fit model
model_output = cplvm.fit_model_vi(X.T, Y.T, compute_size_factors=True, is_H0=False, offset_term=False)
```

Let's inspect the fitted loadings matrices. To do this, let's take the mean of the variational distribution for each component. The variational families are log-normal.

```python
# Shared loadings
S = np.exp(model_output['qs_mean'].numpy() + model_output['qs_stddv'].numpy()**2)

# Foreground-specific loadings
W = np.exp(model_output['qw_mean'].numpy() + model_output['qw_stddv'].numpy()**2)
```

Now we can visualize each component as a 1D line.

```python
# Plot data
plt.scatter(X[:, 0], X[:, 1], color="gray", alpha=0.8)
plt.scatter(Y[:m//2, 0], Y[:m//2, 1], color="green", alpha=0.8)
plt.scatter(Y[m//2:, 0], Y[m//2:, 1], color="orange", alpha=0.8)

X_mean = np.mean(X, axis=0)
Y_mean = np.mean(Y, axis=0)

# Plot S
S_slope = S[1, 0] / S[0, 0]
S_intercept = X_mean[1] - X_mean[0] * S_slope
axes = plt.gca()
xlims = np.array(axes.get_xlim())
x_vals = np.linspace(xlims[0], xlims[1], 100)
y_vals = S_slope * x_vals + S_intercept
plt.plot(x_vals, y_vals, '--', label="S", color="black", linewidth=3)

# Plot first W component
W_slope = W[1, 0] / W[0, 0]
W_intercept = Y_mean[1] - Y_mean[0] * W_slope
axes = plt.gca()
ylims = np.array(axes.get_ylim())
y_vals = np.linspace(ylims[0], ylims[1], 100)
y_vals = W_slope * x_vals + W_intercept
plt.plot(x_vals, y_vals, '--', label="W1", color="red", linewidth=3)

# Plot second W component
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
plt.show()
```

<p align="center">
	<img src="./experiments/simulation_experiments/toy_example/out/cplvm_toy.png" />
</p>

For context, we can visualize the analogous loadings for a suite of other related methods.

<p align="center">
	<img src="./experiments/simulation_experiments/toy_example/out/toy_example.png" />
</p>










