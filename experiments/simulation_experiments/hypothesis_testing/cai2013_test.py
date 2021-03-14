import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, invwishart


def cai_test(X1, X2, alpha=0.05, verbose=False):

    assert X1.shape[1] == X2.shape[1]
    n1, p = X1.shape
    n2 = X2.shape[0]

    ## Compute sample covariance
    sigma1 = np.cov(X1.T)
    sigma2 = np.cov(X2.T)

    ## Compute variance normalization factors
    theta1 = np.empty((p, p))
    theta2 = np.empty((p, p))

    X1_centered = X1 - np.mean(X1, axis=0)
    X2_centered = X2 - np.mean(X2, axis=0)
    for ii in range(p):
        # for jj in range(p):
        # theta1[ii, jj] = 1 / n1 * np.sum((np.multiply(X1_centered[:, ii], X1_centered[:, jj]) - sigma1[ii, jj])**2)
        # theta2[ii, jj] = 1 / n2 * np.sum((np.multiply(X2_centered[:, ii], X2_centered[:, jj]) - sigma2[ii, jj])**2)
        # import ipdb; ipdb.set_trace()
        theta1[ii, :] = (
            1
            / n1
            * np.sum(
                (
                    np.multiply(np.expand_dims(X1_centered[:, ii], axis=1), X1_centered)
                    - sigma1[ii, :]
                )
                ** 2,
                axis=0,
            )
        )
        theta2[ii, :] = (
            1
            / n2
            * np.sum(
                (
                    np.multiply(np.expand_dims(X2_centered[:, ii], axis=1), X2_centered)
                    - sigma2[ii, :]
                )
                ** 2,
                axis=0,
            )
        )

    ## Compute test statistic
    M = (sigma1 - sigma2) ** 2 / (theta1 / n1 + theta2 / n2)
    Mn = np.max(M)

    ## Compute test threshold
    q_alpha = -np.log(8 * np.pi) - 2 * np.log(-1 * np.log(1 - alpha))
    thresh = q_alpha + 4 * np.log(p) - np.log(np.log(p))

    ## Make decision
    is_reject = Mn > thresh

    if verbose:
        print("Test statistic: {}".format(Mn))
        print("Threshold: {}".format(thresh))
        print("Reject? {}".format(is_reject))

    return Mn, is_reject


# Generates covariance based on method 4 here: https://www.amcs.upenn.edu/sites/default/files/Two-Sample%20Covariance%20Matrix%20Testing%20and%20Support%20Recovery%20in%20High-Dimensional%20and%20Sparse%20Settings.pdf
def generate_random_cov(p):
    omega = np.random.uniform(low=1, high=5, size=p)
    O = np.diag(omega)
    Delta = np.empty((p, p))
    for ii in range(p):
        for jj in range(p):
            Delta[ii, jj] = np.power(-1, ii + 1 + jj + 1) * np.power(
                0.4, np.abs(ii + 1 - jj + 1) ** 0.1
            )

    cov = O @ Delta @ O
    return cov


if __name__ == "__main__":

    n1, n2 = 2000, 2000
    p = 10
    mu1, mu2 = np.zeros(p), np.zeros(p)
    cov1, cov2 = generate_random_cov(p), generate_random_cov(p)

    X1 = multivariate_normal(mean=mu1, cov=cov1).rvs(n1)
    X2 = multivariate_normal(mean=mu2, cov=cov1).rvs(n2)

    is_reject = cai_test(X1, X2, alpha=0.05, verbose=True)
    import ipdb

    ipdb.set_trace()
