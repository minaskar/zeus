import numpy as np
import statsmodels.api as sm


def autocorrtime(x, c=5.0):
    """
    Compute the integrated autocorrelation time of a chain according to the recipe of Sokal.

    Args:
        x (ndarray): 1D chain to be analysed.
        c (float): Parameter c of Sokal's recipe (default values is 5.0).

    Returns:
        The autocorrelation time of the chain.
    """

    acf = sm.tsa.stattools.acf(x, nlags=len(x), fft=True)

    def tau(M):
        return 1.0 + 2.0 + np.sum(acf[1:M])

    M = 2
    while M < c * tau(M):
        M += 1

    return tau(M)
