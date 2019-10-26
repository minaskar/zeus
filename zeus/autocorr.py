import numpy as np
import statsmodels.api as sm


def autocorrtime(x, c=5):

    acf = sm.tsa.stattools.acf(x, nlags=len(x), fft=True)

    def tau(M):
        return 1.0 + 2.0 + np.sum(acf[1:M])

    M = 2
    while M < c * tau(M):
        M += 1

    return tau(M)
