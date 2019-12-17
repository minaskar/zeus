import numpy as np
import pytest

from zeus.autocorr import _autocorr_time_1d

def get_chain(seed=42, ndim=5, N=100000):
    np.random.seed(seed)
    a = 0.9
    x = np.empty((N, ndim))
    x[0] = np.zeros(ndim)
    for i in range(1, N):
        x[i] = x[i - 1] * a + np.random.rand(ndim)
    return x

def test_1d(seed=42):
    walker0 = get_chain(seed, ndim=1)
    walker1 = get_chain(seed+1, ndim=1)
    chain = np.hstack((walker0,walker1)).T
    tau = _autocorr_time_1d(chain)
    assert tau < 20.0
