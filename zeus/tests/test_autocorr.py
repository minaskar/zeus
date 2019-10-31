import numpy as np
import pytest

from zeus import autocorrtime

def get_chain(seed=42, ndim=5, N=100000):
    np.random.seed(seed)
    a = 0.9
    x = np.empty((N, ndim))
    x[0] = np.zeros(ndim)
    for i in range(1, N):
        x[i] = x[i - 1] * a + np.random.rand(ndim)
    return x

def test_1d(seed=42):
    chain = get_chain(seed, ndim=1)
    tau = autocorrtime(chain)
    assert tau < 15.0
