import pytest
import numpy as np

from zeus import samples


def test_chain(seed=42):
    np.random.seed(seed)
    s = samples()
    nsteps =np.random.randint(200,400)
    ndim = np.random.randint(2,200)
    nwalkers = 2 * ndim
    for i in range(nsteps):
        x = np.random.rand(nwalkers,ndim)
        s.append(x)
    assert np.shape(s.chain) == (nwalkes,nsteps,ndim)


def test_flatten(seed=42):
    np.random.seed(seed)
    s = samples()
    nsteps =np.random.randint(200,400)
    ndim = np.random.randint(2,200)
    nwalkers = 2 * ndim
    for i in range(nsteps):
        x = np.random.rand(nwalkers,ndim)
        s.append(x)
    burn = np.random.randint(2,100)
    thin = np.random.randint(1,10)
    assert np.shape(s.flatten(burn,thin)) == ((nsteps-burn)*nwalkers/thin,ndim)
