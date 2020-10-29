import pytest
import numpy as np

from zeus import samples


def test_chain(seed=42):
    np.random.seed(seed)
    nsteps =np.random.randint(200,400)
    ndim = np.random.randint(2,200)
    nwalkers = 2 * ndim
    s = samples(ndim, nwalkers)
    s.extend(nsteps, None)
    for i in range(nsteps):
        x = np.random.rand(nwalkers,ndim)
        z = np.random.rand(nwalkers)
        s.save(x, z, None)
    assert np.shape(s.chain) == (nsteps,nwalkers,ndim)
    assert np.shape(s.logprob) == (nsteps,nwalkers)


def test_flatten(seed=42):
    np.random.seed(seed)
    nsteps =np.random.randint(200,400)
    ndim = np.random.randint(2,200)
    nwalkers = 2 * ndim
    s = samples(ndim,nwalkers)
    s.extend(nsteps, None)
    for i in range(nsteps):
        x = np.random.rand(nwalkers,ndim)
        z = np.random.rand(nwalkers)
        s.save(x, z, None)
    assert np.shape(s.flatten()) == (nsteps*nwalkers,ndim)
    burn = np.random.randint(2,100)
    thin = np.random.randint(1,10)
    assert np.shape(s.flatten(burn,thin)) == (np.ceil((nsteps-burn)/thin)*nwalkers,ndim)
    assert np.shape(s.flatten_logprob(burn,thin)) == (np.ceil((nsteps-burn)/thin)*nwalkers)


def test_multiple():
    for seed in range(10):
        test_chain(seed=seed)
        test_flatten(seed=seed)
