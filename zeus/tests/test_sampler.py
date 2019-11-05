import numpy as np
import pytest

import zeus

def logp(x):
    return -0.5 * np.sum((x-1.0)**2.0)


def test_mean(logp=logp,seed=42):
    ndim = np.random.randint(2,5)
    nwalkers = 2 * ndim
    nsteps = np.random.randint(500,1000)
    sampler = zeus.sampler(logp,nwalkers,ndim,verbose=False)
    start = np.random.rand(nwalkers,ndim)
    sampler.run(start,nsteps)
    assert np.all(np.abs(np.mean(sampler.flatten(),axis=0)-1.0) < 0.1)


def test_std(logp=logp,seed=42):
    ndim = np.random.randint(2,5)
    nwalkers = 2 * ndim
    nsteps = np.random.randint(500,1000)
    sampler = zeus.sampler(logp,nwalkers,ndim,verbose=False)
    start = np.random.rand(nwalkers,ndim)
    sampler.run(start,nsteps)
    assert np.all(np.abs(np.std(sampler.flatten(),axis=0)-1.0) < 0.1)
