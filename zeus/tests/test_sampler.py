import numpy as np
import pytest

import zeus

def logp(x):
    return -0.5 * np.sum((x-1.0)**2.0)


def test_mean(logp=logp,seed=42):
    np.random.seed(seed)
    ndim = np.random.randint(2,5)
    nwalkers = 2 * ndim
    nsteps = np.random.randint(3000,5000)
    sampler = zeus.EnsembleSampler(nwalkers,ndim,logp,verbose=False)
    start = np.random.rand(nwalkers,ndim)
    sampler.run_mcmc(start,nsteps)
    assert np.all(np.abs(np.mean(sampler.get_chain(flat=True),axis=0)-1.0) < 0.1)
    assert np.all(np.isfinite(sampler.get_log_prob(flat=True)))
    assert np.all(np.isfinite(sampler.get_log_prob()))


def test_std(logp=logp,seed=42):
    np.random.seed(seed)
    ndim = np.random.randint(2,5)
    nwalkers = 2 * ndim
    nsteps = np.random.randint(3000,5000)
    sampler = zeus.EnsembleSampler(nwalkers,ndim,logp,verbose=False)
    start = np.random.rand(nwalkers,ndim)
    sampler.run_mcmc(start,nsteps)
    assert np.all(np.abs(np.std(sampler.get_chain(flat=True),axis=0)-1.0) < 0.1)


def test_ncall(seed=42):
    np.random.seed(seed)
    def loglike(theta):
        assert len(theta) == 5
        a = theta[:-1]
        b = theta[1:]
        loglike.ncalls += 1
        return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum()
    loglike.ncalls = 0

    ndim = 5
    nsteps = 100
    nwalkers = 2 * ndim
    sampler = zeus.EnsembleSampler(nwalkers,ndim,loglike,verbose=False)
    start = np.random.rand(nwalkers,ndim)
    sampler.run_mcmc(start,nsteps)
    
    assert loglike.ncalls == sampler.ncall + nwalkers




