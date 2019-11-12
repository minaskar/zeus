#import sys, os
#myPath = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, myPath + '/../')

import numpy as np
import pytest

#from start import jitter
from zeus import jitter


def test_case1(seed=42):
    np.random.seed(seed)
    nwalkers = np.random.randint(2,300)
    ndim = np.random.randint(1,300)
    x = np.random.rand(nwalkers, ndim)
    x_jittered = jitter(x, nwalkers, ndim)
    assert np.allclose(x_jittered,x)


def test_case2(seed=42):
    np.random.seed(seed)
    nwalkers = np.random.randint(2,300)
    ndim = np.random.randint(1,300)
    x = np.random.rand(ndim, nwalkers)
    x_jittered = jitter(x, nwalkers, ndim)
    assert np.allclose(x_jittered,x.T)


def test_case3(seed=42):
    np.random.seed(seed)
    nwalkers = np.random.randint(2,300)
    ndim = np.random.randint(1,300)
    x = np.random.rand(ndim)
    x_jittered = jitter(x, nwalkers, ndim)
    assert np.shape(x_jittered)==(nwalkers,ndim)
