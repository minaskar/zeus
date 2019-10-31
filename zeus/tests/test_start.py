#import sys, os
#myPath = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, myPath + '/../')

import numpy as np
import pytest

#from start import jitter
import zeus.start as jitter

def test_case1(seed=123):
    np.random.seed(seed)
    nwalkers = np.random.randint(1,300)
    ndim = np.random.randint(1,300)
    x = np.random.rand(nwalkers, ndim)
    x_jittered = jitter(x, nwalkers, ndim)
    assert np.allclose(x_jittered,x)
