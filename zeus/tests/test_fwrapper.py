import numpy as np
import pytest

#from zeus import fwrapper
from zeus.fwrapper import *

def func0(x):
    return - 0.5 * np.sum(x**2.0)

def func1(x, mu):
    return - 0.5 * np.sum((x-mu)**2.0)

def func2(x, mu, ivar):
    return - 0.5 * np.sum(ivar*(x-mu)**2.0)


def test_none(func=func0,seed=42):
    np.random.seed(seed)
    args = None
    kwargs = None
    wrapped = _FunctionWrapper(func, args, kwargs)
    ndim = np.random.randint(2,200)
    x = np.random.rand(ndim)
    assert np.allclose(wrapped(x),func(x))
