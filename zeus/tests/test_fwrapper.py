import numpy as np
import pytest

#from zeus import fwrapper
from zeus.fwrapper import _FunctionWrapper

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


def test_args1(func=func1,seed=42):
    np.random.seed(seed)
    ndim = np.random.randint(2,200)
    mu = np.random.rand(ndim)
    args = [mu]
    kwargs = None
    wrapped = _FunctionWrapper(func, args, kwargs)
    x = np.random.rand(ndim)
    assert np.allclose(wrapped(x),func(x,mu))


def test_args2(func=func2,seed=42):
    np.random.seed(seed)
    ndim = np.random.randint(2,200)
    mu = np.random.rand(ndim)
    ivar = 1.0 / np.random.rand(ndim)
    args = [mu, ivar]
    kwargs = None
    wrapped = _FunctionWrapper(func, args, kwargs)
    x = np.random.rand(ndim)
    assert np.allclose(wrapped(x),func(x,mu,ivar))


def test_kwargs1(func=func1,seed=42):
    np.random.seed(seed)
    ndim = np.random.randint(2,200)
    mu = np.random.rand(ndim)
    args = None
    kwargs = {'mu' : mu}
    wrapped = _FunctionWrapper(func, args, kwargs)
    x = np.random.rand(ndim)
    assert np.allclose(wrapped(x),func(x,mu))


def test_kwargs2(func=func2,seed=42):
    np.random.seed(seed)
    ndim = np.random.randint(2,200)
    mu = np.random.rand(ndim)
    ivar = 1.0 / np.random.rand(ndim)
    args = None
    kwargs = {'mu' : mu, 'ivar' : ivar}
    wrapped = _FunctionWrapper(func, args, kwargs)
    x = np.random.rand(ndim)
    assert np.allclose(wrapped(x),func(x,mu,ivar))


def test_argskwargs(func=func2,seed=42):
    np.random.seed(seed)
    ndim = np.random.randint(2,200)
    mu = np.random.rand(ndim)
    ivar = 1.0 / np.random.rand(ndim)
    args = [mu]
    kwargs = {'ivar' : ivar}
    wrapped = _FunctionWrapper(func, args, kwargs)
    x = np.random.rand(ndim)
    assert np.allclose(wrapped(x),func(x,mu,ivar))
