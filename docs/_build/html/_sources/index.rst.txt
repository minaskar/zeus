.. title:: zeus documentation

.. image:: ./../zeus.gif

General-purpose MCMC sampler
============================

**zeus** is an efficient parallel Python implementation of the ensemble slice MCMC method.
The latter is a combination of Slice Sampling (Neal 2003) and Adaptive Direction Sampling (Gilks, Roberts, George 1993)
in a parallel ensemble framework with a few extra tricks. **The result is a fast and easy-to-use general-purpose
MCMC sampler that works out-of-the-box without the need to tune any hyperparameters or provide a proposal distribution.**
zeus produces independent samples of extremely low autocorrelation and converges to the target
distribution faster than any other general-purpose gradient-free MCMC sampler.

Basic usage
===========

For instance, if you wanted to draw samples from a 10-dimensional Gaussian, you would do::

    import numpy as np
    import zeus

    def logp(x, ivar):
        return - 0.5 * np.sum(ivar * x**2.0)

    nsteps, nwalkers, ndim = 1000, 100, 10
    ivar = 1.0 / np.random.rand(ndim)
    start = np.random.rand(ndim)

    sampler = zeus.sampler(logp, nwalkers, ndim, args=[ivar])
    sampler.run(start, nsteps)


Installation
============

To install **zeus** using pip run::

    pip install git+https://github.com/minaskar/zeus


Getting Started
===============
- See the :doc:`cookbook` page to learn how to perform Bayesian Inference using **zeus**.
- See the :doc:`faq` page for frequently asked questions about zeus' operation.
- See the :doc:`api` page for detailed API documentation.


Licence
=======

Copyright 2019-2019 Minas Karamanis and contributors.

zeus is free software made available under the GPL-3.0 License.



.. toctree::
   :maxdepth: 2
   :hidden:

   cookbook
   faq
   api
   notebooks/normal_distribution.ipynb
