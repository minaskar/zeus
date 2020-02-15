.. title:: zeus documentation

.. figure:: ./../logo.png

**zeus is a pure-Python implementation of the Ensemble Slice Sampling method.**

- Fast & Robust *Bayesian Inference*,
- No hand-tuning,
- Excellent performance in terms of autocorrelation time and convergence rate,
- Scale to multiple CPUs without any extra effort.

.. image:: https://img.shields.io/badge/GitHub-minaskar%2Fzeus-blue
    :target: https://github.com/minaskar/zeus
.. image:: https://travis-ci.com/minaskar/zeus.svg?token=xnVWRZ3TFg1zxQYQyLs4&branch=master
    :target: https://travis-ci.com/minaskar/zeus
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://github.com/minaskar/zeus/blob/master/LICENSE
.. image:: https://readthedocs.org/projects/zeus-mcmc/badge/?version=latest&token=4455dbf495c5a4eaba52de26ac56628aad85eb3eadc90badfd1703d0a819a0f9
    :target: https://zeus-mcmc.readthedocs.io/en/latest/?badge=latest

Basic usage
===========

For instance, if you wanted to draw samples from a 10-dimensional Gaussian, you would do something like::

    import numpy as np
    import zeus

    def logprob(x, ivar):
        return - 0.5 * np.sum(ivar * x**2.0)

    nsteps, nwalkers, ndim = 1000, 100, 10
    ivar = 1.0 / np.random.rand(ndim)
    start = np.random.randn(nwalkers, ndim)

    sampler = zeus.sampler(logprob, nwalkers, ndim, args=[ivar])
    sampler.run(start, nsteps)
    print(sampler.chain)


Installation
============

To install **zeus** using pip run::

    pip install zeus-mcmc


Getting Started
===============
- See the :doc:`cookbook` page to learn how to perform Bayesian Inference using **zeus**.
- See the :doc:`faq` page for frequently asked questions about zeus' operation.
- See the :doc:`api` page for detailed API documentation.


Licence
=======

Copyright 2019-2020 Minas Karamanis and contributors.

zeus is free software made available under the GPL-3.0 License.



.. toctree::
   :maxdepth: 2
   :hidden:

   cookbook
   faq
   api
   notebooks/normal_distribution.ipynb
   notebooks/multiprocessing.ipynb
   notebooks/MPI.ipynb
