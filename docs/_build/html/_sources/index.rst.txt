.. title:: zeus documentation

.. figure:: ./../logo.png

**zeus is a pure-Python implementation of the Ensemble Slice Sampling method.**

- Fast & Robust *Bayesian Inference*,
- Efficient Markov Chain Monte Carlo,
- No hand-tuning,
- Excellent performance in terms of autocorrelation time and convergence rate,
- Scale to multiple CPUs without any extra effort.

.. image:: https://img.shields.io/badge/GitHub-minaskar%2Fzeus-blue
    :target: https://github.com/minaskar/zeus
.. image:: https://img.shields.io/badge/arXiv-2002.06212-red
    :target: https://arxiv.org/abs/2002.06212
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

    def log_prob(x, ivar):
        return - 0.5 * np.sum(ivar * x**2.0)

    nsteps, nwalkers, ndim = 1000, 100, 10
    ivar = 1.0 / np.random.rand(ndim)
    start = np.random.randn(nwalkers, ndim)

    sampler = zeus.sampler(nwalkers, ndim, log_prob, args=[ivar])
    sampler.run_mcmc(start, nsteps)


Installation
============

To install **zeus** using pip run::

    pip install zeus-mcmc


Getting Started
===============
- See the :doc:`cookbook` page to learn how to perform Bayesian Inference using **zeus**.
- See the :doc:`faq` page for frequently asked questions about zeus' operation.
- See the :doc:`api` page for detailed API documentation.


Attribution
===========

Please cite `Karamanis & Beutler (2020)
<https://arxiv.org/abs/2002.06212>`_ if you find this code useful in your
research. The BibTeX entry for the paper is::

  @article{zeus,
        title={Ensemble Slice Sampling},
        author={Minas Karamanis and Florian Beutler},
        year={2020},
        eprint={2002.06212},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
  }

Licence
=======

Copyright 2019-2020 Minas Karamanis and contributors.

zeus is free software made available under the GPL-3.0 License.


Changelog
=========

**1.1.0 (03/08/20)**

- Added ChainManager to deploy into supercomputing clusters, parallelizing both chains and walkers.
- Added Convergence diagnostic tools (Gelman-Rubin, Geweke).

**1.0.7 (11/05/20)**

- Improved parallel distribution of tasks


.. toctree::
   :maxdepth: 2
   :hidden:

   cookbook
   faq
   api
   notebooks/normal_distribution.ipynb
   notebooks/multiprocessing.ipynb
   notebooks/MPI.ipynb
   notebooks/datafit.ipynb
