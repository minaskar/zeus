.. title:: zeus documentation

.. figure:: ./../logo.png
    :scale: 30 %
    :align: center

**zeus is a Python implementation of the Ensemble Slice Sampling method.**

- Fast & Robust *Bayesian Inference*,
- Efficient *Markov Chain Monte Carlo (MCMC)*,
- Black-box inference, no hand-tuning,
- Excellent performance in terms of autocorrelation time and convergence rate,
- Scale to multiple CPUs without any extra effort.


.. image:: https://img.shields.io/badge/GitHub-minaskar%2Fzeus-blue
    :target: https://github.com/minaskar/zeus
.. image:: https://img.shields.io/badge/arXiv-2002.06212-red
    :target: https://arxiv.org/abs/2002.06212
.. image:: https://img.shields.io/badge/arXiv-2105.03468-brightgreen
    :target: https://arxiv.org/abs/2105.03468
.. image:: https://img.shields.io/badge/ascl-2008.010-blue.svg?colorB=262255
    :target: https://ascl.net/2008.010
.. image:: https://travis-ci.com/minaskar/zeus.svg?token=xnVWRZ3TFg1zxQYQyLs4&branch=master
    :target: https://travis-ci.com/minaskar/zeus
.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://github.com/minaskar/zeus/blob/master/LICENSE
.. image:: https://readthedocs.org/projects/zeus-mcmc/badge/?version=latest&token=4455dbf495c5a4eaba52de26ac56628aad85eb3eadc90badfd1703d0a819a0f9
    :target: https://zeus-mcmc.readthedocs.io/en/latest/?badge=latest
.. image:: https://pepy.tech/badge/zeus-mcmc
    :target: https://pepy.tech/project/zeus-mcmc

Basic use
=========

For instance, if you wanted to draw samples from a *10-dimensional Normal distribution*, you would do something like:

.. code:: Python

    import zeus
    import numpy as np

    def log_prob(x, ivar):
        return - 0.5 * np.sum(ivar * x**2.0)

    nsteps, nwalkers, ndim = 1000, 100, 10
    ivar = 1.0 / np.random.rand(ndim)
    start = np.random.randn(nwalkers, ndim)

    sampler = zeus.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
    sampler.run_mcmc(start, nsteps)
    chain = sampler.get_chain(flat=True)


Installation
============

To install ``zeus`` using ``pip`` run:

.. code:: bash

    pip install zeus-mcmc

To install ``zeus`` in a `[Ana]Conda <https://conda.io/projects/conda/en/latest/index.html>`_ environment use:

.. code:: bash

    conda install -c conda-forge zeus-mcmc


Getting Started
===============
- See the :doc:`cookbook` page to learn how to perform Bayesian Inference using ``zeus``.
- See the :doc:`faq` page for frequently asked questions about ``zeus``' operation.
- See the :doc:`api` page for detailed API documentation.


Citation
========

Please cite the following papers if you found this code useful in your
research::

    @article{karamanis2021zeus,
             title={zeus: A Python implementation of Ensemble Slice Sampling for efficient Bayesian parameter inference},
             author={Karamanis, Minas and Beutler, Florian and Peacock, John A},
             journal={arXiv preprint arXiv:2105.03468},
             year={2021}
            }

    @article{karamanis2020ensemble,
             title = {Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions},
             author = {Karamanis, Minas and Beutler, Florian},
             journal = {arXiv preprint arXiv: 2002.06212},
             year = {2020}
            }

Licence
=======

Copyright 2019-2021 Minas Karamanis and contributors.

``zeus`` is free software made available under the ``GPL-3.0 License``.


Changelog
=========

**2.3.1 (03/08/21)**

- Raise exception if model fails.

**2.3.0 (25/02/21)**

- Added ``sample`` method which advances the chain as a generator.
- Added ``light_mode``. When used, ``light_mode`` can significantly reduce the number of log likelihood evaluations
  and increase the general efficiency of the algorithm. ``light_mode`` works by performing no expansions after the
  end of the tuning phase. The scale factor is set to its opttimal value. This works best for approximately Gaussian
  distributions.
- Added ``start=None`` support for ``run_mcmc``. When used, the sampler proceeds from the last known position of the walkers.
- Added support for both ``thin`` and ``thin_by`` arguments.

**2.2.2 (21/02/21)**

- Added ``log_prob0`` and ``blobs0`` arguments in ``run``.
- Added ``get_last_sample()``, ``get_last_log_prob()`` and ``get_last_blobs()`` methods.

**2.2.0 (03/11/20)**

- Improved vectorization.

**2.1.1 (29/10/20)**

- Added ``blobs`` interface to track arbitrary metadata.
- Updated ``GlobalMove`` and multimodal example.
- Fixed minor bugs.

**2.0.0 (05/10/20)**

- Added new ``Moves`` interface (e.g. ``DifferentialMove``, ``GlobalMove``, etc).
- Plotting capabilities (i.e. ``cornerplot``).
- Updated docs.
- Fixed minor bugs.

**1.2.2 (19/09/20)**

- ``Sampler`` class is deprecated. New ``EnsembleSampler`` class in now available.
- New estimator for the Integrated Autocorrelation Time. It's accurate even with short chains.
- Updated ``ChainManager`` to handle thousands of CPUs.

**1.2.1 (04/08/20)**

- Changed to Flat-not-nested philosophy for diagnostics and ``ChainManager``.

**1.2.0 (03/08/20)**

- Extended ``ChainManager`` with ``gather``, ``scatter``, and ``bcast`` tools.

**1.1.0 (02/08/20)**

- Added ``ChainManager`` to deploy into supercomputing clusters, parallelizing both chains and walkers.
- Added Convergence diagnostic tools (Gelman-Rubin, Geweke).

**1.0.7 (11/05/20)**

- Improved parallel distribution of tasks


.. toctree::
    :maxdepth: 1
    :caption: Cookbook Recipes
    :hidden:

    Overview <cookbook>
    notebooks/normal_distribution.ipynb
    notebooks/datafit.ipynb
    notebooks/multiprocessing.ipynb
    notebooks/MPI.ipynb

.. toctree::
    :maxdepth: 3
    :caption: Help & Reference
    :hidden:

    faq
    api
