========
Cookbook
========

MCMC Sampling recipes
=====================

- `Sampling from a multivariate Normal distribution`_
    Demonstrates how to sample from a correlated multivariate Gaussian distribution and how to perform
    the post-processing of the samples.

- Fitting a model to data (soon)
    In this recipe we are going to produce some mock data and use them to illustrate how *zeus* works in
    realistic scenarios.

.. _Sampling from a multivariate Normal distribution: notebooks/normal_distribution.ipynb



Parallelisation recipes
=======================

- `Multiprocessing`_
    Use many CPUs to sample from an expensive-to-evaluate probability distribution even faster.

- `MPI`_
    Distribute calculation to huge computer clusters.

.. _Multiprocessing: notebooks/multiprocessing.ipynb

.. _MPI: notebooks/MPI.ipynb


Saving Progress recipes
=======================

- Save progress using h5py. (soon)
    Save chains into a file.



Autocorrelation Analysis recipes
================================

- Measure the autocorrelation time and effective sample size of a chain (soon)
    This recipe demonstrates how to compute the autocorrelation time of a chain (i.e. a measure of
    the statistical independence of the samples). Having this we can also calculate the effective sample
    size of the chain.
