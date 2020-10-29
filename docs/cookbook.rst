========
Cookbook
========

MCMC Sampling recipes
=====================

- `Sampling from a multivariate Normal distribution`_
    Demonstrates how to sample from a correlated multivariate Gaussian distribution and how to perform
    the post-processing of the samples.

- `Fitting a model to data`_
    In this recipe we are going to produce some mock data and use them to illustrate how *zeus* works in
    realistic scenarios.

- `Sampling from multimodal distributions`_
    In this recipe we will demonstrate how one can use ``zeus`` with the ``Moves`` interface to sample
    efficiently from challenging high-dimensional multimodal distributions.

.. _Sampling from a multivariate Normal distribution: notebooks/normal_distribution.ipynb

.. _Fitting a model to data: notebooks/datafit.ipynb

.. _Sampling from multimodal distributions: notebooks/multimodal.ipynb


Parallelisation recipes
=======================

- `Multiprocessing`_
    Use many CPUs to sample from an expensive-to-evaluate probability distribution even faster.

- `MPI and ChainManager`_
    Distribute calculation to huge computer clusters.

.. _Multiprocessing: notebooks/multiprocessing.ipynb

.. _MPI and ChainManager: notebooks/MPI.ipynb


Saving Progress recipes
=======================

- `Tracking metadata using the blobs interface`_
    We introduce the blobs interface. An easy way for the user to track arbitrary metadata for every sample of the chain.

.. _Tracking metadata using the blobs interface: notebooks/blobs.ipynb

- Save progress using h5py. (soon)
    Save chains into a file.



Autocorrelation Analysis recipes
================================

- Measure the autocorrelation time and effective sample size of a chain (soon)
    This recipe demonstrates how to compute the autocorrelation time of a chain (i.e. a measure of
    the statistical independence of the samples). Having this we can also calculate the effective sample
    size of the chain.


.. toctree::
    :maxdepth: 2
    :hidden:

    notebooks/normal_distribution.ipynb
    notebooks/datafit.ipynb
    notebooks/multimodal.ipynb
    notebooks/multiprocessing.ipynb
    notebooks/MPI.ipynb
    notebooks/blobs.ipynb