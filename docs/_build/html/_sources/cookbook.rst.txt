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

.. raw:: html

    <style>
        .red {color:red; font-weight:bold;}
        .b {color:#0000FF; background-color:white;}
    </style>

.. role:: red

Convergence Diagnostics and Saving Progress recipes :red:`NEW`
==============================================================

- `Automated Convergence Diagnostics using the callback interface`_ :red:`NEW`
    In this recipe we are going to use the callback interface to monitor convergence and stop sampling automatically.

- `Saving progress to disk using h5py`_ :red:`NEW`
    In this recipe we are going to use the callback interface to save the samples and their corresponding log-probability values in a ``.h5`` file.

- `Tracking metadata using the blobs interface`_
    We introduce the blobs interface. An easy way for the user to track arbitrary metadata for every sample of the chain.

.. _Automated Convergence Diagnostics using the callback interface: notebooks/convergence.ipynb

.. _Saving progress to disk using h5py: notebooks/progress.ipynb

.. _Tracking metadata using the blobs interface: notebooks/blobs.ipynb



.. toctree::
    :maxdepth: 2
    :hidden:

    notebooks/normal_distribution.ipynb
    notebooks/datafit.ipynb
    notebooks/multimodal.ipynb
    notebooks/multiprocessing.ipynb
    notebooks/MPI.ipynb
    notebooks/blobs.ipynb
    notebooks/progress.ipynb
    notebooks/convergence.ipynb