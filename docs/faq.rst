==========================
Frequently Asked Questions
==========================

What is the acceptance rate of ``zeus``?
========================================

Unlike most MCMC methods, ``zeus`` acceptance rate isn't varying during a run. As a matter of fact,
its acceptance rate is identically 1 always. This is because of the Slice Sampler at its core.


Why should I use zeus instead of other MCMC samplers?
=====================================================

The first reason you should think of using ``zeus`` is due to the fact that it doesn't require
any hand tuning at all. There is no need to adjust any hyperparameters or provide a proposal
distribution.

Moreover, unlike other black-box MCMC methods ``zeus`` is more robust to the curse of
dimensionality and handle challenging distributions better.


What are the walkers?
=====================

Walkers are the members of the ensemble and the explore the parameter space in parallel. Collectively,
the converge to the target distribution.


How many walkers should I use?
==============================

At least twice the number of parameters of your problem. A good rule of thump is to use approximately
2.5 times the number of parameters. If your distribution has multiple modes you may want to increase
the number of walkers.


How should I initialize the positions of the walkers?
=====================================================

A good practice seems to be to initialize the walkers from a small ball close to the *Maximum a Posteriori*
estimate. After a few autocorrelation times the walkers would have explored the rest of the usefull regions
of the parameter space (a.k.a. the typical set), producing a great number of independent samples.


How long should I run ``zeus``?
===============================

You don't have to run ``zeus`` for very long. If your goal is to produce 2D/1D contours and/or 1-sigma/2-sigma
constraints for your parameters, running ``zeus`` for a few autocorrelation times (e.g. 10) is more than enough.


What are the ``Moves`` and which one should I use?
==================================================

``zeus`` was originally built on the ``Differential`` and ``Gaussian`` moves. Starting from version
2.0.0, ``zeus`` supports a mixture of different moves/proposals. Moves are recipes that the walkers
follow to cross the parameter space. The ``Differential Move`` remains the default choice but we also
provide a suite of additional moves, such as the ``Global Move`` that can be used when sampling from
challenging target distributions (e.g. highly dimensional multimodal distributions).

The move(s) you should use depends on the particular target distribution. The ``Differential Move`` 
seems to be a good choice for most distributions and 50-50 mixture of the ``Global Move`` and
``Local Move`` seem to perform very well in highly dimensional multimodal distributions when used
after the burnin period is over.