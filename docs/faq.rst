==========================
Frequently Asked Questions
==========================

What is the acceptance rate of zeus?
====================================

Unlike most MCMC methods, **zeus**' acceptance rate isn't varying during a run. As a matter of fact,
its acceptance rate is identically 1 always. This is because of the Slice Sampler at its core.


Why should I use zeus instead of other MCMC samplers?
=====================================================

The first reason you should think of using **zeus** is due to the fact that it doesn't require
any hand tuning at all. There is no need to adjust any hyperparameters or provide a proposal
distribution.

Moreover, unlike other black-box MCMC methods **zeus** is more robust to the curse of
 dimensionality and handle challenging distributions better.

What are the walkers?
=====================

Walkers are the members of the ensemble and the explore the parameter space in parallel. Collectively,
the converge to the target distribution.


How many walkers should I use?
==============================

As many as possible! But at least twice the number of parameters of your problem.
