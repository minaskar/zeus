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

Moreover, unlike other black-box MCMC methods (e.g. Affine Invariant Ensemble Sampler) **zeus** is
more robust to the curse of dimensionality. By construction, **zeus** converges to the true distribution
orders of magnitude faster than its competition. For instance, numerical tests indicate that in order
for the Affine Invariant Ensemble Sampler to match **zeus**' performance, it often requires 10-100
times more walkers and 10-1000 times more steps than **zeus**.

What are the walkers?
=====================

Walkers are the members of the ensemble and the explore the parameter space in parallel. Collectively,
the converge to the target distribution. 


How many walkers should I use?
==============================

As many as possible! But at least twice the number of parameters of your problem.
