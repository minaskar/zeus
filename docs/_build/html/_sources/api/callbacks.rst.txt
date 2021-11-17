=============
The Callbacks
=============

Starting from version 2.4.0, ``zeus`` supports callback functions. Those are functions that are 
called in every iteration of a run. Among other things, these can be used to monitor useful quantities,
assess convergence, and save the chains to disk. Custom callback functions can also be used. Sampling
terminates if a callback function returns ``True`` and continues running while ``False`` or ``None`` is
returned.

Autocorrelation Callback
========================

.. autoclass:: zeus.callbacks.AutocorrelationCallback
    :members:


Split-R Callback
================

.. autoclass:: zeus.callbacks.SplitRCallback
    :members:


Parallel Split-R Callback
=========================

.. autoclass:: zeus.callbacks.ParallelSplitRCallback
    :members:


Minimum Iterations Callback
===========================

.. autoclass:: zeus.callbacks.MinIterCallback
    :members:


Save Progress Callback
======================

.. autoclass:: zeus.callbacks.SaveProgressCallback
    :members: