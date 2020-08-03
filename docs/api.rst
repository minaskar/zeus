===
API
===

**zeus** consists mainly of three parts:

- The Sampler,
- A Suite of Convergence Diagnostics Tests (*Gelman-Rubin, Geweke, Autocorrelation Time*),
- The **Chain Manager** that allows you to deploy **zeus** into huge supercomputing clusters.


The Sampler
-----------

.. autoclass:: zeus.sampler
   :members:
   :undoc-members:


Convergence Diagnostics & Statistics
------------------------------------

.. autofunction:: zeus.diagnostics.GelmanRubin

.. autofunction:: zeus.diagnostics.Geweke

.. autofunction:: zeus.autocorr.AutoCorrTime


The Chain Manager
-----------------

.. autoclass:: zeus.parallel.ChainManager
   :members:
   :undoc-members: