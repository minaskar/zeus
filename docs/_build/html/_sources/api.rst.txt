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

.. autofunction:: zeus.GelmanRubin

.. autofunction:: zeus.Geweke

.. autofunction:: zeus.AutocorrelationTime


The Chain Manager
-----------------

.. autoclass:: zeus.ChainManager
   :members:
   :undoc-members: