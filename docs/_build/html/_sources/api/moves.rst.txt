==================
The Ensemble Moves
==================

``zeus`` was originally built on the ``Differential`` and ``Gaussian`` moves. Starting from version
2.0.0, ``zeus`` supports a mixture of different moves/proposals. Moves are recipes that the walkers
follow to cross the parameter space. The ``Differential Move`` remains the default choice but we also
provide a suite of additional moves, such as the ``Global Move`` that can be used when sampling from
challenging target distributions (e.g. highly dimensional multimodal distributions).

Differential Move
=================

.. autoclass:: zeus.moves.DifferentialMove
    :members:

Gaussian Move
=============

.. autoclass:: zeus.moves.GaussianMove
    :members:

Global Move
===========

.. autoclass:: zeus.moves.GlobalMove
    :members:

KDE Move
========

.. autoclass:: zeus.moves.KDEMove
    :members:

Random Move
===========

.. autoclass:: zeus.moves.RandomMove
    :members: