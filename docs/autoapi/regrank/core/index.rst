regrank.core
============

.. py:module:: regrank.core

.. autoapi-nested-parse::

   Core models for network ranking.

   This submodule provides the primary model implementations for the `regrank` package.
   The main recommended class is SpringRank.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/regrank/core/base/index
   /autoapi/regrank/core/losses/index


Classes
-------

.. autoapisummary::

   regrank.core.SpringRankLegacy


Package Contents
----------------

.. py:class:: SpringRankLegacy(*args, **kwargs)

   Bases: :py:obj:`base.SpringRankLegacy`


   DEPRECATED: Legacy implementation of the SpringRank model.

   .. deprecated:: 0.x.x
      Use :class:`regrank.core.SpringRank` instead. This class will be
      removed in a future version.
