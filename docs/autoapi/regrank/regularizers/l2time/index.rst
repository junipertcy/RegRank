regrank.regularizers.l2time
===========================

.. py:module:: regrank.regularizers.l2time


Attributes
----------

.. autoapisummary::

   regrank.regularizers.l2time.logger


Classes
-------

.. autoapisummary::

   regrank.regularizers.l2time.L2TimeRegularizer


Module Contents
---------------

.. py:data:: logger

.. py:class:: L2TimeRegularizer(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`regrank.regularizers.base_regularizer.BaseRegularizer`


   Implements a time-smoothing L2 regularizer for dynamic SpringRank.

   This method models the evolution of ranks over time by adding a penalty
   on the L2 norm of rank differences between consecutive time steps. It is
   solved directly as a global least-squares problem.


   .. py:method:: fit(data: graph_tool.all.Graph, cfg: omegaconf.DictConfig) -> dict[str, Any]

      Fits the time-L2 regularized SpringRank model.

      :param data: The graph data, expected to have temporal information.
      :param cfg: The Hydra configuration object for this run.

      :returns: A dictionary containing the timewise rankings.
