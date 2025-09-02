regrank.regularizers.l1time
===========================

.. py:module:: regrank.regularizers.l1time


Attributes
----------

.. autoapisummary::

   regrank.regularizers.l1time.logger


Classes
-------

.. autoapisummary::

   regrank.regularizers.l1time.L1TimeRegularizer


Module Contents
---------------

.. py:data:: logger

.. py:class:: L1TimeRegularizer(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`regrank.regularizers.base_regularizer.BaseRegularizer`


   Implements a time-smoothing L1 regularizer for dynamic SpringRank.

   This method models the evolution of ranks over time by adding a penalty
   on the L1 norm of rank differences between consecutive time steps. It is
   solved using a dual-based proximal gradient descent algorithm.


   .. py:attribute:: sslc


   .. py:method:: fit(data: graph_tool.all.Graph, cfg: omegaconf.DictConfig) -> dict[str, Any]

      Fits the time-L1 regularized SpringRank model.

      :param data: The graph data, expected to have temporal information.
      :param cfg: The Hydra configuration object for this run.

      :returns: A dictionary containing the timewise rankings and solver history.
