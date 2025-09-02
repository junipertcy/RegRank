regrank.regularizers.annotated
==============================

.. py:module:: regrank.regularizers.annotated


Attributes
----------

.. autoapisummary::

   regrank.regularizers.annotated.logger


Classes
-------

.. autoapisummary::

   regrank.regularizers.annotated.AnnotatedRegularizer


Module Contents
---------------

.. py:data:: logger

.. py:class:: AnnotatedRegularizer(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`regrank.regularizers.base_regularizer.BaseRegularizer`


   Implements the annotated SpringRank regularizer.

   This method uses group annotations on nodes (the 'goi' or group of interest)
   to apply a regularization penalty that encourages nodes within the same group
   to have similar ranks. It is solved using a dual-based proximal gradient
   descent algorithm.


   .. py:attribute:: sslc


   .. py:method:: fit(data: graph_tool.all.Graph, cfg: omegaconf.DictConfig) -> dict[str, Any]

      Fits the annotated regularized SpringRank model.

      :param data: The graph data.
      :param cfg: The Hydra configuration object for this run.

      :returns: A dictionary containing the primal and dual solutions, solver history,
                and final primal and dual objective values.
