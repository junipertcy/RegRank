regrank.regularizers.huber
==========================

.. py:module:: regrank.regularizers.huber


Attributes
----------

.. autoapisummary::

   regrank.regularizers.huber.cp
   regrank.regularizers.huber.logger


Classes
-------

.. autoapisummary::

   regrank.regularizers.huber.HuberRegularizer


Module Contents
---------------

.. py:data:: cp
   :value: None


.. py:data:: logger

.. py:class:: HuberRegularizer(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`regrank.regularizers.base_regularizer.BaseRegularizer`


   Implements SpringRank with a Huber loss regularizer.

   This method is robust to outliers in the comparison data. It is solved using
   the convex optimization framework CVXPY.


   .. py:method:: fit(data: graph_tool.all.Graph, cfg: omegaconf.DictConfig) -> dict[str, Any]

      Fits the Huber-regularized SpringRank model.

      :param data: The graph data.
      :param cfg: The Hydra configuration object for this run.

      :returns: A dictionary containing the primal solution (rankings) and the
                final primal objective value.
