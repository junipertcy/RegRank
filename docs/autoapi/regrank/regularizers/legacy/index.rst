regrank.regularizers.legacy
===========================

.. py:module:: regrank.regularizers.legacy


Attributes
----------

.. autoapisummary::

   regrank.regularizers.legacy.cp
   regrank.regularizers.legacy.logger


Classes
-------

.. autoapisummary::

   regrank.regularizers.legacy.LegacyRegularizer


Module Contents
---------------

.. py:data:: cp
   :value: None


.. py:data:: logger

.. py:class:: LegacyRegularizer(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`regrank.regularizers.BaseRegularizer`


   Implements the original SpringRank algorithm with multiple solver options.

   This class can solve the standard SpringRank objective using:
   1. A direct Lagrange multiplier approach (for alpha=0).
   2. A regularized linear system (for alpha > 0).
   3. CVXPY for a convex optimization formulation.
   4. LSMR for a least-squares formulation.


   .. py:attribute:: legacy_solvers


   .. py:method:: fit(data: graph_tool.all.Graph, cfg: omegaconf.DictConfig) -> dict[str, Any]

      Fits the legacy SpringRank model using the configured solver.

      :param data: The graph data.
      :param cfg: The Hydra configuration object for this run.

      :returns: A dictionary containing the primal solution (rankings) and,
                if applicable, the final objective value.
