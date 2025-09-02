regrank.regularizers.branch_constrained
=======================================

.. py:module:: regrank.regularizers.branch_constrained


Attributes
----------

.. autoapisummary::

   regrank.regularizers.branch_constrained.logger


Classes
-------

.. autoapisummary::

   regrank.regularizers.branch_constrained.BranchConstrainedRegularizer


Module Contents
---------------

.. py:data:: logger

.. py:class:: BranchConstrainedRegularizer(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`regrank.regularizers.base_regularizer.BaseRegularizer`


   Implements the branch-constrained SpringRank regularizer.

   This regularizer adds a squared hinge loss penalty to the SpringRank objective
   to guide the ranking solution towards a specific branch of the linear extension tree.

   Objective: E(r) = E_SpringRank(r) + λ Σₘ max(0, -dₘ(r_{iₘ} - r_{jₘ}))²


   .. py:attribute:: solver


   .. py:method:: fit(data: graph_tool.all.Graph, cfg: omegaconf.DictConfig) -> dict[str, Any]

      Fits the branch-constrained SpringRank model.

      :param data: The graph data.
      :param cfg: The Hydra configuration object for this run.

      :returns: A dictionary containing the primal solution (rankings), convergence
                status, final objective value, and constraint analysis.
