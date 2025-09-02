regrank.regularizers.dictionary
===============================

.. py:module:: regrank.regularizers.dictionary


Attributes
----------

.. autoapisummary::

   regrank.regularizers.dictionary.logger


Classes
-------

.. autoapisummary::

   regrank.regularizers.dictionary.DictionaryRegularizer


Module Contents
---------------

.. py:data:: logger

.. py:class:: DictionaryRegularizer(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`regrank.regularizers.base_regularizer.BaseRegularizer`


   Implements a dictionary learning-based SpringRank regularizer.

   This method learns a sparse representation of the ranking vector `s` as `s ≈ DA`,
   where `D` is a learned dictionary and `A` are sparse codes. The objective
   balances the SpringRank loss with a reconstruction error and a sparsity penalty.

   Objective: E(s, D, A) = E_SpringRank(s) + (γ/2)||s - DA||² + λ||A||₁


   .. py:method:: fit(data: graph_tool.all.Graph, cfg: omegaconf.DictConfig) -> Dict[str, Any]

      Fits the dictionary learning SpringRank model using alternating optimization.

      :param data: The graph data.
      :param cfg: The Hydra configuration object for this run.

      :returns: A dictionary containing the primal solution (rankings), the learned
                dictionary, the sparse codes, and reconstruction error.
