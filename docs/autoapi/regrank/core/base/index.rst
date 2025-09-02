regrank.core.base
=================

.. py:module:: regrank.core.base


Attributes
----------

.. autoapisummary::

   regrank.core.base.logger


Classes
-------

.. autoapisummary::

   regrank.core.base.BaseModel
   regrank.core.base.SpringRank
   regrank.core.base.SpringRankLegacy


Module Contents
---------------

.. py:data:: logger

.. py:class:: BaseModel(loss, cfg: omegaconf.DictConfig, reg=None)

   .. py:attribute:: loss


   .. py:attribute:: local_reg
      :value: None



   .. py:attribute:: cfg


   .. py:method:: compute_summary(g, goi, sslc=None, dual_v=None, primal_s=None)
      :staticmethod:



.. py:class:: SpringRank(cfg: omegaconf.DictConfig)

   Bases: :py:obj:`BaseModel`


   .. py:attribute:: regularizer


   .. py:attribute:: result


   .. py:method:: fit(data, **kwargs) -> dict[str, Any]

      Main fitting method - delegates to specific regularizers.



.. py:class:: SpringRankLegacy(alpha=0)

   .. py:attribute:: alpha
      :value: 0



   .. py:method:: fit_from_adjacency(adj)

      Fit SpringRank directly from adjacency matrix.



   .. py:method:: fit_scaled(data, scale=0.75)


   .. py:method:: fit(data)


   .. py:method:: get_ranks(A)

      params:
      - A: a (square) np.ndarray

      returns:
      - ranks, np.array

      TODO:
      - support passing in other formats (eg a sparse matrix)



   .. py:method:: get_inverse_temperature(A, ranks)

      given an adjacency matrix and the ranks for that matrix, calculates the
      temperature of those ranks



   .. py:method:: scale_ranks(ranks, scaling_factor)
      :staticmethod:



   .. py:method:: csr_SpringRank(A)
      :staticmethod:


      Main routine to calculate SpringRank by solving linear system
      Default parameters are initialized as in the standard SpringRank model

      :param A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)

      Output:
          rank: N-dim array, indeces represent the nodes' indices used in ordering the matrix A



   .. py:method:: compute_sr(A, alpha=0)

      Solve the SpringRank system.
      If alpha = 0, solves a Lagrange multiplier problem.
      Otherwise, performs L2 regularization to make full rank.

      :param A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)
      :param alpha: regularization term. Defaults to 0.

      Output:
          ranks: Solution to SpringRank



   .. py:method:: eqs39(beta, s, A)
