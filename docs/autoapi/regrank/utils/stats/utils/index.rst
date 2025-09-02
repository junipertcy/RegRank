regrank.utils.stats.utils
=========================

.. py:module:: regrank.utils.stats.utils


Functions
---------

.. autoapisummary::

   regrank.utils.stats.utils.negacc
   regrank.utils.stats.utils.f_objective
   regrank.utils.stats.utils.compute_accuracy
   regrank.utils.stats.utils.betaLocal
   regrank.utils.stats.utils.betaGlobal


Module Contents
---------------

.. py:function:: negacc(A: numpy.ndarray, ranking: numpy.ndarray, beta: float) -> float

   Calculates the negative accuracy for beta optimization.


.. py:function:: f_objective(A: numpy.ndarray, ranking: numpy.ndarray, beta: float) -> float

   Objective function for global beta optimization.


.. py:function:: compute_accuracy(A: numpy.ndarray, ranking: numpy.ndarray, beta_local: float, beta_global: float) -> tuple[float, float]

   Computes local and global accuracy metrics.


.. py:function:: betaLocal(adj: scipy.sparse.csr_matrix, ranking: numpy.ndarray) -> float

   Finds the optimal local beta value.


.. py:function:: betaGlobal(adj: scipy.sparse.csr_matrix, ranking: numpy.ndarray) -> float

   Finds the optimal global beta value.
