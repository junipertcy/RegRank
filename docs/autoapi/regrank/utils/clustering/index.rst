regrank.utils.clustering
========================

.. py:module:: regrank.utils.clustering


Functions
---------

.. autoapisummary::

   regrank.utils.clustering.determine_optimal_epsilon
   regrank.utils.clustering.cluster_1d_array


Module Contents
---------------

.. py:function:: determine_optimal_epsilon(arr, min_samples=2)

   Determines the optimal epsilon value for clustering using the k-distance graph method.

   Parameters:
   arr (np.ndarray): The input array of float values.
   min_samples (int): The number of nearest neighbors to consider.

   Returns:
   float: The optimal epsilon value.


.. py:function:: cluster_1d_array(arr, min_samples=2)

   Clusters a 1D array of float values such that adjacent values are grouped together.

   Parameters:
   arr (list or np.ndarray): The input array of float values.
   min_samples (int): The number of nearest neighbors to consider for determining epsilon.

   Returns:
   tuple: A tuple containing a list of clusters and a dictionary mapping each index of the input array to the index of the identified cluster.
