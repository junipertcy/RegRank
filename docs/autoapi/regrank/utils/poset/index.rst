regrank.utils.poset
===================

.. py:module:: regrank.utils.poset


Attributes
----------

.. autoapisummary::

   regrank.utils.poset.Edge
   regrank.utils.poset.EdgeList
   regrank.utils.poset.A


Functions
---------

.. autoapisummary::

   regrank.utils.poset.potential_poset_from_adjacency
   regrank.utils.poset.has_cycle
   regrank.utils.poset.break_cycles_exact
   regrank.utils.poset.is_linear_extension


Module Contents
---------------

.. py:data:: Edge

.. py:data:: EdgeList

.. py:function:: potential_poset_from_adjacency(A: numpy.ndarray | sage.all.Sequence[sage.all.Sequence[float]]) -> EdgeList

   Derive directed cover relations from the *skew-symmetric* component
   of a weighted adjacency matrix.

   For every unordered pair (i, j):

       •  If  R_ij > 0  ⇒  i ≺ j
       •  If  R_ij < 0  ⇒  j ≺ i

   :param A: Weighted adjacency matrix.
   :type A: (n, n) array-like (will be coerced to float)

   :returns: **covers** -- Minimal set of covers implied by the sign pattern.
   :rtype: list[(i, j)]


.. py:function:: has_cycle(edges: collections.abc.Iterable[Edge], n: int) -> bool

   Depth-first search for a directed cycle.

   :param edges: Directed edges.
   :type edges: iterable[(u, v)]
   :param n: Number of vertices, assumed labelled 0 … n-1.
   :type n: int

   :rtype: bool   – True iff a cycle exists.


.. py:function:: break_cycles_exact(covers: EdgeList, n: int, solver=None) -> tuple[EdgeList, EdgeList]

   Remove the *smallest* number of edges so the graph is acyclic.

   Uses a standard MILP formulation of the minimum feedback-arc set.

   :returns: **kept_edges, dropped_edges** -- Partition of `covers` such that `kept_edges` is acyclic and
             `dropped_edges` is the proven minimum set removed.
   :rtype: (list, list)


.. py:function:: is_linear_extension(covers: EdgeList, ordering: sage.all.Sequence[int]) -> bool

   True iff `ordering` is a linear extension of the poset defined by `covers`.


.. py:data:: A
