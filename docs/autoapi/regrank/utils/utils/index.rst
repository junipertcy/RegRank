regrank.utils.utils
===================

.. py:module:: regrank.utils.utils


Attributes
----------

.. autoapisummary::

   regrank.utils.utils.logger


Functions
---------

.. autoapisummary::

   regrank.utils.utils.generate_poset_from_graph
   regrank.utils.utils.generate_dag
   regrank.utils.utils.dag_to_bt_matrix
   regrank.utils.utils.namedgraph_to_bt_matrix
   regrank.utils.utils.compute_spearman_correlation
   regrank.utils.utils.filter_by_time
   regrank.utils.utils.add_erroneous_edges
   regrank.utils.utils.D_operator
   regrank.utils.utils.D_operator_reg_t_sparse
   regrank.utils.utils.D_operator_reg_t
   regrank.utils.utils.D_operator_reg_sparse
   regrank.utils.utils.D_operator_reg
   regrank.utils.utils.D_operator_b_sparse
   regrank.utils.utils.D_operator_b
   regrank.utils.utils.implicit2explicit


Module Contents
---------------

.. py:data:: logger

.. py:function:: generate_poset_from_graph(dag: graph_tool.all.Graph) -> graph_tool.all.Graph

   Generate a poset (partially ordered set) from a directed acyclic graph (DAG).

   :param dag: A directed acyclic graph (DAG) represented as a graph-tool Graph object.

   :returns: A graph-tool Graph object representing the poset.


.. py:function:: generate_dag(num_vertices: int, num_edges: int | None = None, request: str | None = None) -> graph_tool.all.Graph

   Generate a Directed Acyclic Graph (DAG) with flexible structure.

   :param num_vertices: Number of vertices in the graph (required).
   :param num_edges: Number of edges to add for a random DAG (optional).
   :param request: Type of combinatorial structure if num_edges is not given.
                   Supported requests: 'complete', 'linear', 'star', 'antichain'.

   :returns: A graph-tool Graph object representing the DAG.

   :raises ValueError: If input is inconsistent or unsupported.

   References (TBD):
       - [efficient random sampling of directed ordered acyclic graphs](https://arxiv.org/pdf/2303.14710)


.. py:function:: dag_to_bt_matrix(dag: graph_tool.all.Graph, node_map: dict[int, float]) -> scipy.sparse.csc_matrix

   Converts a DAG into a sparse matrix based on the Bradley-Terry model.

   The value of the entry (i, j) is the probability that player i beats
   player j, calculated only if a directed edge from i to j exists in the DAG.

   :param dag: A graph-tool Graph object representing the DAG.
   :param node_map: A dictionary mapping each vertex index (int) to its
                    skill score (float). The skill scores are used as the
                    parameters in the Bradley-Terry model.

   :returns: A SciPy csc_matrix where M[i, j] is the Bradley-Terry probability
             if edge (i, j) exists, and 0 otherwise.


.. py:function:: namedgraph_to_bt_matrix(named_graph, default_rank=0.0) -> scipy.sparse.csc_matrix

   Convert a NamedGraph with assigned rankings into a Bradley-Terry model matrix.

   Only generate BT probabilities for node pairs (i,j) where an edge exists between them.

   :param named_graph: Instance of NamedGraph class with nodes that have assigned rankings
                       via the assign_ranking() method (stored in "true_ranking" vertex property)
   :param default_rank: Default ranking value for nodes without assigned rankings (default: 0.0)

   :returns: A scipy.sparse.csc_matrix of shape (N, N) where N is the number of vertices.
             M[i, j] represents the Bradley-Terry probability that node i beats node j:
             P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))
             where s_i is the ranking/skill score of node i.
             Only entries for connected node pairs are non-zero.


.. py:function:: compute_spearman_correlation(g, s)

.. py:function:: filter_by_time(g, time)

.. py:function:: add_erroneous_edges(g, nid=0, times=1, method='single_point_mutation')

.. py:function:: D_operator(s)

.. py:function:: D_operator_reg_t_sparse(a, s)

.. py:function:: D_operator_reg_t(a, s)

.. py:function:: D_operator_reg_sparse(a, s)

.. py:function:: D_operator_reg(a, s)

.. py:function:: D_operator_b_sparse(a)

.. py:function:: D_operator_b(a)

.. py:function:: implicit2explicit(f, a, m, n)

   assumes f(x) is a linear operator (x has size n)
   so it can be represented f(x) = A*x for some matrix x
   (for now, assume A is square for simplicity)
   A = A * identity
