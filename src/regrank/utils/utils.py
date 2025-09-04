import itertools
import math
from logging import getLogger
from random import sample

import graph_tool.all as gt
import numpy as np
from numba import njit  # type: ignore
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

logger = getLogger(__name__)
# from numpy.random import default_rng
# import scipy.sparse


def generate_poset_from_graph(dag: gt.Graph) -> gt.Graph:
    """
    Generate a poset (partially ordered set) from a directed acyclic graph (DAG).

    Args:
        dag: A directed acyclic graph (DAG) represented as a graph-tool Graph object.

    Returns:
        A graph-tool Graph object representing the poset.
    """
    poset: gt.Graph = gt.Graph(directed=True)
    poset.add_vertex(dag.num_vertices())

    for e in dag.edges():
        u = e.source()
        v = e.target()
        if u != v:
            poset.add_edge(u, v)

    return poset


def generate_dag(
    num_vertices: int,
    num_edges: int | None = None,
    request: str | None = None,
    seed: int | None = 69,
) -> tuple[gt.Graph, dict[int, int]]:
    """
    Generate a Directed Acyclic Graph (DAG) with flexible structure.

    Args:
        num_vertices: Number of vertices in the graph (required).
        num_edges: Number of edges to add for a random DAG (optional).
        request: Type of combinatorial structure if num_edges is not given.
            Supported requests: 'complete', 'linear', 'star', 'antichain'.

    Returns:
        A graph-tool Graph object representing the DAG.

    Raises:
        ValueError: If input is inconsistent or unsupported.

    References (TBD):
        - [efficient random sampling of directed ordered acyclic graphs](https://arxiv.org/pdf/2303.14710)
    """
    if num_edges is None and request is None:
        raise ValueError(
            "If num_edges is not specified, you must provide a `request` "
            "for a specific DAG structure (e.g., 'complete', 'linear')."
        )

    g: gt.Graph = gt.Graph(directed=True)
    g.add_vertex(num_vertices)
    np.random.seed(seed)

    # Establish a random topological sort for all generation methods
    nodes: np.ndarray = np.arange(num_vertices)
    np.random.shuffle(nodes)
    node_map: dict[int, int] = {node: i for i, node in enumerate(nodes)}

    if num_edges is not None:
        # --- Generate a random DAG with a specific number of edges ---
        max_possible_edges = num_vertices * (num_vertices - 1) // 2
        if num_edges > max_possible_edges:
            raise ValueError(
                f"Cannot create a DAG with {num_edges} edges and {num_vertices} "
                f"vertices. Maximum possible is {max_possible_edges}."
            )

        density_threshold = math.log(num_vertices + 1e-9) * num_vertices

        if num_edges > density_threshold:
            # Dense graph strategy
            all_possible_edges = list(itertools.combinations(nodes, 2))
            indices = np.random.choice(
                len(all_possible_edges), num_edges, replace=False
            )
            for i in indices:
                u, v = all_possible_edges[i]
                g.add_edge(u, v)
        else:
            # Sparse graph strategy
            edges_added = 0
            while edges_added < num_edges:
                u, v = np.random.choice(num_vertices, 2, replace=False)
                if node_map[u] < node_map[v] and g.edge(u, v) is None:
                    g.add_edge(u, v)
                    edges_added += 1
        return g, node_map

    # --- Handle "request" for a specific combinatorial structure ---
    # We can safely ignore the type here because we've already checked that
    # request is not None if num_edges is None.
    request = request.lower()  # type: ignore

    if request == "complete":
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                g.add_edge(nodes[i], nodes[j])
        return g, node_map

    elif request == "linear":
        for i in range(num_vertices - 1):
            g.add_edge(nodes[i], nodes[i + 1])
        return g, node_map

    elif request == "star":
        center_node = nodes[0]
        for i in range(1, num_vertices):
            g.add_edge(center_node, nodes[i])
        return g, node_map

    elif request == "antichain":
        return g, node_map

    else:
        raise ValueError(
            f"Unsupported request '{request}'. Supported requests are: "
            "'complete', 'linear', 'star', and 'antichain'."
        )


def dag_to_bt_matrix(dag: gt.Graph, node_map: dict[int, float]) -> csc_matrix:
    """
    Converts a DAG into a sparse matrix based on the Bradley-Terry model.

    The value of the entry (i, j) is the probability that player i beats
    player j, calculated only if a directed edge from i to j exists in the DAG.

    Args:
        dag: A graph-tool Graph object representing the DAG.
        node_map: A dictionary mapping each vertex index (int) to its
                  skill score (float). The skill scores are used as the
                  parameters in the Bradley-Terry model.

    Returns:
        A SciPy csc_matrix where M[i, j] is the Bradley-Terry probability
        if edge (i, j) exists, and 0 otherwise.
    """
    num_vertices = dag.num_vertices()

    # Lists to store the data for the sparse matrix in coordinate format
    rows = []
    cols = []
    data = []

    # Iterate over all edges in the directed acyclic graph
    for edge in dag.edges():
        source_node = int(edge.source())
        target_node = int(edge.target())

        # Retrieve the skill scores from the node_map
        skill_i = node_map[source_node]
        skill_j = node_map[target_node]

        # (1) Calculate the Bradley-Terry probability using exponential scores
        exp_skill_i = np.exp(skill_i)
        exp_skill_j = np.exp(skill_j)

        # j beat i
        probability = exp_skill_j / (exp_skill_i + exp_skill_j)

        # Append the data for the sparse matrix
        rows.append(source_node)
        cols.append(target_node)
        data.append(probability)

        # i beat j
        probability = exp_skill_i / (exp_skill_i + exp_skill_j)

        # Append the data for the sparse matrix
        rows.append(target_node)
        cols.append(source_node)
        data.append(probability)

    # (2) Create the csc_matrix. By default, entries without an edge are 0.
    bt_matrix = csc_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices))

    return bt_matrix


def namedgraph_to_bt_matrix(named_graph, default_rank=0.0) -> csc_matrix:
    """
    Convert a NamedGraph with assigned rankings into a Bradley-Terry model matrix.

    Only generate BT probabilities for node pairs (i,j) where an edge exists between them.

    Args:
        named_graph: Instance of NamedGraph class with nodes that have assigned rankings
                     via the assign_ranking() method (stored in "true_ranking" vertex property)
        default_rank: Default ranking value for nodes without assigned rankings (default: 0.0)

    Returns:
        A scipy.sparse.csc_matrix of shape (N, N) where N is the number of vertices.
        M[i, j] represents the Bradley-Terry probability that node i beats node j:
        P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))
        where s_i is the ranking/skill score of node i.
        Only entries for connected node pairs are non-zero.
    """

    N = named_graph.num_vertices()
    vertex_list = list(named_graph.g.vertices())

    # Extract rankings for all nodes
    ranks = np.zeros(N)
    for idx, v in enumerate(vertex_list):
        if "true_ranking" in named_graph.g.vp:
            ranks[idx] = named_graph.g.vp["true_ranking"][v]
        else:
            ranks[idx] = default_rank

    # Prepare data for sparse matrix construction
    rows = []
    cols = []
    data = []

    # Only iterate over existing edges in the graph
    for edge in named_graph.g.edges():
        i = int(edge.source())
        j = int(edge.target())

        s_i = ranks[i]
        s_j = ranks[j]

        # Bradley-Terry probabilities
        exp_i = np.exp(s_i)
        exp_j = np.exp(s_j)

        # Probability that i beats j
        p_i_beats_j = exp_i / (exp_i + exp_j)
        rows.append(i)
        cols.append(j)
        data.append(p_i_beats_j)

        # Probability that j beats i
        p_j_beats_i = exp_j / (exp_i + exp_j)
        rows.append(j)
        cols.append(i)
        data.append(p_j_beats_i)

    # Create and return the sparse matrix
    bt_matrix = csc_matrix((data, (rows, cols)), shape=(N, N))
    return bt_matrix


def compute_spearman_correlation(g, s):
    return


def filter_by_time(g, time):
    mask_e = g.new_edge_property("bool")
    for edge in g.edges():
        if g.ep["etime"][edge] == time:
            mask_e[edge] = 1
        else:
            mask_e[edge] = 0
    return mask_e


def add_erroneous_edges(g, nid=0, times=1, method="single_point_mutation"):
    if method == "single_point_mutation":
        # scenario 1: add "single point mutation"
        for _ in range(int(times)):
            for node in range(g.num_vertices()):
                if node != nid:
                    # nid always endorsed by others
                    g.add_edge(node, nid)

    elif method == "random_edges":
        # scenario 2: add random edges
        for _ in range(int(times)):
            src, tar = sample(range(g.num_vertices()), 2)
            g.add_edge(src, tar)
    else:
        raise NotImplementedError(
            "method should be either `single_point_mutation` or `random_edges`."
        )
    return g


def D_operator(s):
    # output = np.zeros(n**2)
    n = len(s)
    output = np.zeros(n**2 - n, dtype=np.float64)  # if we avoid the zero rows
    k = 0
    for i in range(n):
        for j in range(i):
            output[k] = s[i] - s[j]
            k += 1
        # Avoid letting j = i since that's just s[i] - s[i] so it's a zero row
        for j in range(i + 1, n):
            output[k] = s[i] - s[j]
            k += 1
        # for j in range(n):
        #     # k = i + j*n
        #     output[k] = s[i] - s[j]
        #     k += 1
    return output


def D_operator_reg_t_sparse(a: csr_matrix, s):
    n = a.shape[0]
    output_t = np.zeros(n, dtype=np.float64)  # if we avoid the zero rows

    for ind in zip(*a.nonzero(), strict=False):
        i, j = ind[0], ind[1]
        if i < j:
            k = n * i + j - i - 1
        elif i > j:
            k = n * i + j - i
        else:
            continue

        output_t[i] += (a[i, j] ** 0.5) * s[k]
        output_t[j] -= (a[i, j] ** 0.5) * s[k]
    return output_t


@njit
def D_operator_reg_t(a, s):
    n = len(a)
    output_t = np.zeros(n, dtype=np.float64)  # if we avoid the zero rows
    k = 0
    for i in range(n):
        for j in range(i):
            output_t[i] += (a[i, j] ** 0.5) * s[k]
            output_t[j] -= (a[i, j] ** 0.5) * s[k]
            k += 1
        # Avoid letting j = i since that's just s[i] - s[i] so it's a zero row
        for j in range(i + 1, n):
            output_t[i] += (a[i, j] ** 0.5) * s[k]
            output_t[j] -= (a[i, j] ** 0.5) * s[k]
            k += 1
    return output_t


def D_operator_reg_sparse(a: csr_matrix, s):
    n = a.shape[0]
    output = np.zeros(n**2 - n, dtype=np.float64)  # if we avoid the zero rows
    for ind in zip(*a.nonzero(), strict=False):
        i, j = ind[0], ind[1]
        if i < j:
            k = n * i + j - i - 1
        elif i > j:
            k = n * i + j - i
        else:
            continue
        output[k] = (a[i, j] ** 0.5) * (s[i] - s[j])
    return output


@njit
def D_operator_reg(a, s):
    n = len(a)
    output = np.zeros(n**2 - n, dtype=np.float64)  # if we avoid the zero rows
    k = 0
    for i in range(n):
        for j in range(i):
            output[k] = (a[i, j] ** 0.5) * (s[i] - s[j])
            k += 1
        # Avoid letting j = i since that's just s[i] - s[i] so it's a zero row
        for j in range(i + 1, n):
            output[k] = (a[i, j] ** 0.5) * (s[i] - s[j])
            k += 1
    return output


def D_operator_b_sparse(a: csr_matrix):
    n = a.shape[0]
    output = np.zeros(n**2 - n, dtype=np.float64)  # if we avoid the zero rows
    for ind in zip(*a.nonzero(), strict=False):
        i, j = ind[0], ind[1]
        if i < j:
            k = n * i + j - i - 1
        elif i > j:
            k = n * i + j - i
        else:
            continue
        output[k] = a[ind] ** 0.5
    return output


@njit
def D_operator_b(a):
    n = len(a)
    output = np.zeros(n**2 - n, dtype=np.float64)  # if we avoid the zero rows
    # k = n
    k = 0
    for i in range(n):
        for j in range(i):
            output[k] = a[i, j] ** 0.5

            # raise Exception(i, j, k, n * i + j - i - 1)
            k += 1
        # Avoid letting j = i since that's just s[i] - s[i] so it's a zero row
        for j in range(i + 1, n):
            output[k] = a[i, j] ** 0.5
            k += 1
    return output


def implicit2explicit(f, a, m, n):
    """assumes f(x) is a linear operator (x has size n)
    so it can be represented f(x) = A*x for some matrix x
    (for now, assume A is square for simplicity)
    A = A * identity
    """
    e = np.zeros(n, dtype=np.float64)  # length n vector
    A = np.zeros((m, n), dtype=np.float64)  # (n ** 2 - n) x n matrix
    for i in range(n):
        # Loop over columns of identity
        e[i] = 1
        output = f(a, e)
        A[:, i] = output
        e[i] = 0
    return A
