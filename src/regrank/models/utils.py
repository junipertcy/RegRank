import itertools
import math

import graph_tool.all as gt
import numpy as np
from scipy.sparse import csc_matrix


def generate_dag(
    num_vertices: int, num_edges: int | None = None, request: str | None = None
) -> gt.Graph:
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
